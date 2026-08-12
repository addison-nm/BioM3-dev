"""BioM3 multidomain finetuning: train a composed K-canvas ProteoScribe decoder.

Takes K per-family experts (trained beforehand by ``biom3_finetune_stage3``) and
trains a cross-domain coupling on top of them, optionally adapting the experts
themselves under a prior penalty. Input is a multidomain JSONL plus a curated
split manifest; per-domain captions are re-composed every epoch and embedded to
z_c on-device, so no precomputed embedding corpus is involved.

The pre-flight audit is not advisory. Before any weights move, the run asserts
that the coupling is at its additive null (so the composed model starts equal to
the independent experts), that the cross term is actually wired into the forward
pass, and that exactly the intended parameters are trainable. ``--audit_only``
runs those checks, writes the report, and exits.

    biom3_finetune_multidomain \
        --config_path configs/stage3_multidomain/finetune_multidomain_v1.json \
        --run_id my_multidomain_run
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import torch

import biom3.Stage3.cond_diff_transformer_layer as mod
from biom3.backend.device import set_float32_matmul_precision, setup_logger
from biom3.core.dataloaders import load_compose_plugins
from biom3.core.distributed import get_global_rank
from biom3.core.helpers import load_json_config, convert_to_namespace
from biom3.core.run_utils import setup_file_logging, teardown_file_logging
from biom3.Stage3.finetune_embedder import (
    build_protein_to_zp_embedder,
    build_text_to_zc_embedder,
)
from biom3.Stage3.multidomain import trainer as md_trainer
from biom3.Stage3.multidomain.audit import (
    assert_additive_null,
    audit_trainable_parameters,
    enforce_audit,
)
from biom3.Stage3.multidomain.data import MultiDomainDataModule
from biom3.Stage3.multidomain.io import (
    ALL_PAIRS,
    MultiDomainSpec,
    build_from_spec,
)
from biom3.Stage3.multidomain.PL_wrapper import (
    PRIOR_WEIGHT,
    PL_ProtARDM_Multidomain,
)
from biom3.Stage3.PL_wrapper import (
    alpha_spec_uses_zp,
    normalize_alpha_spec,
    resolve_eval_alpha,
)

logger = setup_logger(__name__)

_LOGS_SUBDIR = "logs"
_ARTIFACTS_SUBDIR = "artifacts"


def get_multidomain_args(parser):
    """Arguments specific to composed multidomain training."""
    parser.add_argument('--expert_weights', nargs='+', default=None,
                        help='one weights path per domain, in N->C order')
    parser.add_argument('--expert_sha256', nargs='+', default=None,
                        help='optional sha256 pin per expert, verified at load')
    parser.add_argument('--domain_ids', nargs='+', default=None,
                        help='domain labels for logging and the checkpoint spec')
    parser.add_argument('--num_domains', default=None, type=int,
                        help='K; must match expert_weights and the schema expect_k')
    parser.add_argument('--coupling_topology', default=ALL_PAIRS, type=str,
                        help='cross-domain coupling shape, recorded in the spec')
    parser.add_argument('--train_experts', default='False', type=str,
                        help='Component B: unfreeze the experts and adapt them')
    parser.add_argument('--expert_prior_lambda', default=0.0, type=float,
                        help='weight on ||W - W_ref||^2; 0 disables the prior')
    parser.add_argument('--prior_mode', default=PRIOR_WEIGHT, type=str,
                        choices=['weight', 'generation'],
                        help='weight-space penalty (cheap) or generation-space KL')
    parser.add_argument('--audit_additive_null', default='True', type=str,
                        help='assert the composed model starts equal to the experts')
    parser.add_argument('--audit_only', default=False, action='store_true',
                        help='run the pre-flight audit, write the report, and exit')
    parser.add_argument('--init_from', default=None, type=str,
                        help='initialise the whole composed model from a prior '
                             'multidomain checkpoint, then specialize')

    # Data
    parser.add_argument('--finetune_data_path', default=None, type=str,
                        help='multidomain JSONL of {sequence, source, domains: [...]}')
    parser.add_argument('--split_manifest_path', default=None, type=str,
                        help='curated split manifest (required)')
    parser.add_argument('--record_schema', default=None, type=str,
                        help='output schema; both outputs go through map_domains')
    parser.add_argument('--compose_plugins', default=None,
                        help='external .py files registering compose functions')
    parser.add_argument('--sequence_output_key', default='sequences', type=str)
    parser.add_argument('--caption_key', default='captions', type=str)
    parser.add_argument('--full_sequence_key', default='sequence', type=str,
                        help='record key holding the full-length protein; the '
                             'split-manifest fingerprint hashes this')
    parser.add_argument('--domains_key', default='domains', type=str)
    parser.add_argument('--lazy_records', default='False', type=str)

    # Frozen conditioning front-end
    parser.add_argument('--stage1_config_path', default=None, type=str)
    parser.add_argument('--stage2_config_path', default=None, type=str)
    parser.add_argument('--pencl_weights', default=None, type=str)
    parser.add_argument('--facilitator_weights', default=None, type=str)
    parser.add_argument('--train_alpha', default='zc', type=str)
    parser.add_argument('--eval_alpha', default='spread', type=str)
    parser.add_argument('--zp_batch_size', default=64, type=int)

    # Run
    parser.add_argument('--output_root', default='./outputs/multidomain', type=str)
    parser.add_argument('--checkpoints_folder', default='checkpoints', type=str)
    parser.add_argument('--runs_folder', default='runs', type=str)
    parser.add_argument('--run_id', default='multidomain_run', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--max_steps', default=-1, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr', default=2.5e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--precision', default='bf16', type=str)
    parser.add_argument('--devices_per_node', default=1, type=int)
    parser.add_argument('--num_nodes', default=1, type=int)
    parser.add_argument('--acc_grad_batches', default=1, type=int)
    parser.add_argument('--distributed_strategy', default=md_trainer.DEEPSPEED,
                        type=str)
    parser.add_argument('--log_every_n_steps', default=50, type=int)
    parser.add_argument('--limit_val_batches', default=1.0, type=float)
    parser.add_argument('--limit_train_batches', default=1.0, type=float,
                        help='fraction of the train loader per epoch; a small '
                             'value makes a smoke run reach validation and '
                             'checkpointing without a full pass')
    parser.add_argument('--save_top_k', default=3, type=int)
    parser.add_argument('--checkpoint_every_n_epochs', default=None, type=int)
    parser.add_argument('--float32_matmul_precision', default='high', type=str)
    parser.add_argument('--metrics_all_domains', default='False', type=str)
    parser.add_argument('--task', default='proteins', type=str)
    return parser


def parse_arguments(argv):
    """CLI > JSON config > defaults, mirroring the other Stage 3 runners."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config_path', '-c', type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(
        description='BioM3 Stage 3: composed multidomain finetuning')
    parser.add_argument('--config_path', '-c', type=str, default=None)
    mod.add_model_args(parser)
    parser.add_argument('--num_classes', default=29, type=int)
    parser.add_argument('--num_y_class_labels', default=6, type=int)
    parser.add_argument('--text_emb_dim', default=512, type=int)
    parser.add_argument('--image_size', default=32, type=int)
    parser.add_argument('--diffusion_steps', default=1024, type=int)
    get_multidomain_args(parser)

    if pre_args.config_path:
        config = load_json_config(pre_args.config_path)
        config.pop('description', None)
        config.pop('notes', None)
        config.pop('tags', None)
        known = {action.dest for action in parser._actions}
        unknown = set(config) - known
        if unknown:
            logger.warning("Ignoring unrecognised config keys: %s", sorted(unknown))
        parser.set_defaults(**{k: v for k, v in config.items() if k in known})

    args = parser.parse_args(argv)
    return _coerce_args(args)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('true', '1', 'yes')


def _coerce_args(args):
    """Normalize config-sourced values that argparse leaves as strings."""
    for flag in ('train_experts', 'audit_additive_null', 'lazy_records',
                 'metrics_all_domains'):
        setattr(args, flag, _as_bool(getattr(args, flag)))

    # Normalize the alpha names ("zc"/"zp"/"blend"/"spread") to the numeric form
    # the rest of the pipeline expects; alpha_spec_uses_zp cannot read the names.
    args.train_alpha = normalize_alpha_spec(args.train_alpha)
    args.eval_alpha = resolve_eval_alpha(args.eval_alpha)

    if isinstance(args.record_schema, str):
        args.record_schema = json.loads(args.record_schema)
    if args.compose_plugins:
        plugins = args.compose_plugins
        if isinstance(plugins, str):
            plugins = [plugins]
        load_compose_plugins(plugins)

    if args.expert_weights is None:
        raise ValueError("--expert_weights is required: one path per domain")
    if isinstance(args.expert_weights, str):
        args.expert_weights = [args.expert_weights]
    if args.num_domains is None:
        args.num_domains = len(args.expert_weights)
    if args.num_domains != len(args.expert_weights):
        raise ValueError(
            f"--num_domains={args.num_domains} but {len(args.expert_weights)} "
            "expert weight paths were given"
        )
    if args.domain_ids is None:
        args.domain_ids = [f"domain{i}" for i in range(args.num_domains)]
    if len(args.domain_ids) != args.num_domains:
        raise ValueError(
            f"--domain_ids has {len(args.domain_ids)} entries but K="
            f"{args.num_domains}"
        )
    if args.train_experts and args.expert_prior_lambda <= 0:
        logger.warning(
            "train_experts is on with expert_prior_lambda=%s: the experts are "
            "unconstrained and free to drift away from their single-domain fold",
            args.expert_prior_lambda,
        )
    return args


def load_embedder_configs(args):
    if not args.stage1_config_path or not args.stage2_config_path:
        raise ValueError(
            "multidomain finetuning needs --stage1_config_path and "
            "--stage2_config_path to build the frozen text->z_c front-end"
        )
    stage1_args = convert_to_namespace(load_json_config(args.stage1_config_path))
    stage2_args = convert_to_namespace(load_json_config(args.stage2_config_path))
    return stage1_args, stage2_args


def load_data(args, stage1_args):
    data_module = MultiDomainDataModule(
        jsonl_path=args.finetune_data_path,
        record_schema=args.record_schema,
        text_model_path=stage1_args.text_model_path,
        text_max_length=stage1_args.text_max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        diffusion_steps=args.diffusion_steps,
        image_size=args.image_size,
        num_domains=args.num_domains,
        split_manifest_path=args.split_manifest_path,
        sequence_key=args.sequence_output_key,
        caption_key=args.caption_key,
        full_sequence_key=args.full_sequence_key,
        domains_key=args.domains_key,
        lazy=args.lazy_records,
        needs_unique_sequences=alpha_spec_uses_zp(args.train_alpha)
        or alpha_spec_uses_zp(args.eval_alpha),
    )
    data_module.setup()
    return data_module


def _precompute_zp_lookup(args, stage1_args, data_module):
    """Embed every unique domain sequence once, then release ESM-2."""
    sequences = data_module.unique_sequences()
    logger.info("Precomputing z_p for %d unique domain sequences", len(sequences))
    embedder = build_protein_to_zp_embedder(
        stage1_args, args.pencl_weights, device=None)
    vectors = embedder.embed_sequences(sequences, batch_size=args.zp_batch_size)
    lookup = {seq: vectors[i].cpu() for i, seq in enumerate(sequences)}
    del embedder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return lookup


def load_model(args, stage1_args, stage2_args, data_module=None):
    if stage2_args.emb_dim != args.text_emb_dim:
        raise ValueError(
            f"Facilitator emb_dim={stage2_args.emb_dim} != text_emb_dim="
            f"{args.text_emb_dim}; z_c would not match the conditioning MLP"
        )

    spec = MultiDomainSpec.from_args(
        args, args.num_domains,
        domain_ids=tuple(args.domain_ids),
        coupling_topology=args.coupling_topology,
        train_experts=args.train_experts,
        expert_sources=tuple(args.expert_weights),
    )
    model = build_from_spec(
        spec, template_args=args,
        expert_weights=args.expert_weights,
        expert_sha256=args.expert_sha256,
    )

    if args.init_from:
        from biom3.Stage3.multidomain.io import (
            extract_composed_state_dict, load_composed_state_dict,
        )
        checkpoint = torch.load(args.init_from, map_location="cpu",
                                weights_only=False)
        load_composed_state_dict(
            model, extract_composed_state_dict(checkpoint), label=args.init_from)
        logger.info("Initialised the composed model from %s", args.init_from)

    for expert in model.experts:
        for param in expert.parameters():
            param.requires_grad_(args.train_experts)

    embedder = build_text_to_zc_embedder(
        stage1_args, stage2_args, args.pencl_weights, args.facilitator_weights)

    zp_lookup = None
    if data_module is not None and (alpha_spec_uses_zp(args.train_alpha)
                                    or alpha_spec_uses_zp(args.eval_alpha)):
        zp_lookup = _precompute_zp_lookup(args, stage1_args, data_module)

    PL_model = PL_ProtARDM_Multidomain(
        args, model, embedder, spec=spec, zp_lookup=zp_lookup,
        train_alpha=args.train_alpha, eval_alpha=args.eval_alpha,
        expert_prior_lambda=args.expert_prior_lambda,
        prior_mode=args.prior_mode,
        metrics_all_domains=args.metrics_all_domains,
    )
    return PL_model, spec


def run_preflight_audit(args, PL_model, artifacts_dir):
    """Additive null, wiring, and the trainable inventory. Raises on failure."""
    model = PL_model.model
    if args.audit_additive_null and not args.init_from:
        assert_additive_null(model)

    report = audit_trainable_parameters(
        model,
        train_experts=args.train_experts,
        optimizer=PL_model.configure_optimizers(),
    )
    report["coupling_topology"] = args.coupling_topology
    report["domain_ids"] = list(args.domain_ids)
    report["expert_sources"] = list(args.expert_weights)

    if get_global_rank() == 0:
        path = os.path.join(artifacts_dir, "multidomain_audit.json")
        with open(path, "w") as handle:
            json.dump(report, handle, indent=2, default=str)
        logger.info("Wrote audit report to %s", path)

    # init_from starts from a trained coupling, so the null no longer holds.
    enforce_audit(report, require_null_coupling=not args.init_from)
    return report


def main(args):
    start_time = datetime.now()
    warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
    logging.getLogger("tensorboardX.x2num").setLevel(logging.ERROR)

    run_dir = os.path.join(args.output_root, args.runs_folder, args.run_id)
    logs_dir = os.path.join(run_dir, _LOGS_SUBDIR)
    artifacts_dir = os.path.join(run_dir, _ARTIFACTS_SUBDIR)
    checkpoint_dir = os.path.join(
        args.output_root, args.checkpoints_folder, args.run_id)
    if get_global_rank() == 0:
        for path in (logs_dir, artifacts_dir, checkpoint_dir):
            os.makedirs(path, exist_ok=True)
    _, file_handler = setup_file_logging(artifacts_dir)

    set_float32_matmul_precision(args.float32_matmul_precision)
    seed = args.seed
    if seed <= 0:
        seed = int(np.random.randint(2 ** 31))
        args.seed = seed
    md_trainer.pl.seed_everything(seed, workers=True)
    logger.info("Using seed: %s", seed)

    try:
        stage1_args, stage2_args = load_embedder_configs(args)
        # The audit is purely a property of the model, so it does not pay for
        # loading the corpus, resolving the split manifest, or precomputing z_p.
        data_module = None if args.audit_only else load_data(args, stage1_args)
        PL_model, spec = load_model(args, stage1_args, stage2_args, data_module)

        run_preflight_audit(args, PL_model, artifacts_dir)
        if args.audit_only:
            logger.info("--audit_only: audit passed, exiting before training")
            return 0

        if get_global_rank() == 0:
            with open(os.path.join(artifacts_dir, "args.json"), "w") as handle:
                json.dump(vars(args), handle, indent=2, default=str)
            with open(os.path.join(artifacts_dir, "spec.json"), "w") as handle:
                json.dump(spec.to_dict(), handle, indent=2)

        trainer = md_trainer.build_trainer(
            args, checkpoint_dir=checkpoint_dir, logs_dir=logs_dir)
        resume = md_trainer.find_resume_checkpoint(checkpoint_dir)
        if resume:
            logger.info("Resuming from %s", resume)
        trainer.fit(PL_model, datamodule=data_module, ckpt_path=resume)

        elapsed = datetime.now() - start_time
        logger.info("Training finished in %s", elapsed)
        return 0
    finally:
        teardown_file_logging("biom3", file_handler)


if __name__ == "__main__":
    sys.exit(main(parse_arguments(sys.argv[1:])))
