"""BioM3 multidomain generation: decode proteins from a composed checkpoint.

Each prompt supplies one caption and one target length per domain. The captions
are embedded to z_c through the same frozen front-end training used, then the K
canvases are decoded in parallel by one composed forward per step, so every
domain conditions on its partners as they fill in.

Prompts come from one of two sources:

* ``--reference_records`` — accessions in the multidomain JSONL. Each record's
  per-domain captions and lengths are taken verbatim, which is the way to ask
  "what would the model write for *this* protein".
* ``--prompts_path`` — a JSONL of ``{"name", "captions": [...], "lengths": [...]}``
  for prompts that do not correspond to an existing record.

The model is always rebuilt by ``build_multidomain_from_checkpoint``, which reads
the architecture from the spec stored in the checkpoint rather than from a config
that may have moved on. Every consumer of a multidomain checkpoint goes through
that one function.

    biom3_sample_multidomain \
        --checkpoint outputs/.../last.ckpt \
        --finetune_data_path data/multidomain/luciferase_v1.jsonl \
        --reference_records A0A010RLF0 \
        --num_samples 10 --out generated.fasta
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer

from biom3.backend.device import get_device, setup_logger
from biom3.core.dataloaders import read_jsonl_records
from biom3.core.helpers import convert_to_namespace, load_json_config
from biom3.Stage3.finetune_embedder import build_text_to_zc_embedder
from biom3.Stage3.multidomain.io import build_multidomain_from_checkpoint
from biom3.Stage3.multidomain.preprocess import encode_captions
from biom3.Stage3.multidomain.sampling import (
    assemble_domains,
    decode_assemblies,
    generate_multidomain,
)

logger = setup_logger(__name__)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="BioM3 Stage 3: composed multidomain generation")
    parser.add_argument('--config_path', '-c', type=str, default=None,
                        help='JSON config supplying any of the options below')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='trained composed multidomain checkpoint')

    parser.add_argument('--finetune_data_path', type=str, default=None,
                        help='multidomain JSONL, for --reference_records')
    parser.add_argument('--reference_records', nargs='+', default=None,
                        help='accessions to condition on; omit with --num_records '
                             'to take the first N records')
    parser.add_argument('--num_records', type=int, default=None,
                        help='take the first N records of the JSONL as prompts')
    parser.add_argument('--prompts_path', type=str, default=None,
                        help='JSONL of {name, captions: [...], lengths: [...]}')
    parser.add_argument('--accession_key', type=str, default='accession')
    parser.add_argument('--domain_caption_key', type=str, default='caption')
    parser.add_argument('--domains_key', type=str, default='domains')

    parser.add_argument('--stage1_config_path', type=str, default=None)
    parser.add_argument('--stage2_config_path', type=str, default=None)
    parser.add_argument('--pencl_weights', type=str, default=None)
    parser.add_argument('--facilitator_weights', type=str, default=None)

    parser.add_argument('--num_samples', type=int, default=1,
                        help='sequences generated per prompt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--token_strategy', type=str, default='sample',
                        choices=['sample', 'argmax'])
    parser.add_argument('--allow_structural_tokens', action='store_true',
                        help='permit <START>/<END>/<PAD> at generated positions; '
                             'off by default so only residues are emitted')
    parser.add_argument('--no_couple', action='store_true',
                        help='zero-coupling ablation: decode each canvas by its '
                             'expert alone, the baseline any claim about the '
                             'coupling has to beat')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--linker', type=str, default='',
                        help='string joining the per-domain decodes; the default '
                             'concatenates, which is correct only while the '
                             'canvases are disjoint')

    parser.add_argument('--out', type=str, default=None,
                        help='output FASTA; a sibling .json holds the per-domain '
                             'breakdown')
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args(argv)
    if args.config_path:
        config = load_json_config(args.config_path)
        known = {action.dest for action in parser._actions}
        defaults = {k: v for k, v in config.items() if k in known}
        parser.set_defaults(**defaults)
        args = parser.parse_args(argv)

    if not args.checkpoint:
        raise ValueError("--checkpoint is required")
    if not args.out:
        raise ValueError("--out is required")
    if not (args.reference_records or args.num_records or args.prompts_path):
        raise ValueError(
            "supply prompts via --reference_records, --num_records, or "
            "--prompts_path"
        )
    return args


def load_prompts(args):
    """Resolve prompts to ``[{name, captions: [K], lengths: [K]}]``."""
    if args.prompts_path:
        prompts = []
        for i, record in enumerate(read_jsonl_records(args.prompts_path)):
            if len(record["captions"]) != len(record["lengths"]):
                raise ValueError(
                    f"{args.prompts_path} row {i}: {len(record['captions'])} "
                    f"captions but {len(record['lengths'])} lengths"
                )
            prompts.append({
                "name": record.get("name", f"prompt{i}"),
                "captions": list(record["captions"]),
                "lengths": [int(n) for n in record["lengths"]],
            })
        return prompts

    if not args.finetune_data_path:
        raise ValueError(
            "--reference_records / --num_records need --finetune_data_path")
    records = read_jsonl_records(args.finetune_data_path)

    if args.reference_records:
        wanted = list(args.reference_records)
        by_accession = {r[args.accession_key]: r for r in records}
        missing = [a for a in wanted if a not in by_accession]
        if missing:
            raise ValueError(
                f"{args.finetune_data_path} has no record(s) for {missing}")
        selected = [by_accession[a] for a in wanted]
    else:
        selected = records[:args.num_records]

    prompts = []
    for record in selected:
        domains = record[args.domains_key]
        prompts.append({
            "name": record[args.accession_key],
            "captions": [d[args.domain_caption_key] for d in domains],
            "lengths": [int(d.get("sequence_length", len(d["sequence"])))
                        for d in domains],
        })
    return prompts


def build_conditioning(args, prompts, stage1_args, stage2_args, device):
    """Embed every prompt's per-domain captions to z_c, ``[P, K, emb_dim]``."""
    embedder = build_text_to_zc_embedder(
        stage1_args, stage2_args, args.pencl_weights, args.facilitator_weights,
        device=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(stage1_args.text_model_path)

    flat = [caption for prompt in prompts for caption in prompt["captions"]]
    num_domains = len(prompts[0]["captions"])
    input_ids = encode_captions(tokenizer, flat, stage1_args.text_max_length)

    chunks = []
    with torch.no_grad():
        for start in range(0, input_ids.size(0), 64):
            chunk = input_ids[start:start + 64].to(device)
            chunks.append(embedder(chunk).cpu())
    z_c = torch.cat(chunks).reshape(len(prompts), num_domains, -1)
    del embedder
    return z_c


def write_outputs(args, results):
    """FASTA of assembled proteins, plus a JSON with the per-domain breakdown."""
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        for row in results:
            lengths = "|".join(str(len(s)) for s in row["domains"])
            handle.write(f">{row['name']}_s{row['replica']} "
                         f"len={len(row['sequence'])} domains={lengths}\n")
            handle.write(f"{row['sequence']}\n")

    json_path = os.path.splitext(args.out)[0] + ".json"
    with open(json_path, "w") as handle:
        json.dump(results, handle, indent=2)
    logger.info("Wrote %d sequences to %s (breakdown in %s)",
                len(results), args.out, json_path)


def main(args):
    device = args.device or get_device()
    torch.manual_seed(args.seed)

    stage1_args = convert_to_namespace(load_json_config(args.stage1_config_path))
    stage2_args = convert_to_namespace(load_json_config(args.stage2_config_path))

    model, spec = build_multidomain_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    logger.info("Loaded composed model: K=%d domains=%s topology=%s",
                spec.num_domains, list(spec.domain_ids), spec.coupling_topology)

    prompts = load_prompts(args)
    for prompt in prompts:
        if len(prompt["lengths"]) != spec.num_domains:
            raise ValueError(
                f"prompt {prompt['name']!r} has {len(prompt['lengths'])} domains "
                f"but the checkpoint was trained for {spec.num_domains}"
            )
    logger.info("Generating %d sample(s) for each of %d prompt(s)",
                args.num_samples, len(prompts))

    z_c = build_conditioning(args, prompts, stage1_args, stage2_args, device)
    seq_len = spec.image_size ** 2

    # One row per (prompt, replica); the seed is derived from both so a row is
    # reproducible regardless of how the batches happen to be packed.
    rows = [(p_idx, replica)
            for p_idx in range(len(prompts))
            for replica in range(args.num_samples)]

    results = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        batch_y = torch.stack([z_c[p_idx] for p_idx, _ in batch])
        batch_lengths = [prompts[p_idx]["lengths"] for p_idx, _ in batch]
        seeds = [args.seed + p_idx * 100_003 + replica for p_idx, replica in batch]
        generator = torch.Generator().manual_seed(seeds[0])

        states, _ = generate_multidomain(
            model, batch_y, batch_lengths,
            seq_len=seq_len, device=device,
            token_strategy=args.token_strategy,
            restrict_to_residues=not args.allow_structural_tokens,
            couple=not args.no_couple,
            sample_seeds=seeds,
            generator=generator,
        )
        for (p_idx, replica), domains in zip(batch, decode_assemblies(states.cpu())):
            results.append({
                "name": prompts[p_idx]["name"],
                "replica": replica,
                "requested_lengths": prompts[p_idx]["lengths"],
                "domains": domains,
                "sequence": assemble_domains(domains, linker=args.linker),
                "coupled": not args.no_couple,
            })
        logger.info("  %d/%d", len(results), len(rows))

    write_outputs(args, results)
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_arguments(sys.argv[1:])))
