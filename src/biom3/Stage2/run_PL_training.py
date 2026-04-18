#!/usr/bin/env python3

"""Training script for BioM3 Stage 2 (Facilitator).

Loads pre-computed Stage 1 embeddings (z_t, z_p) and trains the Facilitator
to align z_t to z_p. After fitting, runs a predict pass to emit the final
{text, protein, text_to_protein}-embedding dicts consumed by Stage 3.
"""

import argparse
import gc
import json
import logging
import os
import random
import socket
import sys
import warnings
from datetime import datetime

import numpy as np
import torch

from biom3.backend.device import BACKEND_NAME, _XPU, setup_logger

if BACKEND_NAME == _XPU:
    import lightning as pl
    from lightning import Trainer
    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
    from lightning.pytorch.callbacks import (
        LearningRateMonitor, DeviceStatsMonitor, EarlyStopping,
    )
else:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    from pytorch_lightning.callbacks import (
        LearningRateMonitor, DeviceStatsMonitor, EarlyStopping,
    )

from biom3.Stage2.PL_wrapper import PL_Facilitator
from biom3.Stage2.preprocess import Facilitator_DataModule
from biom3.core.helpers import load_json_config
from biom3.core.run_utils import (
    backup_if_exists, setup_file_logging, teardown_file_logging, write_manifest,
)
from biom3.backend.device import print_gpu_initialization, get_rank

logger = setup_logger(__name__)

_LOGS_SUBDIR = "logs"
_ARTIFACTS_SUBDIR = "artifacts"


def str_to_bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError("Input must be 'True' or 'False'")


def nonestr_to_none(s):
    if isinstance(s, str):
        if s.lower() == 'none':
            return None
        return s
    elif s is None:
        return None
    else:
        raise ValueError("Input must be string or 'None'")


def get_args(parser):
    parser.add_argument('--description', type=str, default="")
    parser.add_argument('--tags', type=str, nargs="*", default=[])
    parser.add_argument('--notes', type=str, nargs="*", default=[])

    parser.add_argument('--swissprot_data_path', type=str, default='None',
                        help='Path to Stage 1 SwissProt embeddings .pt dict.')
    parser.add_argument('--pfam_data_path', type=str, default='None',
                        help='Path to Stage 1 Pfam embeddings .pt dict.')
    parser.add_argument('--output_swissprot_dict_path', type=str, default=None,
                        help='Where to save Stage 2 SwissProt embeddings dict.')
    parser.add_argument('--output_pfam_dict_path', type=str, default=None,
                        help='Where to save Stage 2 Pfam embeddings dict.')

    parser.add_argument('--output_root', type=str, default='./outputs/Stage2/pretraining')
    parser.add_argument('--checkpoints_folder', type=str, default='checkpoints')
    parser.add_argument('--resume_from_checkpoint', type=str, default='None')
    parser.add_argument('--pretrained_weights', type=str, default='None')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--valid_size', type=float, default=0.2)
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    parser.add_argument('--limit_val_batches', type=float, default=1.0)
    parser.add_argument('--log_every_n_steps', type=int, default=10)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--scale_learning_rate', type=str, default='False')

    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'xpu', 'cpu'])
    parser.add_argument('--gpu_devices', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Alias exposed for Facilitator_Dataset device logic.')
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--acc_grad_batches', type=int, default=1)

    parser.add_argument('--save_metrics_history', type=str, default='True')
    parser.add_argument('--metrics_history_every_n_steps', type=int, default=1)
    parser.add_argument('--metrics_history_ranks', type=int, nargs='*', default=[0])
    parser.add_argument('--metrics_history_all_ranks_val_loss', type=str, default='False')

    parser.add_argument('--save_benchmark', type=str, default='False')
    parser.add_argument('--benchmark_skip_first_epoch', type=str, default='True')
    parser.add_argument('--benchmark_all_ranks_memory', type=str, default='False')

    parser.add_argument('--early_stopping_metric', type=str, default='None')
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0)
    parser.add_argument('--early_stopping_mode', type=str, default='min')

    parser.add_argument('--checkpoint_monitors', default=None)
    parser.add_argument('--checkpoint_every_n_steps', default=None)
    parser.add_argument('--checkpoint_every_n_epochs', default=None)
    parser.add_argument('--use_sync_safe_checkpoint', type=str, default='False')
    return parser


def get_model_args(parser):
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--hid_dim', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--loss_type', type=str, default='MSE',
                        choices=['MSE', 'MMD'])
    return parser


def get_path_args(parser):
    parser.add_argument('--run_id', default=None, type=str)
    parser.add_argument('--runs_folder', default='runs', type=str)
    return parser


def get_wrapper_args(parser):
    parser.add_argument('--wandb', type=str, default='False')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=[])
    return parser


def retrieve_all_args(args):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config_path', '-c', type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(args)

    parser = argparse.ArgumentParser(description='Stage 2: Facilitator training')
    parser.add_argument('--config_path', '-c', type=str, default=None)
    get_args(parser)
    get_model_args(parser)
    get_path_args(parser)
    get_wrapper_args(parser)

    if pre_args.config_path is not None:
        json_config = load_json_config(pre_args.config_path)
        parser.set_defaults(**json_config)

    args = parser.parse_args(args)

    args.scale_learning_rate = str_to_bool(args.scale_learning_rate)
    args.wandb = str_to_bool(args.wandb)
    args.save_metrics_history = str_to_bool(args.save_metrics_history)
    args.metrics_history_all_ranks_val_loss = str_to_bool(
        args.metrics_history_all_ranks_val_loss
    )
    args.use_sync_safe_checkpoint = str_to_bool(args.use_sync_safe_checkpoint)
    args.save_benchmark = str_to_bool(args.save_benchmark)
    args.benchmark_skip_first_epoch = str_to_bool(args.benchmark_skip_first_epoch)
    args.benchmark_all_ranks_memory = str_to_bool(args.benchmark_all_ranks_memory)

    args.swissprot_data_path = nonestr_to_none(args.swissprot_data_path)
    args.pfam_data_path = nonestr_to_none(args.pfam_data_path)
    args.resume_from_checkpoint = nonestr_to_none(args.resume_from_checkpoint)
    args.pretrained_weights = nonestr_to_none(args.pretrained_weights)
    args.early_stopping_metric = nonestr_to_none(args.early_stopping_metric)
    args.checkpoint_every_n_steps = nonestr_to_none(args.checkpoint_every_n_steps)
    args.checkpoint_every_n_epochs = nonestr_to_none(args.checkpoint_every_n_epochs)

    if args.run_id is None:
        args.run_id = f"stage2_{args.loss_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Facilitator_DataModule expects Rama's sentinel strings 'None', not Python None.
    # Re-inject the sentinels for the DataModule's branching logic.
    args.swissprot_data_path_sentinel = args.swissprot_data_path if args.swissprot_data_path is not None else 'None'
    args.pfam_data_path_sentinel = args.pfam_data_path if args.pfam_data_path is not None else 'None'
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def clear_gpu_cache():
    torch.set_float32_matmul_precision('medium')
    gc.collect()


_TRAINING_ENV_PREFIXES = (
    "CUDA_", "NCCL_", "TORCH_", "WANDB_",
    "MASTER_", "WORLD_SIZE", "RANK", "LOCAL_RANK",
    "SLURM_", "PBS_", "COBALT_", "PALS_", "PMI_", "OMPI_",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "ZE_", "CCL_", "ONEAPI_",
)

_SENSITIVE_ENV_SUBSTRINGS = (
    "KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH",
)


def _collect_training_env():
    env = {"hostname": socket.gethostname()}
    for key, val in sorted(os.environ.items()):
        if key.startswith(_TRAINING_ENV_PREFIXES):
            upper = key.upper()
            if any(s in upper for s in _SENSITIVE_ENV_SUBSTRINGS):
                continue
            env[key] = val
    return env


def load_data(args) -> Facilitator_DataModule:
    # Facilitator_DataModule branches on exact string 'None' for missing inputs.
    # Swap the Python None back to 'None' for the constructor.
    shim_args = argparse.Namespace(**vars(args))
    shim_args.swissprot_data_path = shim_args.swissprot_data_path_sentinel
    shim_args.pfam_data_path = shim_args.pfam_data_path_sentinel
    return Facilitator_DataModule(args=shim_args)


def build_pl_model(args) -> PL_Facilitator:
    return PL_Facilitator(args=args)


def load_pretrained_weights(PL_model, checkpoint_path: str):
    logger.info("Loading pretrained Facilitator weights from %s", checkpoint_path)
    sd = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
        sd = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in sd.items()}
    missing, unexpected = PL_model.model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Missing keys: %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys: %d", len(unexpected))
    return PL_model


def save_model(args, checkpoint_path, artifacts_path, PL_model, trainer):
    if not trainer.is_global_zero:
        return
    try:
        best_ckpt_fpath = trainer.checkpoint_callback.best_model_path
    except Exception:
        best_ckpt_fpath = None

    best_state_dict_fpath = os.path.join(checkpoint_path, 'state_dict.best.pth')
    backup_if_exists(best_state_dict_fpath)

    if best_ckpt_fpath and os.path.exists(best_ckpt_fpath):
        logger.info("Loading best checkpoint from %s", best_ckpt_fpath)
        ckpt = torch.load(best_ckpt_fpath, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict', ckpt)
        sd = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in sd.items()}
        torch.save(sd, best_state_dict_fpath)
        logger.info("Saved best state_dict to %s", best_state_dict_fpath)

        artifacts_copy = os.path.join(artifacts_path, 'state_dict.best.pth')
        backup_if_exists(artifacts_copy)
        torch.save(sd, artifacts_copy)
    else:
        logger.warning("No best checkpoint path reported; skipping save_model")


def get_final_embeddings(PL_model, final_dataloader, device='cuda', devices=1):
    logger.info("Running predict pass to collect z_t_to_p embeddings...")
    pred_trainer = Trainer(
        max_epochs=1,
        devices=devices,
        accelerator=device,
        enable_progress_bar=True,
        logger=False,
    )
    PL_model.eval()
    PL_model.text_to_protein_joint_embeddings = []
    pred_trainer.predict(PL_model, dataloaders=final_dataloader)
    return PL_model.text_to_protein_joint_embeddings


def save_embeddings(embedding_data, predictions, output_dict_path):
    n_text = len(embedding_data['text_embedding'])
    if n_text != predictions.shape[0]:
        raise AssertionError(
            f"Shape mismatch: text_embedding={n_text}, predictions={predictions.shape[0]}"
        )
    embedding_data['text_to_protein_embedding'] = [pred for pred in predictions]
    os.makedirs(os.path.dirname(os.path.abspath(output_dict_path)), exist_ok=True)
    torch.save(embedding_data, output_dict_path)
    logger.info("Saved Stage 2 embeddings to %s", output_dict_path)


def train_model(args, PL_model, data_module):
    logger.info("Beginning Stage 2 training...")
    output_root = args.output_root
    run_id = args.run_id
    gpu_devices = args.gpu_devices
    num_nodes = args.num_nodes
    acc_grad_batches = args.acc_grad_batches
    epochs = args.epochs
    precision = args.precision
    use_wandb = args.wandb

    if args.scale_learning_rate:
        n = num_nodes * gpu_devices
        logger.info("Scaling LR by %s", n)
        args.lr *= n

    checkpoint_dir = os.path.join(output_root, args.checkpoints_folder, run_id)
    run_dir = os.path.join(output_root, args.runs_folder, run_id)
    logs_dir = os.path.join(run_dir, _LOGS_SUBDIR)
    artifacts_dir = os.path.join(run_dir, _ARTIFACTS_SUBDIR)
    if get_rank() == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)

    loggers = [TensorBoardLogger(save_dir=logs_dir, version="")]
    if use_wandb:
        loggers.append(WandbLogger(
            name=args.wandb_name or run_id,
            save_dir=logs_dir,
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            tags=args.wandb_tags,
            group=run_id,
        ))

    callbacks = [LearningRateMonitor(logging_interval='step')]
    if args.device != 'cpu':
        callbacks.append(DeviceStatsMonitor())

    checkpoint_monitors = args.checkpoint_monitors
    if checkpoint_monitors is None:
        checkpoint_monitors = [{"metric": "valid_loss", "mode": "min"}]
    elif isinstance(checkpoint_monitors, str):
        checkpoint_monitors = json.loads(checkpoint_monitors)

    every_n_steps = args.checkpoint_every_n_steps
    checkpoint_callbacks = []
    for i, mon in enumerate(checkpoint_monitors):
        ckpt_kwargs = dict(
            dirpath=checkpoint_dir,
            verbose=True,
            monitor=mon["metric"],
            mode=mon["mode"],
            enable_version_counter=False,
        )
        if i == 0:
            ckpt_kwargs["save_top_k"] = 2
            ckpt_kwargs["save_last"] = "link"
            if every_n_steps is not None:
                ckpt_kwargs["every_n_train_steps"] = int(every_n_steps)
        else:
            metric_slug = mon["metric"].replace("/", "_")
            ckpt_kwargs["save_top_k"] = 1
            ckpt_kwargs["save_last"] = False
            ckpt_kwargs["filename"] = f"best-{metric_slug}-{{epoch}}"
        if args.use_sync_safe_checkpoint:
            from biom3.Stage3.callbacks import SyncSafeModelCheckpoint
            checkpoint_callbacks.append(SyncSafeModelCheckpoint(**ckpt_kwargs))
        else:
            from biom3.Stage3.callbacks import LoggingModelCheckpoint
            checkpoint_callbacks.append(LoggingModelCheckpoint(**ckpt_kwargs))
    callbacks = checkpoint_callbacks + callbacks

    if args.save_metrics_history:
        from biom3.Stage3.callbacks import MetricsHistoryCallback
        callbacks.append(MetricsHistoryCallback(
            output_dir=artifacts_dir,
            save_ranks=args.metrics_history_ranks,
            every_n_steps=args.metrics_history_every_n_steps,
            all_ranks_val_loss=args.metrics_history_all_ranks_val_loss,
        ))

    if args.save_benchmark:
        from biom3.Stage3.callbacks import TrainingBenchmarkCallback
        callbacks.append(TrainingBenchmarkCallback(
            output_dir=artifacts_dir,
            batch_size=args.batch_size,
            acc_grad_batches=acc_grad_batches,
            gpu_devices=gpu_devices,
            num_nodes=num_nodes,
            precision=precision,
            training_strategy='primary_only',
            num_workers=args.num_workers,
            skip_first_epoch=args.benchmark_skip_first_epoch,
            all_ranks_memory=args.benchmark_all_ranks_memory,
        ))

    if args.early_stopping_metric is not None:
        callbacks.append(EarlyStopping(
            monitor=args.early_stopping_metric,
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            mode=args.early_stopping_mode,
            verbose=True,
        ))

    if args.device == 'cuda' and (gpu_devices > 1 or num_nodes > 1):
        strategy = 'ddp'
    else:
        strategy = 'auto'

    if args.device == 'xpu' and isinstance(precision, str) and 'bf16' in precision:
        import types as _types
        _orig_training = PL_model.training_step
        _orig_validation = PL_model.validation_step

        def _bf16_training_step(_self, batch, batch_idx):
            with torch.autocast(device_type='xpu', dtype=torch.bfloat16):
                return _orig_training(batch, batch_idx)

        def _bf16_validation_step(_self, batch, batch_idx):
            with torch.autocast(device_type='xpu', dtype=torch.bfloat16):
                return _orig_validation(batch, batch_idx)

        PL_model.training_step = _types.MethodType(_bf16_training_step, PL_model)
        PL_model.validation_step = _types.MethodType(_bf16_validation_step, PL_model)
        trainer_precision = '32'
    else:
        trainer_precision = precision

    trainer = Trainer(
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        devices=gpu_devices,
        num_nodes=num_nodes,
        accelerator=args.device,
        strategy=strategy,
        accumulate_grad_batches=acc_grad_batches,
        logger=loggers,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        precision=trainer_precision,
        max_epochs=epochs,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=0,
    )

    if args.resume_from_checkpoint is None:
        logger.info("Train from scratch")
        trainer.fit(PL_model, data_module)
    else:
        logger.info("Resume from checkpoint: %s", args.resume_from_checkpoint)
        trainer.fit(PL_model, data_module, ckpt_path=args.resume_from_checkpoint)

    if get_rank() == 0:
        print_gpu_initialization()

    save_model(
        args=args,
        checkpoint_path=checkpoint_callbacks[0].dirpath,
        artifacts_path=artifacts_dir,
        PL_model=PL_model,
        trainer=trainer,
    )

    return trainer, artifacts_dir


def main(args):
    start_time = datetime.now()

    warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
    logging.getLogger("tensorboardX.x2num").setLevel(logging.ERROR)

    run_dir = os.path.join(args.output_root, args.runs_folder, args.run_id)
    logs_dir = os.path.join(run_dir, _LOGS_SUBDIR)
    artifacts_dir = os.path.join(run_dir, _ARTIFACTS_SUBDIR)
    checkpoint_dir = os.path.join(args.output_root, args.checkpoints_folder, args.run_id)
    if get_rank() == 0:
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)
    log_path, file_handler = setup_file_logging(artifacts_dir)

    clear_gpu_cache()

    seed = args.seed
    if seed <= 0:
        seed = np.random.randint(2 ** 32)
        args.seed = seed
    set_seed(seed)
    logger.info("Using seed: %s", seed)

    data_module = load_data(args)
    PL_model = build_pl_model(args)
    logger.info("Facilitator parameters: %s", sum(p.numel() for p in PL_model.model.parameters()))

    if args.pretrained_weights is not None and os.path.exists(args.pretrained_weights):
        PL_model = load_pretrained_weights(PL_model, args.pretrained_weights)
    elif args.pretrained_weights is not None:
        logger.warning("Pretrained weights path does not exist: %s", args.pretrained_weights)

    trainer, artifacts_dir = train_model(args, PL_model, data_module)

    # ---- Final predict pass + save transformed embeddings ----
    if get_rank() == 0:
        if args.swissprot_data_path_sentinel != 'None' and data_module.all_swiss_dataloader is not None:
            logger.info("Infer SwissProt dataset...")
            preds = get_final_embeddings(
                PL_model, data_module.all_swiss_dataloader,
                device=args.device, devices=1,
            )
            if args.output_swissprot_dict_path:
                save_embeddings(
                    data_module.swissprot_data, preds, args.output_swissprot_dict_path,
                )

        if args.pfam_data_path_sentinel != 'None' and data_module.all_pfam_dataloader is not None:
            logger.info("Infer Pfam dataset...")
            preds = get_final_embeddings(
                PL_model, data_module.all_pfam_dataloader,
                device=args.device, devices=1,
            )
            if args.output_pfam_dict_path:
                save_embeddings(
                    data_module.pfam_data, preds, args.output_pfam_dict_path,
                )

    # ---- Manifest + args.json ----
    if get_rank() == 0:
        elapsed = datetime.now() - start_time
        args_path = os.path.join(artifacts_dir, "args.json")
        backup_if_exists(args_path)
        with open(args_path, "w") as f:
            json.dump(vars(args), f, indent=2, default=str)

        total_params = sum(p.numel() for p in PL_model.model.parameters())
        outputs = {
            "seed": args.seed,
            "total_params": total_params,
            "batch_size": args.batch_size,
            "effective_lr": args.lr,
            "precision": args.precision,
            "gpu_devices": args.gpu_devices,
            "num_nodes": args.num_nodes,
            "acc_grad_batches": args.acc_grad_batches,
            "epochs": args.epochs,
            "loss_type": args.loss_type,
            "emb_dim": args.emb_dim,
            "hid_dim": args.hid_dim,
        }

        resolved_paths = {
            "checkpoint_dir": os.path.abspath(checkpoint_dir),
            "artifacts_dir": os.path.abspath(artifacts_dir),
        }
        if args.swissprot_data_path is not None:
            resolved_paths["swissprot_data_path"] = os.path.abspath(args.swissprot_data_path)
        if args.pfam_data_path is not None:
            resolved_paths["pfam_data_path"] = os.path.abspath(args.pfam_data_path)
        if args.output_swissprot_dict_path:
            resolved_paths["output_swissprot_dict_path"] = os.path.abspath(args.output_swissprot_dict_path)
        if args.output_pfam_dict_path:
            resolved_paths["output_pfam_dict_path"] = os.path.abspath(args.output_pfam_dict_path)
        if args.pretrained_weights is not None:
            resolved_paths["pretrained_weights"] = os.path.abspath(args.pretrained_weights)
        if args.resume_from_checkpoint is not None:
            resolved_paths["resume_from_checkpoint"] = os.path.abspath(args.resume_from_checkpoint)

        manifest_path = write_manifest(
            args, artifacts_dir, start_time, elapsed,
            outputs=outputs,
            resolved_paths=resolved_paths,
            environment=_collect_training_env(),
        )
        logger.info("Build manifest written to %s", manifest_path)

    teardown_file_logging("biom3", file_handler)


def parse_arguments(args):
    return retrieve_all_args(args)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
