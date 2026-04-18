#!/usr/bin/env python3

"""Training script for BioM3 Stage 1 (PenCL).

Follows the Stage 3 convention: config composition via JSON + _base_configs,
per-run output layout under {output_root}/{checkpoints_folder|runs_folder}/{run_id}/,
build manifest + args.json + run.log in the artifacts dir. Dispatches across
dataset_type in {default, masked, pfam, pfam_ablated}.
"""

import argparse
import contextlib
import gc
import io
import json
import logging
import os
import random
import socket
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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

import biom3.Stage1.preprocess as prep
import biom3.Stage1.model as mod
import biom3.Stage1.PL_wrapper as PL_mod
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
    parser.add_argument('--description', type=str, default="",
                        help='Free-text description of the run.')
    parser.add_argument('--tags', type=str, nargs="*", default=[],
                        help='Tags for the run.')
    parser.add_argument('--notes', type=str, nargs="*", default=[],
                        help='Notes for the run.')

    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to Swiss-Prot CSV.')
    parser.add_argument('--pfam_data_path', type=str, default='None',
                        help='Path to Pfam CSV (required for dataset_type=pfam).')
    parser.add_argument('--pfam_splits_dir', type=str, default=None,
                        help='Where to write per-rank Pfam shards. '
                             'Default: under the run directory.')
    parser.add_argument('--dataset_type', type=str, default='default',
                        choices=['default', 'masked', 'pfam', 'pfam_ablated'])
    parser.add_argument('--model_type', type=str, default='default',
                        choices=['default', 'pfam'],
                        help='Model class: default (PEN_CL) or pfam (pfam_PEN_CL). '
                             'Auto-set from dataset_type when unspecified.')

    parser.add_argument('--output_root', type=str, default='./outputs/Stage1/pretraining')
    parser.add_argument('--checkpoints_folder', type=str, default='checkpoints')
    parser.add_argument('--resume_from_checkpoint', type=str, default='None')
    parser.add_argument('--pretrained_weights', type=str, default='None')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--valid_size', type=float, default=0.1)
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    parser.add_argument('--limit_val_batches', type=float, default=1.0)
    parser.add_argument('--log_every_n_steps', type=int, default=10)

    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--head_lr', type=float, default=1e-3)
    parser.add_argument('--protein_encoder_lr', type=float, default=1e-4)
    parser.add_argument('--text_encoder_lr', type=float, default=1e-6)
    parser.add_argument('--choose_optim', type=str, default='AdamW')
    parser.add_argument('--scale_learning_rate', type=str, default='False')

    parser.add_argument('--precision', type=str, default='32',
                        help="Training precision: '32', '16', 'bf16', 'bf16-mixed'.")
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'xpu', 'cpu'])
    parser.add_argument('--gpu_devices', type=int, default=1)
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

    parser.add_argument('--checkpoint_monitors', default=None,
                        help='JSON list of {"metric","mode"} dicts, or None for default val_loss.')
    parser.add_argument('--checkpoint_every_n_steps', default=None)
    parser.add_argument('--checkpoint_every_n_epochs', default=None)
    parser.add_argument('--use_sync_safe_checkpoint', type=str, default='False')
    return parser


def get_model_args(parser):
    parser.add_argument('--seq_model_path', type=str,
                        default='./weights/LLMs/esm2_t33_650M_UR50D.pt')
    parser.add_argument('--pretrained_seq', type=str, default='True')
    parser.add_argument('--trainable_seq', type=str, default='True')
    parser.add_argument('--pLM_n_layers_to_finetune', type=int, default=1)
    parser.add_argument('--rep_layer', type=int, default=33)
    parser.add_argument('--protein_encoder_embedding', type=int, default=1280)

    parser.add_argument('--text_model_path', type=str,
                        default='./weights/LLMs/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    parser.add_argument('--pretrained_text', type=str, default='True')
    parser.add_argument('--trainable_text', type=str, default='True')
    parser.add_argument('--bLM_n_layers_to_finetune', type=int, default=1)
    parser.add_argument('--text_encoder_embedding', type=int, default=768)
    parser.add_argument('--text_max_length', type=int, default=512)

    parser.add_argument('--proj_embedding_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.8)

    parser.add_argument('--sequence_keyword', type=str, default='protein_sequence')
    parser.add_argument('--id_keyword', type=str, default='primary_Accession')
    return parser


def get_path_args(parser):
    parser.add_argument('--run_id', default=None, type=str,
                        help='Unique identifier for this training run.')
    parser.add_argument('--runs_folder', default='runs', type=str,
                        help='Subdirectory under output_root for per-run logs and artifacts.')
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

    parser = argparse.ArgumentParser(description='Stage 1: PenCL training')
    parser.add_argument('--config_path', '-c', type=str, default=None,
                        help='Path to JSON config file. CLI overrides JSON.')
    get_args(parser)
    get_model_args(parser)
    get_path_args(parser)
    get_wrapper_args(parser)

    if pre_args.config_path is not None:
        json_config = load_json_config(pre_args.config_path)
        parser.set_defaults(**json_config)

    args = parser.parse_args(args)

    # Type conversions (idempotent)
    args.pretrained_seq = str_to_bool(args.pretrained_seq)
    args.trainable_seq = str_to_bool(args.trainable_seq)
    args.pretrained_text = str_to_bool(args.pretrained_text)
    args.trainable_text = str_to_bool(args.trainable_text)
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

    args.data_path = nonestr_to_none(args.data_path) if args.data_path is not None else None
    args.pfam_data_path = nonestr_to_none(args.pfam_data_path)
    args.pfam_splits_dir = nonestr_to_none(args.pfam_splits_dir)
    args.resume_from_checkpoint = nonestr_to_none(args.resume_from_checkpoint)
    args.pretrained_weights = nonestr_to_none(args.pretrained_weights)
    args.early_stopping_metric = nonestr_to_none(args.early_stopping_metric)
    args.checkpoint_every_n_steps = nonestr_to_none(args.checkpoint_every_n_steps)
    args.checkpoint_every_n_epochs = nonestr_to_none(args.checkpoint_every_n_epochs)

    # Auto-derive model_type from dataset_type if user left it at the default
    if args.dataset_type in ('pfam', 'pfam_ablated') and args.model_type == 'default':
        logger.info("Auto-setting model_type='pfam' for dataset_type=%s", args.dataset_type)
        args.model_type = 'pfam'

    # Construct a default run_id if missing
    if args.run_id is None:
        args.run_id = f"stage1_{args.dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Default pfam_splits_dir under the run directory so writes don't land next to input
    if args.dataset_type in ('pfam', 'pfam_ablated') and args.pfam_splits_dir is None:
        args.pfam_splits_dir = os.path.join(
            args.output_root, args.runs_folder, args.run_id, 'pfam_splits',
        )

    return args


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def clear_gpu_cache():
    torch.set_float32_matmul_precision('medium')
    gc.collect()


def get_dataloaders_models(args):
    data_module_options = {
        'default': prep.Default_DataModule,
        'masked': prep.Default_DataModule,
        'pfam': prep.Pfam_DataModule,
        'pfam_ablated': prep.Pfam_DataModule,
    }
    data_module_class = data_module_options.get(args.dataset_type, prep.Default_DataModule)
    data_module = data_module_class(args=args)

    model_options = {
        'default': mod.PEN_CL,
        'pfam': mod.pfam_PEN_CL,
    }
    model_class = model_options.get(args.model_type, mod.PEN_CL)
    model = model_class(args=args)

    wrapper_options = {
        'default': PL_mod.PL_PEN_CL,
        'masked': PL_mod.mask_PL_PEN_CL,
        'pfam': PL_mod.pfam_PL_PEN_CL,
        'pfam_ablated': PL_mod.pfam_PL_PEN_CL,
    }
    wrapper_class = wrapper_options.get(args.dataset_type, PL_mod.PL_PEN_CL)

    PL_model = wrapper_class(
        args=args,
        model=model,
        text_tokenizer=model.text_encoder.tokenizer,
        sequence_tokenizer=model.protein_encoder.alphabet,
    )
    return data_module, PL_model


def load_pretrained_weights(PL_model, checkpoint_path: str):
    logger.info("Loading pretrained weights from %s", checkpoint_path)
    sd = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
        sd = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in sd.items()}
    missing, unexpected = PL_model.model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Missing keys when loading pretrained weights: %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys when loading pretrained weights: %d", len(unexpected))
    return PL_model


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
        logger.info("Copied best state_dict to %s", artifacts_copy)
    else:
        logger.warning("No best checkpoint path reported by trainer; skipping save_model")


def train_model(args, PL_model, data_module):
    logger.info("Beginning Stage 1 training...")

    output_root = args.output_root
    checkpoints_folder = args.checkpoints_folder
    runs_folder = args.runs_folder
    run_id = args.run_id
    log_every_n_steps = args.log_every_n_steps
    gpu_devices = args.gpu_devices
    num_nodes = args.num_nodes
    acc_grad_batches = args.acc_grad_batches
    epochs = args.epochs
    val_check_interval = args.val_check_interval
    limit_val_batches = args.limit_val_batches
    resume_from_checkpoint = args.resume_from_checkpoint
    precision = args.precision
    use_wandb = args.wandb

    if args.scale_learning_rate:
        n = num_nodes * gpu_devices
        logger.info(
            "Scaling LRs by num_nodes x gpu_devices = %s x %s = %s",
            num_nodes, gpu_devices, n,
        )
        args.head_lr *= n
        args.protein_encoder_lr *= n
        args.text_encoder_lr *= n

    checkpoint_dir = os.path.join(output_root, checkpoints_folder, run_id)
    run_dir = os.path.join(output_root, runs_folder, run_id)
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

    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [lr_monitor]
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

    # Strategy: plain DDP on CUDA, 'auto' elsewhere (XPU handles distributed init
    # externally; CPU-only smoke tests don't need DDP).
    if args.device == 'cuda' and (gpu_devices > 1 or num_nodes > 1):
        strategy = 'ddp'
    else:
        strategy = 'auto'

    # XPU manual bf16 autocast (Stage 3 / Rama pattern): PL's bf16 AMP plugin is
    # broken on Aurora, so we wrap training_step/validation_step ourselves.
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
        logger.info("Manual XPU bf16 autocast enabled; Trainer precision forced to '32'.")
    else:
        trainer_precision = precision

    trainer_params = {
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'enable_checkpointing': True,
        'devices': gpu_devices,
        'num_nodes': num_nodes,
        'accelerator': args.device,
        'strategy': strategy,
        'accumulate_grad_batches': acc_grad_batches,
        'logger': loggers,
        'log_every_n_steps': log_every_n_steps,
        'callbacks': callbacks,
        'precision': trainer_precision,
        'max_epochs': epochs,
        'val_check_interval': val_check_interval,
        'limit_val_batches': limit_val_batches,
        'num_sanity_val_steps': 0,
    }

    logger.info("Initializing Trainer with strategy=%s, precision=%s", strategy, trainer_precision)
    trainer = Trainer(**trainer_params)

    if resume_from_checkpoint is None:
        logger.info("Train from scratch")
        trainer.fit(PL_model, data_module)
    else:
        logger.info("Resume from checkpoint: %s", resume_from_checkpoint)
        trainer.fit(PL_model, data_module, ckpt_path=resume_from_checkpoint)

    if get_rank() == 0:
        print_gpu_initialization()

    save_model(
        args=args,
        checkpoint_path=checkpoint_callbacks[0].dirpath,
        artifacts_path=artifacts_dir,
        PL_model=PL_model,
        trainer=trainer,
    )


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

    data_module, PL_model = get_dataloaders_models(args=args)
    logger.info("PenCL parameters: %s", sum(p.numel() for p in PL_model.model.parameters()))

    if args.pretrained_weights is not None and os.path.exists(args.pretrained_weights):
        PL_model = load_pretrained_weights(PL_model, args.pretrained_weights)
    elif args.pretrained_weights is not None:
        logger.warning("Pretrained weights path does not exist: %s", args.pretrained_weights)

    train_model(args=args, PL_model=PL_model, data_module=data_module)

    if get_rank() == 0:
        elapsed = datetime.now() - start_time

        args_path = os.path.join(artifacts_dir, "args.json")
        backup_if_exists(args_path)
        with open(args_path, "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
        logger.info("Args written to %s", args_path)

        total_params = sum(p.numel() for p in PL_model.model.parameters())
        trainable_params = sum(
            p.numel() for p in PL_model.model.parameters() if p.requires_grad
        )

        outputs = {
            "seed": args.seed,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "batch_size": args.batch_size,
            "head_lr": args.head_lr,
            "protein_encoder_lr": args.protein_encoder_lr,
            "text_encoder_lr": args.text_encoder_lr,
            "precision": args.precision,
            "gpu_devices": args.gpu_devices,
            "num_nodes": args.num_nodes,
            "acc_grad_batches": args.acc_grad_batches,
            "epochs": args.epochs,
            "dataset_type": args.dataset_type,
            "model_type": args.model_type,
        }

        resolved_paths = {
            "checkpoint_dir": os.path.abspath(checkpoint_dir),
            "artifacts_dir": os.path.abspath(artifacts_dir),
        }
        if args.data_path is not None:
            resolved_paths["data_path"] = os.path.abspath(args.data_path)
        if args.pfam_data_path is not None:
            resolved_paths["pfam_data_path"] = os.path.abspath(args.pfam_data_path)
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
