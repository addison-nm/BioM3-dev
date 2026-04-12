#!/usr/bin/env python3

"""Training script for BioM3 Stage 3

Support for PyTorch Lightning and Weights&Biases

"""

import sys
import os
import io
import json
import shutil
import contextlib
import socket
import logging
import warnings
import numpy as np
import random
import gc
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ----- Retrieve available device -----
from biom3.backend.device import BACKEND_NAME, _XPU, setup_logger

# Import pytorch lightning based on device
if BACKEND_NAME == _XPU:
    # lightning imports (from local installation)
    import lightning as pl
    from lightning import Trainer
    from lightning.fabric.strategies import DeepSpeedStrategy
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, EarlyStopping
    # Additional necessities
    from lightning.pytorch.plugins.environments import ClusterEnvironment, MPIEnvironment
    from lightning.pytorch.utilities.model_summary import ModelSummary
else:
    # PyTorch Lightning imports
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DeepSpeedStrategy
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, EarlyStopping
    # Additional necessities
    from pytorch_lightning.plugins.environments import ClusterEnvironment

# DeepSpeed is needed for the DeepSpeedStrategy
import deepspeed

# WandB
import wandb

# Custom modules
import biom3.Stage3.preprocess as prep
import biom3.Stage3.cond_diff_transformer_layer as mod
import biom3.Stage3.PL_wrapper as PL_mod
from biom3.Stage3.io import prepare_model_ProteoScribe
from biom3.core.helpers import load_json_config
from biom3.core.run_utils import (
    backup_if_exists, setup_file_logging, teardown_file_logging, write_manifest,
)
from biom3.backend.device import print_gpu_initialization, get_device, get_rank

logger = setup_logger(__name__)

_LOGS_SUBDIR = "logs"
_ARTIFACTS_SUBDIR = "artifacts"


def get_args(parser):
    """
    Configure argument parser with all training, model, and data parameters.
    
    This function adds a comprehensive set of arguments to the provided parser,
    including data paths, training hyperparameters, checkpointing options,
    model architecture settings, diffusion parameters, and dataset configurations.
    It serves as the central configuration point for the entire training pipeline.
    
    Args:
        parser: ArgumentParser object to which arguments will be added
        
    Returns:
        The parser with the complete set of arguments added
    """
    parser.add_argument('--description', default="", type=str,
                        help='human-readable description of this config (stored in args.json)')
    parser.add_argument('--tags', type=str, nargs='+', default=[],
                        help='tags for categorizing this run (stored in args.json)')
    parser.add_argument('--notes', type=str, nargs='+', default=[],
                        help='free-form notes about this run (stored in args.json)')

    parser.add_argument('--data_root', default="./data/ARDM_temp_homolog_family_dataset.csv", type=Path,
                        help='path to dataset root directory')

    parser.add_argument('--output_root', default=None, type=str,
                        help='base directory for all training outputs')
    parser.add_argument('--checkpoints_folder', default=None, type=str,
                        help='subdirectory under output_root for checkpoints')
    parser.add_argument('--resume_from_checkpoint', default='None', type=str,
                        help='checkpoint path to last model iteration (usually last.ckpt)')


    parser.add_argument('--dataset', default="normal", type=str,
                        choices=['normal', 'sequence'],
                        help='which dataset to train on')
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loader workers')
    parser.add_argument('--warmup_steps', default=500, type=int,
                        help='number of learning rate warmup steps')
    parser.add_argument('--total_steps', default=1000, type=int,
                        help='total number of steps of minibatch gradient descent')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='mini-batch size')
    parser.add_argument('--weight_decay', default=1e-6, type=float,
                        help='weight decay')
    parser.add_argument('--lr', default=3e-4, type=float,
                        help='learning rate')
    parser.add_argument('--ema_inv_gamma', default=1.0, type=float,
                        help='inverse gamma parameter for exponential moving average')
    parser.add_argument('--ema_power', default=0.75, type=float,
                        help='power parameter for exponential moving average')
    parser.add_argument('--ema_max_value', default=0.999, type=float,
                        help='max value parameter for exponential moving average')
    parser.add_argument('--precision', default='no', type=str, choices=['no', 'fp16', 'bf16', '32'],
                        help='whether to use 16-bit or 32-bit training')
    parser.add_argument('--seed', default=0, type=int,
                        help='random number seed')
    parser.add_argument('--checkpoint_dir', default='./checkpoint/', type=Path,
                        help='path to checkpoint directory')
    parser.add_argument('--checkpoint_prefix', default='channels',
                        help='prefix for local checkpoint')
    parser.add_argument('--device', default='cuda', type=str, 
                        choices=["cpu", "cuda", "xpu"],
                        help='computational device')
    parser.add_argument('--model_option', default='transformer', type=str,
                        choices=['Unet', 'transformer'],
                        help='Choose model architecture')
    parser.add_argument('--download', default='True', type=str,
                        help='Download dataset')
    # Dataset paths (generalized)
    parser.add_argument('--primary_data_path', default='None', type=str,
                        help='path to primary training HDF5 dataset')
    parser.add_argument('--secondary_data_paths', default=None, type=str, nargs='+',
                        help='one or more paths to secondary HDF5 datasets')
    parser.add_argument('--training_strategy', default='auto', type=str,
                        choices=['auto', 'primary_only', 'combine'],
                        help='data handling strategy (auto: primary_only if no secondary, combine otherwise)')
    # Deprecated aliases (mapped to primary/secondary in retrieve_all_args)
    parser.add_argument('--swissprot_data_root', default='None', type=str,
                        help='(deprecated, use --primary_data_path) path to SwissProt data')
    parser.add_argument('--pfam_data_root', default='None', type=str,
                        help='(deprecated, use --secondary_data_paths) path to Pfam data')

    parser.add_argument('--pretrained_weights', default='None', type=str,
                        help='path to .bin weight or checkpoint file containing model weights')
    
    parser.add_argument('--scale_learning_rate', default='True', type=str,
                        help='scale the specified learning rate by the number of devices')
    
    # Finetuning
    parser.add_argument('--finetune', default='False', type=str,
                        help='flag to run finetuning')
    parser.add_argument('--finetune_last_n_blocks', default=-2, type=int,
                        help='Number of last transformer blocks to finetune (-1: finetune all blocks, 0: no blocks)')
    parser.add_argument('--finetune_last_n_layers', default=-2, type=int,
                        help='Number of last transformer layers to finetune (-1: finetune all layers, 0: no layers)')
    parser.add_argument('--finetune_output_layers', default="True", type=str,
                        help='Whether to finetune the transformer output layers (norm and out)')

    # diffusion param
    parser.add_argument('--diffusion_steps', default=256, type=int,
                        help='number of timesteps, should be as long as the sequence')
    parser.add_argument('--task', default='MNIST', type=str,
                        help='problem system: MNIST or proteins')
    parser.add_argument('--enter_eval', default=1000, type=int,
                        help='iteration step to evaluation performance.')


    # number of epochs
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of epochs for training...')
    parser.add_argument('--sequence_keyname', default='seq', type=str,
                        help='key name that belongs to the sequence list..')
    parser.add_argument('--facilitator', default='None', type=str,
                choices=['MSE', 'MMD', 'Default', 'None'],
                help='Option whether facilitator was used')
    parser.add_argument('--valid_size', default=0.1, type=float,
                        help='Validation dataset size...')
    parser.add_argument('--num_workers', default=0, type=int,  #  NOTE: CHANGED TO 0 TO PREVENT CUDA OOM ERROR
                        help='Number of dataloader workers...')

    # training on pfam database...
    parser.add_argument('--max_steps', default=100000, type=int,
                        help='number of iteration steps for training...')
    parser.add_argument('--val_check_interval', default=10000, type=int,
                        help='number of steps before starting evaluation on validation...')
    parser.add_argument('--limit_val_batches', default=0.05, type=float,
                        help='number of samples to validate on...')
    parser.add_argument('--log_every_n_steps', default=10001, type=float,
                        help='number of samples to validate on...')
    parser.add_argument('--start_secondary', default='False', type=str,
                        help='flag for phase transition: load primary weights then train on primary+secondary')
    # Deprecated alias
    parser.add_argument('--start_pfam_trainer', default='False', type=str,
                        help='(deprecated, use --start_secondary)')

    # Metrics history
    parser.add_argument('--save_metrics_history', default='True', type=str,
                        help='Save training/validation metrics history to artifacts dir')
    parser.add_argument('--metrics_history_ranks', type=int, nargs='+', default=[0],
                        help='Rank indices on which to save metrics history')
    parser.add_argument('--metrics_history_every_n_steps', default=1, type=int,
                        help='Record training metrics every N global steps')
    parser.add_argument('--metrics_history_all_ranks_val_loss', default='False',
                        type=str,
                        help='Diagnostic: dump val_loss per rank at epoch end '
                             '(one file per rank) to check sync_dist consistency')

    # Early stopping
    parser.add_argument('--early_stopping_metric', default=None, type=str,
                        help='Metric to monitor for early stopping (e.g. val_loss). None to disable.')
    parser.add_argument('--early_stopping_patience', default=10, type=int,
                        help='Number of checks with no improvement before stopping')
    parser.add_argument('--early_stopping_min_delta', default=0.0, type=float,
                        help='Minimum change to qualify as an improvement')
    parser.add_argument('--early_stopping_mode', default='min', type=str,
                        choices=['min', 'max'],
                        help='Whether to minimize or maximize the monitored metric')

    # Periodic checkpoint saving
    parser.add_argument('--checkpoint_every_n_steps', default=None, type=int,
                        help='Save checkpoints every N training steps (in addition to best-metric saves)')
    parser.add_argument('--checkpoint_every_n_epochs', default=None, type=int,
                        help='Save checkpoints every N epochs (in addition to best-metric saves)')

    # Multi-metric checkpoint monitors (JSON config only)
    parser.add_argument('--checkpoint_monitors', default=None, type=str,
                        help='JSON list of {metric, mode} dicts for checkpoint saving (set via config)')

    parser.add_argument('--use_sync_safe_checkpoint', default='False', type=str,
                        help='Use SyncSafeModelCheckpoint to bypass reduce_boolean_decision '
                             '(workaround for XPU/CCL integer all-reduce bug)')

    return parser


def get_model_args(parser):
    """
    Configure argument parser with model-specific parameters.
    
    This function adds arguments to the provided parser for configuring 
    model training settings including image size, class counts, embedding dimensions,
    optimization parameters, and hardware utilization options.
    
    Args:
        parser: ArgumentParser object to which arguments will be added
        
    Returns:
        The parser with added model-specific arguments
    """
    parser.add_argument('--image_size', default=16, type=int,
                        help='size of training images, for rescaling')
    parser.add_argument('--num_classes', default=3, type=int,
                        help='No. of classes for transformer')
    parser.add_argument('--text_emb_dim', default=256, type=int,
                        help='size of the text embedding')
    parser.add_argument('--num_y_class_labels', default=10, type=int,
            help='No. of y class labels for conditioning tranformer')
    parser.add_argument('--choose_optim', default='AdamW', type=str,
            help='Choose optimizer for training')
    parser.add_argument('--acc_grad_batches', default=4, type=int,
            help='Choose how many gradient steps to accumulate')
    parser.add_argument('--gpu_devices', default=1, type=int,
            help='Number of gpus to used for training')
    parser.add_argument('--num_nodes', default=1, type=int,
            help='Number of nodes to used for training')
    def float_or_str(value):
        try:
            return float(value)
        except ValueError:
            return value
    parser.add_argument('--scheduler_gamma', default=None, type=float_or_str,
                        help='Define the learning rate scheduler')
    return parser


def get_path_args(parser):
    """
    Configure argument parser with path-related parameters.
    
    This function adds arguments to the provided parser for specifying 
    various file paths and directories needed for training outputs, 
    checkpoints, logs, and version tracking.
    
    Args:
        parser: ArgumentParser object to which arguments will be added
        
    Returns:
        The parser with added path-specific arguments
    """
    parser.add_argument('--run_id', default=None, type=str,
                        help='unique identifier for this training run')
    parser.add_argument('--runs_folder', default='runs', type=str,
                        help='subdirectory under output_root for per-run logs and artifacts')
    parser.add_argument('--resume_from_checkpoint_state_dict_path', default=None, type=str,
                        help='Path to the deepspeed pytorch ckpt saved as state dict...')
    return parser


def get_wrapper_args(parser):
    """

    """
    parser.add_argument('--hydra', action="store_true", 
                        help='Whether to run with Hydra.')
    parser.add_argument('--wandb', type=str, default="False",
                        help='Flag to use Weights&Biases.')
    parser.add_argument('--wandb_name', type=str, default=None, 
                        help='Weights&Biases run name.')
    parser.add_argument('--wandb_entity', type=str, default=None, 
                        help='Weights&Biases entity.')
    parser.add_argument('--wandb_project', type=str, default=None, 
                        help='Weights&Biases project.')
    parser.add_argument('--wandb_tags', type=str, nargs="*", default=[],
                        help='Weights&Biases tags.')
    

    return parser


def set_seed(seed):
    """
    Set random seeds for reproducibility across different libraries.
    
    This function ensures deterministic behavior by setting identical random
    seeds for PyTorch, NumPy, and Python's built-in random module based on
    the seed value specified in the arguments.
    
    Args:
        seed: Random seed
        
    Returns:
        None
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return


def compile_model(
        args: any,
    ) -> pl.LightningModule:
    """
    Create and compile a PyTorch Lightning model for training.
    
    This function instantiates a base model using the provided arguments
    and wraps it in a PyTorch Lightning module (PL_ProtARDM) to enable
    distributed training, checkpointing, and other Lightning features.
    
    Args:
        args: Configuration object containing model parameters
        data_shape: Tuple specifying input data dimensions (default: (16,16))
        num_classes: Number of output classes (default: 3)
        
    Returns:
        A PyTorch Lightning module ready for training
    """
    model = prepare_model_ProteoScribe(
        config_args=args,
        model_fpath=args.pretrained_weights,
        device=args.device,
        strict=True,
        eval=False,
        attempt_correction=True,
        verbosity=2
    )

    PL_model = PL_mod.PL_ProtARDM(
        args=args,
        model=model,
    )
    return PL_model



def get_model_params(
        model_param_df_path,
        model: nn.Module
        ):
    """
    Calculate and save model parameter statistics to a CSV file.
    
    This function computes the total number of parameters in the provided
    neural network model, displays this count on the console, and saves
    the information to a CSV file at the specified path.
    
    Args:
        model_param_df_path: File path where the parameter statistics CSV will be saved
        model: PyTorch neural network module to analyze
        
    Returns:
        None
    """
    # save csv file for model param description:
    total_params = sum(
            param.numel() for param in model.parameters()
    )
    logger.info('Total number of model parameters: %s', total_params)
    model_param = {}
    model_param['total_params'] = [total_params]
    model_param_df = pd.DataFrame(model_param)
    model_param_df.to_csv(model_param_df_path, index=False)
    return


def get_protein_dataloader(args=any) -> DataLoader:
    """
    Create a DataLoader for protein sequence data.

    This function loads protein data from the specified path, prepares it
    for model training by converting sequences to numerical format and
    extracting text embeddings, then creates and returns a DataLoader
    with the appropriate batch size and shuffle settings.

    Args:
        args: Configuration object containing data_root path and batch_size

    Returns:
        DataLoader object with prepared protein sequence data
    """
    data = torch.load(args.data_root)
    num_seq_list, text_emb = prep.prepare_protein_data(
            args=args,
            data_dict=data
    )
    train_dataset = prep.protein_dataset(
            num_seq_list=num_seq_list,
            text_emb=text_emb
    )
    protein_dataloader = DataLoader(
            #subset_dataset,
            train_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=True
    )
    return protein_dataloader


def get_deepspeed_model(args: any, PL_model) -> pl.LightningModule:
    """
    Load a model from a DeepSpeed checkpoint.

    This function handles the conversion of a DeepSpeed ZeRO checkpoint into a
    standard PyTorch state dictionary format and then loads it into the
    provided PyTorch Lightning model. The conversion process consolidates
    distributed model parameters into a single file.

    Args:
        args: Configuration object containing a 'resume_from_checkpoint' path
        PL_model: PyTorch Lightning model structure to load weights into

    Returns:
        A PyTorch Lightning module with loaded weights from the checkpoint
    """
    PL_model.model = prepare_model_ProteoScribe(
        config_args=args,
        model_fpath=args.resume_from_checkpoint,
        device=args.device,
        strict=True,
        eval=False,
        attempt_correction=True,
    )
    return PL_model


def save_model(
        args: any,
        checkpoint_path: str,
        artifacts_path: str,
        PL_model: pl.LightningModule,
        trainer: Trainer,
        extra_checkpoint_callbacks=None,
    ) -> None:
    """
    Save a PyTorch Lightning model trained with DeepSpeed.

    Converts DeepSpeed ZeRO checkpoints into standard PyTorch state
    dictionaries. All derived weight files are written to checkpoint_path
    (alongside the raw .ckpt dirs). A copy of the best state_dict is
    also placed in artifacts_path for convenient downstream access.

    Only executes on the global_zero process in distributed environments.

    Args:
        args: Configuration object containing model parameters
        checkpoint_path: Directory containing Lightning checkpoints
        artifacts_path: Directory for run artifacts (receives best state_dict copy)
        PL_model: PyTorch Lightning model to be saved
        trainer: Trainer instance with checkpoint callback info
    """
    image_size = args.image_size
    num_classes = args.num_classes

    # once saved via the model checkpoint callback...
    # we have a saved folder containing the deepspeed checkpoint rather than a single file
    last_ckpt_fpath = os.path.join(checkpoint_path, 'last.ckpt')
    best_ckpt_fpath = trainer.checkpoint_callback.best_model_path
    last_single_ckpt_fpath = os.path.join(checkpoint_path, 'single_model.last.pth')
    best_single_ckpt_fpath = os.path.join(checkpoint_path, 'single_model.best.pth')
    last_state_dict_fpath = os.path.join(checkpoint_path, 'state_dict.last.pth')
    best_state_dict_fpath = os.path.join(checkpoint_path, 'state_dict.best.pth')
    last_state_dict_ema_fpath = os.path.join(checkpoint_path, 'state_dict_ema.last.pth')
    best_state_dict_ema_fpath = os.path.join(checkpoint_path, 'state_dict_ema.best.pth')

    symlink_to = "best"  # symlink from state_dict.pth to either best or last
    
    if trainer.is_global_zero:
        # Back up any derived files from a previous run in this directory
        for fpath in (
            best_single_ckpt_fpath, best_state_dict_fpath, best_state_dict_ema_fpath,
            last_single_ckpt_fpath, last_state_dict_fpath, last_state_dict_ema_fpath,
            os.path.join(checkpoint_path, 'state_dict.pth'),
            os.path.join(checkpoint_path, 'state_dict_ema.pth'),
            os.path.join(checkpoint_path, 'params.csv'),
        ):
            backup_if_exists(fpath)

        # Check whether last and best point to the same checkpoint
        # (last.ckpt is a symlink when save_last="link")
        last_real = os.path.realpath(last_ckpt_fpath)
        best_real = os.path.realpath(best_ckpt_fpath)
        same_checkpoint = (last_real == best_real)

        # ---------------- BEST MODEL ----------------
        logger.info('Converting DeepSpeed checkpoint to fp32 (best)...')
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            convert_zero_checkpoint_to_fp32_state_dict(
                best_ckpt_fpath, best_single_ckpt_fpath
            )
        logger.info('Save model (best)')
        new_temp_model = mod.get_model(
            args=args,
            data_shape=(image_size, image_size),
            num_classes=num_classes
        ).cpu()
        loaded_model = PL_mod.PL_ProtARDM.load_from_checkpoint(
            best_single_ckpt_fpath,
            args=args, model=new_temp_model
        )
        # make sure that datatypes are the same ...
        if PL_model.dtype != loaded_model.dtype:
            msg = "Data types are not matching."
            msg += f" Expected {PL_model.dtype}. Loaded: {loaded_model.dtype}"
            assert False, msg
        else:
            # save model state dict without pytorch lightning wrapper
            torch.save(loaded_model.model.state_dict(), best_state_dict_fpath)
            if hasattr(loaded_model, 'ema_model'):
                logger.info('Also saving EMA model...')
                torch.save(loaded_model.ema_model.state_dict(), best_state_dict_ema_fpath)
            # check model params
            get_model_params(
                    os.path.join(checkpoint_path, 'params.csv'),
                    model=loaded_model.model
            )

        # ---------------- LAST MODEL ----------------
        if same_checkpoint:
            logger.info('Last checkpoint is same as best — creating symlinks')
            os.symlink(best_single_ckpt_fpath, last_single_ckpt_fpath)
            os.symlink(best_state_dict_fpath, last_state_dict_fpath)
            if os.path.exists(best_state_dict_ema_fpath):
                os.symlink(best_state_dict_ema_fpath, last_state_dict_ema_fpath)
        else:
            logger.info('Converting DeepSpeed checkpoint to fp32 (last)...')
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                convert_zero_checkpoint_to_fp32_state_dict(
                    last_ckpt_fpath, last_single_ckpt_fpath
                )
            logger.info('Save model (last)')
            new_temp_model = mod.get_model(
                args=args,
                data_shape=(image_size, image_size),
                num_classes=num_classes
            ).cpu()
            loaded_model = PL_mod.PL_ProtARDM.load_from_checkpoint(
                last_single_ckpt_fpath,
                args=args, model=new_temp_model
            )
            if PL_model.dtype != loaded_model.dtype:
                msg = "Data types are not matching."
                msg += f" Expected {PL_model.dtype}. Loaded: {loaded_model.dtype}"
                assert False, msg
            else:
                torch.save(loaded_model.model.state_dict(), last_state_dict_fpath)
                if hasattr(loaded_model, 'ema_model'):
                    logger.info('Also saving EMA model...')
                    torch.save(loaded_model.ema_model.state_dict(), last_state_dict_ema_fpath)
                get_model_params(
                        os.path.join(checkpoint_path, 'params.csv'),
                        model=loaded_model.model
                )
        
        symlink_path = os.path.join(checkpoint_path, 'state_dict.pth')
        symlink_ema_path = os.path.join(checkpoint_path, 'state_dict_ema.pth')
        if symlink_to == "best":
            os.symlink(best_state_dict_fpath, symlink_path)
            if os.path.exists(best_state_dict_ema_fpath):
                os.symlink(best_state_dict_ema_fpath, symlink_ema_path)
        elif symlink_to == "last":
            os.symlink(last_state_dict_fpath, symlink_path)
            if os.path.exists(last_state_dict_ema_fpath):
                os.symlink(last_state_dict_ema_fpath, symlink_ema_path)

        # Copy best state_dict to artifacts directory
        os.makedirs(artifacts_path, exist_ok=True)
        artifact_best = os.path.join(artifacts_path, 'state_dict.best.pth')
        backup_if_exists(artifact_best)
        shutil.copy2(best_state_dict_fpath, artifact_best)
        logger.info("Copied best state_dict to %s", artifact_best)

        # ---- Process additional checkpoint monitors ----
        checkpoint_summary = {
            "primary": {
                "metric": "val_loss",
                "best_path": artifact_best,
                "best_score": float(trainer.checkpoint_callback.best_model_score)
                    if trainer.checkpoint_callback.best_model_score is not None else None,
            },
            "additional": [],
        }
        if extra_checkpoint_callbacks:
            for cb in extra_checkpoint_callbacks:
                if not cb.best_model_path:
                    logger.warning("No best checkpoint found for monitor %s", cb.monitor)
                    continue
                metric_slug = cb.monitor.replace("/", "_")
                extra_sd_fname = f"state_dict.best_{metric_slug}.pth"
                extra_sd_fpath = os.path.join(checkpoint_path, extra_sd_fname)
                backup_if_exists(extra_sd_fpath)
                # Convert DeepSpeed checkpoint to state_dict
                extra_single_fpath = os.path.join(checkpoint_path, f"single_model.best_{metric_slug}.pth")
                backup_if_exists(extra_single_fpath)
                logger.info("Converting DeepSpeed checkpoint for %s...", cb.monitor)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    convert_zero_checkpoint_to_fp32_state_dict(
                        cb.best_model_path, extra_single_fpath
                    )
                extra_temp_model = mod.get_model(
                    args=args,
                    data_shape=(image_size, image_size),
                    num_classes=num_classes
                ).cpu()
                extra_loaded = PL_mod.PL_ProtARDM.load_from_checkpoint(
                    extra_single_fpath,
                    args=args, model=extra_temp_model
                )
                torch.save(extra_loaded.model.state_dict(), extra_sd_fpath)
                # Copy to artifacts
                artifact_extra = os.path.join(artifacts_path, extra_sd_fname)
                backup_if_exists(artifact_extra)
                shutil.copy2(extra_sd_fpath, artifact_extra)
                logger.info("Saved best state_dict for %s to %s", cb.monitor, artifact_extra)
                checkpoint_summary["additional"].append({
                    "metric": cb.monitor,
                    "mode": cb.mode,
                    "best_score": float(cb.best_model_score)
                        if cb.best_model_score is not None else None,
                    "artifact": extra_sd_fname,
                })

        summary_path = os.path.join(artifacts_path, 'checkpoint_summary.json')
        backup_if_exists(summary_path)
        with open(summary_path, 'w') as f:
            json.dump(checkpoint_summary, f, indent=2)
        logger.info("Checkpoint summary written to %s", summary_path)

    return


def str_to_bool(s):
    """
    Convert string representation of boolean values to Python bool type.
    
    This utility function handles conversion of case-insensitive string 
    values 'true' and 'false' to their corresponding Python boolean values. 
    It raises an error for any other input string.
    
    Args:
        s: String to convert, expected to be 'true' or 'false' (case-insensitive)
        
    Returns:
        bool: True if input is 'true' (case-insensitive), False if 'false'
        
    Raises:
        ValueError: If the input is anything other than 'true' or 'false'
    """
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
        else:
            return s
    elif s is None:
        return None
    else:
        raise ValueError("Input must be string or 'None'")


# === main functions ===

def clear_gpu_cache():
    """
    Free up GPU memory by clearing caches and running garbage collection.
    
    This utility function performs three operations to optimize GPU memory usage:
    1. Sets PyTorch's float32 matrix multiplication precision to 'medium' 
       (balances performance and accuracy)
    2. Empties the CUDA memory cache to release unused memory
    3. Runs Python's garbage collector to remove unreferenced objects
    
    This function is useful to call between training runs or when switching between
    memory-intensive operations to prevent out-of-memory errors.
    
    Args:
        None
        
    Returns:
        None
    """
    torch.set_float32_matmul_precision('medium')
    # torch.cuda.empty_cache()
    # torch.xpu.empty_cache()
    gc.collect()
    return


def retrieve_all_args(args):
    """
    Collect and consolidate all command line arguments from various components.

    Supports an optional ``--config_path`` argument pointing to a JSON file.
    When provided, the JSON values override argparse defaults but are themselves
    overridden by any explicitly passed CLI arguments (i.e. CLI > JSON > defaults).

    Args:
        args: list of CLI argument strings (typically ``sys.argv[1:]``)

    Returns:
        argparse.Namespace: A namespace object containing all parsed arguments
    """
    # Pre-parse to extract --config_path before building the full parser,
    # so we can set JSON values as defaults before the real parse.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config_path', '-c', type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(args)

    parser = argparse.ArgumentParser(description='Stage 3: ProteoScribe')
    parser.add_argument('--config_path', '-c', type=str, default=None,
                        help='Path to JSON config file. Values are overridden by CLI args.')
    get_args(parser=parser)
    get_model_args(parser=parser)
    mod.add_model_args(parser=parser)
    get_path_args(parser=parser)
    get_wrapper_args(parser=parser)

    if pre_args.config_path is not None:
        json_config = load_json_config(pre_args.config_path)
        parser.set_defaults(**json_config)

    args = parser.parse_args(args)

    # Type conversions (idempotent — pass through values already of the target type)
    args.resume_from_checkpoint = nonestr_to_none(args.resume_from_checkpoint)
    args.download = str_to_bool(args.download)

    # New generalized dataset args
    args.primary_data_path = nonestr_to_none(args.primary_data_path)
    args.start_secondary = str_to_bool(args.start_secondary)

    # Map deprecated aliases to new names
    args.swissprot_data_root = nonestr_to_none(args.swissprot_data_root)
    args.pfam_data_root = nonestr_to_none(args.pfam_data_root)
    args.start_pfam_trainer = str_to_bool(args.start_pfam_trainer)
    if args.primary_data_path is None and args.swissprot_data_root is not None:
        logger.warning("--swissprot_data_root is deprecated; use --primary_data_path")
        args.primary_data_path = args.swissprot_data_root
    if args.secondary_data_paths is None and args.pfam_data_root is not None:
        logger.warning("--pfam_data_root is deprecated; use --secondary_data_paths")
        args.secondary_data_paths = [args.pfam_data_root]
    if not args.start_secondary and args.start_pfam_trainer:
        logger.warning("--start_pfam_trainer is deprecated; use --start_secondary")
        args.start_secondary = args.start_pfam_trainer

    # Resolve training_strategy
    if args.training_strategy == 'auto':
        args.training_strategy = 'combine' if args.secondary_data_paths else 'primary_only'

    args.finetune = str_to_bool(args.finetune)
    args.finetune_output_layers = str_to_bool(args.finetune_output_layers)
    args.pretrained_weights = nonestr_to_none(args.pretrained_weights)
    args.wandb = str_to_bool(args.wandb)
    args.scale_learning_rate = str_to_bool(args.scale_learning_rate)
    args.save_metrics_history = str_to_bool(args.save_metrics_history)
    args.metrics_history_all_ranks_val_loss = str_to_bool(
        args.metrics_history_all_ranks_val_loss
    )
    args.use_sync_safe_checkpoint = str_to_bool(args.use_sync_safe_checkpoint)
    args.early_stopping_metric = nonestr_to_none(args.early_stopping_metric)
    args.checkpoint_every_n_steps = nonestr_to_none(args.checkpoint_every_n_steps)
    args.checkpoint_every_n_epochs = nonestr_to_none(args.checkpoint_every_n_epochs)

    return args


def load_data(
        args, *,
        primary_data_path,
        secondary_data_paths,
        facilitator,
    ):
    """
    Initialize and prepare a data module for protein sequence datasets.

    Creates an :class:`HDF5DataModule` from one primary HDF5 file and zero
    or more secondary HDF5 files.  The *facilitator* name determines the
    HDF5 group name (e.g. ``MMD_data``).

    Args:
        args: Configuration namespace (batch_size, num_workers, etc.)
        primary_data_path: Path to the primary HDF5 training dataset.
        secondary_data_paths: List of paths to secondary HDF5 datasets, or None.
        facilitator: Facilitator name used as HDF5 group prefix.

    Returns:
        A configured PyTorch Lightning data module ready for training.
    """
    data_module = PL_mod.HDF5DataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        valid_size=args.valid_size,
        seed=args.seed,
        diffusion_steps=args.diffusion_steps,
        image_size=args.image_size,
        primary_path=primary_data_path,
        secondary_paths=secondary_data_paths,
        group_name=facilitator + '_data',
    )
    data_module.setup()
    return data_module


def load_model(
    args, *,
    data_module
    ):
    """
    Initialize and configure a model based on dataset characteristics.
    
    This function performs several important setup steps:
    1. Calculates training epoch length adjusted for distributed training
    2. Validates diffusion step count against data dimensionality
    3. Logs GPU information
    4. Instantiates the model with appropriate parameters
    5. Reports model size
    
    The function ensures all requirements for training are met before
    returning the configured PyTorch Lightning model.
    
    Args:
        args: Configuration object containing model parameters and training settings
        data_module: PyTorch Lightning data module with dataset information
        
    Returns:
        A configured PyTorch Lightning model ready for training
    """
    gpu_devices = args.gpu_devices
    acc_grad_batches = args.acc_grad_batches
    diffusion_steps = args.diffusion_steps
    image_size = args.image_size
    num_nodes = args.num_nodes
    batch_size = args.batch_size

    args.traindata_len = len(data_module.train_dataloader()) // gpu_devices // acc_grad_batches
    logger.info('Length of dataloader: %s', len(data_module.train_dataloader()))
    logger.info('Numer of devices: %s', gpu_devices)
    logger.info('Number of nodes: %s', num_nodes)
    logger.info('Batch size: %s', batch_size)
    logger.info('Length of dataloader per device: %s', len(data_module.train_dataloader()) // gpu_devices)
    logger.info('Length of a training epoch in batch gradient updates: %s', args.traindata_len)
    w, h = image_size, image_size
    # Ensure diffusion steps are sufficient for data dimensions
    if diffusion_steps < int(w*h):
        logger.warning('Make sure that the number of diffusion steps is equal to or greather than the data cardinality')
    if get_rank() == 0:
        print_gpu_initialization()
    # Compile model architecture
    PL_model = compile_model(
        args=args,
    )
    logger.info('Model size: %s', sum(p.numel() for p in PL_model.model.parameters()))
    return PL_model


def load_pretrained_weights(
        PL_model,
        checkpoint_path: str,
        args=None,
    ):
    """
    Load pretrained model weights into a PyTorch Lightning model.

    Supports raw state dicts (.bin, .pth, .pt), Lightning checkpoints (.ckpt),
    and sharded DeepSpeed checkpoint directories. Format is auto-detected.
    Parameter renaming/correction is applied automatically.

    Args:
        PL_model: PyTorch Lightning model to load weights into
        checkpoint_path: Path to model weights (file or directory)
        args: Configuration namespace (used to rebuild model graph if needed).
              If None, uses PL_model.script_args.

    Returns:
        The model with loaded pretrained weights
    """
    if args is None:
        args = PL_model.script_args

    logger.info("Loading pretrained weights from: %s", checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    PL_model.model = prepare_model_ProteoScribe(
        config_args=args,
        model_fpath=checkpoint_path,
        device='cpu',
        strict=True,
        eval=False,
        attempt_correction=True,
    )

    logger.info("Pretrained weights loaded successfully")
    return PL_model


def freeze_except_last_n_blocks_and_layers(
        PL_model,
        n_blocks,
        n_layers,
        finetune_output_layers=True
    ):
    """
    Freeze all model parameters except those in the last n transformer blocks and
    layers.
    
    This enables efficient finetuning by only updating parameters in the last
    n transformer blocks while keeping all other layers frozen. This approach
    reduces computational cost and can help prevent catastrophic forgetting.
    
    Args:
        PL_model: PyTorch Lightning model with loaded pretrained weights
        n_blocks: Number of last transformer blocks to keep trainable (default: 1)
                  If n_blocks=0, no blocks will be trainable (complete freezing)
                  If n_blocks=-1, all blocks will be trainable (no freezing)
                  If n_blocks >= total blocks, all blocks will be trainable
        n_layers: Number of last transformer layers to keep trainable (default: 1)
                  If n_layers=0, no layers will be trainable (complete freezing)
                  If n_layers=-1, all layers will be trainable (no freezing)
                  If n_layers >= total layers, all layers will be trainable
        
    Returns:
        The model with frozen parameters (except last n blocks)
    """
    if n_blocks == -1 and n_layers == -1:
        logger.info("n_blocks=n_layers=-1: All parameters will remain trainable (no freezing)")
        # return PL_model

    if n_blocks == 0 and n_layers == 0:
        logger.info("n_blocks=n_layers=0: All parameters will be frozen (no training)")
    else:
        msg = f"Freezing all parameters except the last {n_layers} layer(s) "
        msg += f"of each of the last {n_blocks} transformer block(s)..."
        logger.info(msg)
    
    # First, freeze all parameters
    for param in PL_model.model.parameters():
        param.requires_grad = False
    
    # Then unfreeze only the last k layers of the last n transformer blocks
    # The model structure is: PL_model.model.transformer.transformer_blocks[bidx][depth]
    if hasattr(PL_model.model, 'transformer'):
        transformer = PL_model.model.transformer
        if hasattr(transformer, 'transformer_blocks') and len(transformer.transformer_blocks) > 0:
            total_blocks = len(transformer.transformer_blocks)
            if n_blocks == -1:
                n_blocks = total_blocks
            blocks_to_unfreeze = min(n_blocks, total_blocks)
            # Unfreeze the last k layers of the last n blocks
            logger.info("Attemtping to unfreeze last %s blocks...", blocks_to_unfreeze)
            for i in range(total_blocks - blocks_to_unfreeze, total_blocks):
                logger.debug("***** Freezing block %s...", i)
                block = transformer.transformer_blocks[i]
                if len(block) > 0:
                    total_layers = len(block)
                    if n_layers == -1:
                        n_layers = total_layers
                    layers_to_unfreeze = min(n_layers, total_layers)
                    for j in range(total_layers - layers_to_unfreeze, total_layers):
                        layer = block[j]
                        logger.debug("*** Freezing block %s layer %s", i, j)
                        for param in layer.parameters():
                            param.requires_grad = True
                else:
                    for param in block.parameters():
                        param.requires_grad = True
            
            logger.info("Unfroze last %s transformer block(s) out of %s total blocks", blocks_to_unfreeze, total_blocks)
            if blocks_to_unfreeze < n_blocks:
                logger.info("Note: Requested %s blocks but only %s exist in model", n_blocks, total_blocks)
        else:
            logger.warning("Could not find transformer_blocks in model")
    else:
        logger.warning("Could not find transformer attribute in model")
    
    # Also unfreeze the final output layers (norm and out) for better finetuning
    if finetune_output_layers:
        if hasattr(PL_model.model, 'transformer'):
            transformer = PL_model.model.transformer
            if hasattr(transformer, 'norm'):
                for param in transformer.norm.parameters():
                    param.requires_grad = True
                logger.info("Unfroze final LayerNorm")
            if hasattr(transformer, 'out'):
                for param in transformer.out.parameters():
                    param.requires_grad = True
                logger.info("Unfroze final output layer")
    
    # Count trainable vs frozen parameters
    trainable_params = sum(p.numel() for p in PL_model.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in PL_model.model.parameters())
    frozen_params = total_params - trainable_params
    
    logger.info("Trainable parameters: %s (%.2f%%)", f"{trainable_params:,}", 100 * trainable_params / total_params)
    logger.info("Frozen parameters: %s (%.2f%%)", f"{frozen_params:,}", 100 * frozen_params / total_params)
    
    return PL_model


_TRAINING_ENV_PREFIXES = (
    "CUDA_", "NCCL_", "TORCH_", "DEEPSPEED_", "WANDB_",
    "MASTER_", "WORLD_SIZE", "RANK", "LOCAL_RANK",
    "SLURM_", "PBS_", "COBALT_", "PALS_", "PMI_", "OMPI_",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS",
)

_SENSITIVE_ENV_SUBSTRINGS = (
    "KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH",
)


def _collect_training_env():
    """Collect environment variables relevant to distributed training and HPC.

    Variables whose names contain sensitive substrings (API keys, tokens, etc.)
    are excluded.
    """
    env = {"hostname": socket.gethostname()}
    for key, val in sorted(os.environ.items()):
        if key.startswith(_TRAINING_ENV_PREFIXES):
            upper = key.upper()
            if any(s in upper for s in _SENSITIVE_ENV_SUBSTRINGS):
                continue
            env[key] = val
    return env


###########################
##  Training Entrypoint  ##
###########################

def train_model(
        args,
        PL_model,
        data_module,
        ds_config=None,
        verbosity=1,
    ):
    """
    Train a PyTorch Lightning model with DeepSpeed distributed optimization.
    
    This function sets up the full training infrastructure including DeepSpeed 
    configuration, logging, checkpointing, and handles different training scenarios 
    (new training, resumed training, or phase-transfer training). Training parameters 
    adapt based on the dataset type, with special handling for Pfam datasets.
    
    The function supports two training modes:
    1. Epoch-based training (standard mode when not using Pfam)
    2. Step-based training with validation interval monitoring (for Pfam)
    
    After training completes, model weights are automatically saved in various formats.
    
    Args:
        args: Configuration object containing training parameters and paths
        PL_model: PyTorch Lightning model to be trained
        data_module: Data module providing training and validation data
        ds_config: DeepSpeed configurations. If None, uses defaults.
        
    Returns:
        None - results are saved to the paths specified in args
    """
    logger.info('Beginning Training...')
    # Note: Nested args must be accessed via dictionaries, not attributes.
    output_root = args.output_root
    checkpoints_folder = args.checkpoints_folder
    runs_folder = args.runs_folder
    run_id = args.run_id
    log_every_n_steps = args.log_every_n_steps
    num_training_batches = getattr(args, 'traindata_len', None)
    if num_training_batches and log_every_n_steps > num_training_batches:
        log_every_n_steps = max(1, int(num_training_batches))
        logger.info("Clamped log_every_n_steps to %d (number of training batches)",
                     log_every_n_steps)
    training_strategy = args.training_strategy
    gpu_devices = args.gpu_devices
    num_nodes = args.num_nodes
    acc_grad_batches = args.acc_grad_batches
    epochs = args.epochs
    start_secondary = args.start_secondary  # Expect bool
    assert isinstance(start_secondary, bool), "start_secondary not bool"
    max_steps = args.max_steps
    val_check_interval = args.val_check_interval
    limit_val_batches = args.limit_val_batches
    resume_from_checkpoint = args.resume_from_checkpoint  # Expect str or None
    assert resume_from_checkpoint is None or (
            isinstance(resume_from_checkpoint, str) and 
            resume_from_checkpoint != "None"
        ), f"resume_from_checkpoint should be str or None (not str `None`)." + \
           f" Got {resume_from_checkpoint} (type {type(resume_from_checkpoint)})"
    precision = args.precision
    use_wandb = args.wandb

    # Scale the learning rate with number of total devices
    if args.scale_learning_rate:
        n = num_nodes * gpu_devices
        logger.info("Scaling learning rate with effective batch size: "
                     "num_nodes x gpu_devices = %s x %s = %s", num_nodes, gpu_devices, n)
        args.lr = args.lr * n
    logger.info("Effective learning rate: %s", args.lr)
    
    # Configure DeepSpeed optimization settings
    if ds_config is None:
        ds_config = {
            "zero_optimization": {
                "stage": 1,
                "allgather_bucket_size": 5e8,
                "reduce_bucket_size": 5e8,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": False
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": False
                },
                "overlap_comm": True,
                "contiguous_gradients": True
            },
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
        }
    
    # strategy = DeepSpeedStrategy(
    #     config=ds_config
    # )

    # Derived output paths
    checkpoint_dir = os.path.join(output_root, checkpoints_folder, run_id)
    run_dir = os.path.join(output_root, runs_folder, run_id)
    logs_dir = os.path.join(run_dir, _LOGS_SUBDIR)
    artifacts_dir = os.path.join(run_dir, _ARTIFACTS_SUBDIR)

    loggers = []

    # Set up TensorBoard logging
    logger.info("Setting up TensorBoard logging...")
    tb_logger = TensorBoardLogger(
        save_dir=logs_dir,
        version="",
    )
    loggers.append(tb_logger)

    # Set up Weights&Biases logging
    logger.info("Setting up Weights&Biases logging...")
    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(
            name=args.wandb_name if args.wandb_name else run_id,
            save_dir=logs_dir,
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            tags=args.wandb_tags,
            group=run_id,
            log_model="all",
        )  # TODO: investigate "all"
        loggers.append(wandb_logger)

    # Monitor learning rate changes
    logger.info("Setting up LearningRateMonitor...")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Monitor GPU usage
    logger.info("Setting up DeviceStatesMonitor...")
    gpu_logger = DeviceStatsMonitor()

    # ---- Checkpoint callbacks ----
    logger.info("Configuring ModelCheckpoint...")

    # Parse checkpoint_monitors from config (list of {metric, mode} dicts)
    checkpoint_monitors = getattr(args, 'checkpoint_monitors', None)
    if checkpoint_monitors is None:
        checkpoint_monitors = [{"metric": "val_loss", "mode": "min"}]
    elif isinstance(checkpoint_monitors, str):
        checkpoint_monitors = json.loads(checkpoint_monitors)

    # Optional periodic saving
    every_n_steps = getattr(args, 'checkpoint_every_n_steps', None)
    every_n_epochs = getattr(args, 'checkpoint_every_n_epochs', None)
    # For step-based training (combine strategy), default periodic saving to log_every_n_steps
    if training_strategy == 'combine' and every_n_steps is None:
        every_n_steps = int(log_every_n_steps)

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
            # Primary monitor: keep top-2, save last symlink
            ckpt_kwargs["save_top_k"] = 2
            ckpt_kwargs["save_last"] = "link"
            if every_n_steps is not None:
                ckpt_kwargs["every_n_train_steps"] = int(every_n_steps)
        else:
            # Additional monitors: keep best-1, unique filename
            metric_slug = mon["metric"].replace("/", "_")
            ckpt_kwargs["save_top_k"] = 1
            ckpt_kwargs["save_last"] = False
            ckpt_kwargs["filename"] = f"best-{metric_slug}-{{epoch}}"
        if getattr(args, 'use_sync_safe_checkpoint', False):
            from biom3.Stage3.callbacks import SyncSafeModelCheckpoint
            checkpoint_callbacks.append(SyncSafeModelCheckpoint(**ckpt_kwargs))
        else:
            from biom3.Stage3.callbacks import LoggingModelCheckpoint
            checkpoint_callbacks.append(LoggingModelCheckpoint(**ckpt_kwargs))

    # ---- Metrics history ----
    if getattr(args, 'save_metrics_history', True):
        from biom3.Stage3.callbacks import MetricsHistoryCallback
        metrics_cb = MetricsHistoryCallback(
            output_dir=artifacts_dir,
            save_ranks=getattr(args, 'metrics_history_ranks', [0]),
            every_n_steps=getattr(args, 'metrics_history_every_n_steps', 1),
            all_ranks_val_loss=getattr(
                args, 'metrics_history_all_ranks_val_loss', False
            ),
        )
    else:
        metrics_cb = None

    # ---- Early stopping ----
    early_stopping_metric = getattr(args, 'early_stopping_metric', None)
    if isinstance(early_stopping_metric, str) and early_stopping_metric.lower() == 'none':
        early_stopping_metric = None

    callbacks = checkpoint_callbacks + [lr_monitor, gpu_logger]
    if metrics_cb is not None:
        callbacks.append(metrics_cb)
    if early_stopping_metric is not None:
        logger.info("Enabling early stopping on %s (patience=%d)",
                     early_stopping_metric, args.early_stopping_patience)
        callbacks.append(EarlyStopping(
            monitor=early_stopping_metric,
            patience=args.early_stopping_patience,
            min_delta=getattr(args, 'early_stopping_min_delta', 0.0),
            mode=getattr(args, 'early_stopping_mode', 'min'),
            verbose=True,
        ))

    # Define common trainer parameters
    trainer_params = {
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'enable_checkpointing': True,
        'devices': gpu_devices,
        'num_nodes': num_nodes,
        'accelerator': args.device,
        'strategy': 'deepspeed_stage_2',  # TODO: use strategy defined above?
        'accumulate_grad_batches': acc_grad_batches,
        'logger': loggers,
        'log_every_n_steps': log_every_n_steps,
        'callbacks': callbacks,
    }

    # Configure training mode: epoch-based (primary_only) or step-based (combine)
    if training_strategy == 'primary_only':
        trainer_params['max_epochs'] = epochs
    else:
        trainer_params['max_steps'] = max_steps
        trainer_params['val_check_interval'] = val_check_interval
        trainer_params['limit_val_batches'] = limit_val_batches

    # Initialize trainer with configured parameters
    logger.info('Initializing Trainer...')
    trainer = Trainer(**trainer_params)

    # wrap optimizer and model with intel extension for pytorch 
    # optimizer = torch.optim.AdamW(PL_model.parameters(), lr=lr)
    # PL_model, optimizer = ipex.optimize(PL_model, optimizer=optimizer, dtype=torch.float32)

    # Handle different training scenarios
    if args.finetune:
        # Finetuning
        logger.info("Start finetuning")
        trainer.fit(PL_model, data_module)
    else:
        # Pretraining
        if resume_from_checkpoint is None:
            logger.info("Train from scratch")
            trainer.fit(PL_model, data_module)
        elif start_secondary:
            logger.info('Start training ProteoScribe with secondary data ...')
            trainer.fit(PL_model, data_module)
        else:
            logger.info('Continue training ProteoScribe from checkpoint ...')
            trainer.fit(PL_model, data_module, ckpt_path=resume_from_checkpoint)

    if get_rank() == 0:
        print_gpu_initialization()

    # Save dataset split indices
    if get_rank() == 0 and hasattr(data_module, 'split_info'):
        splits_path = os.path.join(artifacts_dir, "dataset_splits.pt")
        torch.save(data_module.split_info, splits_path)
        logger.info("Saved dataset splits to %s", splits_path)

    # Save trained model in multiple formats
    save_model(
            args=args,
            checkpoint_path=checkpoint_callbacks[0].dirpath,
            artifacts_path=artifacts_dir,
            PL_model=PL_model,
            trainer=trainer,
            extra_checkpoint_callbacks=checkpoint_callbacks[1:],
    )


def main(args, use_hydra=False, ds_config=None,):

    start_time = datetime.now()

    # ----- Suppress noisy library warnings -----
    warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")
    warnings.filterwarnings("ignore", message=".*isinstance.*treespec.*")
    warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
    logging.getLogger("tensorboardX.x2num").setLevel(logging.ERROR)

    # ----- Set up output directories and file logging -----
    run_dir = os.path.join(args.output_root, args.runs_folder, args.run_id)
    logs_dir = os.path.join(run_dir, _LOGS_SUBDIR)
    artifacts_dir = os.path.join(run_dir, _ARTIFACTS_SUBDIR)
    checkpoint_dir = os.path.join(
        args.output_root, args.checkpoints_folder, args.run_id,
    )
    if get_rank() == 0:
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)
    log_path, file_handler = setup_file_logging(artifacts_dir)

    # ----- Process passed parameters -----
    seed = args.seed
    primary_data_path = args.primary_data_path
    secondary_data_paths = args.secondary_data_paths
    facilitator = args.facilitator

    # ----- Clear the GPU cache -----
    clear_gpu_cache()

    # ----- For reproducibility -----
    if seed <= 0:
        seed = np.random.randint(2**32)
        args.seed = seed
    set_seed(seed)
    logger.info("Using seed: %s", seed)

    # ----- Load Data -----
    data_module = load_data(
        args=args,
        primary_data_path=primary_data_path,
        secondary_data_paths=secondary_data_paths,
        facilitator=facilitator,
    )

    # ----- Load Model -----
    PL_model = load_model(
        args=args,
        data_module=data_module
    )

    # ----- Load pretrained weights and freeze if finetuning -----
    # TODO: handle case where resume_from_checkpoint is specified
    finetuning = args.finetune
    if finetuning:
        finetune_last_n_blocks = args.finetune_last_n_blocks
        finetune_last_n_layers = args.finetune_last_n_layers
        finetune_output_layers = args.finetune_output_layers
        pretrained_weights = args.pretrained_weights
        if finetune_last_n_layers == -2:
            # If flag is set to finetune and layers not specified (default -2)
            # set to default actionable value 1
            finetune_last_n_layers = 1
        if finetune_last_n_blocks == -2:
            # If flag is set to finetune and blocks not specified (default -2)
            # set to default actionable value 1
            finetune_last_n_blocks = 1
        if pretrained_weights is None:
            logger.warning("Finetuning flag --finetune set to True but "
                           "pretrained_weights path not specified.")
            logger.warning("Proceeding with loaded weights")
        elif os.path.exists(pretrained_weights):
            PL_model = load_pretrained_weights(
                PL_model=PL_model,
                checkpoint_path=pretrained_weights
            )
            # Freeze parameters based on user configuration
            PL_model = freeze_except_last_n_blocks_and_layers(
                PL_model=PL_model,
                n_blocks=finetune_last_n_blocks,
                n_layers=finetune_last_n_layers,
                finetune_output_layers=finetune_output_layers
            )
        else:
            logger.warning("Pretrained checkpoint not found at %s", pretrained_weights)
            logger.warning("Proceeding with randomly initialized weights")
    else:
        pass

    # ----- Train Model -----
    train_model(
        args=args,
        PL_model=PL_model,
        data_module=data_module,
        ds_config=ds_config,
    )

    # ----- Write args.json and build manifest (rank 0 only) -----
    if get_rank() == 0:
        elapsed = datetime.now() - start_time

        # Save args.json
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
            "effective_lr": args.lr,
            "precision": args.precision,
            "gpu_devices": args.gpu_devices,
            "num_nodes": args.num_nodes,
            "acc_grad_batches": args.acc_grad_batches,
            "deepspeed_stage": "2",
        }
        outputs["training_strategy"] = args.training_strategy
        if args.training_strategy == 'combine':
            outputs["max_steps"] = args.max_steps
            outputs["val_check_interval"] = args.val_check_interval
        else:
            outputs["epochs"] = args.epochs

        if args.finetune:
            outputs["finetune"] = True
            outputs["finetune_last_n_blocks"] = args.finetune_last_n_blocks
            outputs["finetune_last_n_layers"] = args.finetune_last_n_layers

        resolved_paths = {
            "checkpoint_dir": os.path.abspath(checkpoint_dir),
            "artifacts_dir": os.path.abspath(artifacts_dir),
        }
        if args.primary_data_path is not None:
            resolved_paths["primary_data_path"] = os.path.abspath(
                args.primary_data_path
            )
        if args.secondary_data_paths is not None:
            resolved_paths["secondary_data_paths"] = [
                os.path.abspath(p) for p in args.secondary_data_paths
            ]
        if args.pretrained_weights is not None:
            resolved_paths["pretrained_weights"] = os.path.abspath(
                args.pretrained_weights
            )
        if args.resume_from_checkpoint is not None:
            resolved_paths["resume_from_checkpoint"] = os.path.abspath(
                args.resume_from_checkpoint
            )

        manifest_path = write_manifest(
            args, artifacts_dir, start_time, elapsed,
            outputs=outputs,
            resolved_paths=resolved_paths,
            environment=_collect_training_env(),
        )
        logger.info("Build manifest written to %s", manifest_path)

    # ----- Clean up -----
    teardown_file_logging("biom3", file_handler)


def parse_arguments(args):
    return retrieve_all_args(args)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)    
