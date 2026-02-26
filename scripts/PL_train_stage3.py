#!/usr/bin/env python3

"""Training script for BioM3 Stage 3

Support for PyTorch Lightning and Weights&Biases

"""

import os
import numpy as np
import random
import gc
import argparse
from pathlib import Path
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# PyTorch Lightning imports
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.environments import ClusterEnvironment

# # lightning imports (from local installation)
# import lightning as pl
# from lightning import Trainer
# from lightning.fabric.strategies import DeepSpeedStrategy
# from lightning.fabric.loggers import TensorBoardLogger
# from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
# from lightning.pytorch.callbacks import ModelCheckpoint

# to make the model and pytorch lightning compatible with Intel hardware 
#   i) import the ipex module and 
#   ii) set the device to 'xpu' for the model
# import intel_extension_for_pytorch as ipex

# DeepSpeed is needed for the DeepSpeedStrategy
import deepspeed

# WandB
import wandb

# Custom modules
import biom3.Stage3.preprocess as prep
import biom3.Stage3.cond_diff_transformer_layer as mod
import biom3.Stage3.helper_funcs as help_tools
import biom3.Stage3.PL_wrapper as PL_mod

from mpi4py import MPI


# class MyClusterEnvironment(ClusterEnvironment):
#     @property
#     def creates_processes_externally(self) -> bool:
#         """Return True if the cluster is managed (you don't launch processes yourself)"""
#         return True

#     def world_size(self) -> int:
#         return int(os.environ["WORLD_SIZE"])

#     def global_rank(self) -> int:
#         return int(os.environ["RANK"])

#     def local_rank(self) -> int:
#         return int(os.environ["LOCAL_RANK"])

#     def node_rank(self) -> int:
#         return int(os.environ["NODE_RANK"])

#     @property
#     def main_address(self) -> str:
#         return os.environ["MASTER_ADDR"]

#     @property
#     def main_port(self) -> int:
#         return int(os.environ["MASTER_PORT"])

#     def set_world_size(self, size:int) -> None:
#         pass

#     def set_global_rank(self, rank:int) -> None:
#         pass

#     @staticmethod
#     def detect() -> bool:
#         """Detects the environment settings corresponding to this cluster and returns ``True`` if they match."""
#         return True


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
    parser.add_argument('--data-root', default="./data/ARDM_temp_homolog_family_dataset.csv", type=Path,
                        help='path to dataset root directory')

    parser.add_argument('--tb_logger_path', default=None, type=str,
                        help='checkpoint path to pretrained weights')
    parser.add_argument('--tb_logger_folder', default=None, type=str,
                        help='tensorboard path containing checkpoints')
    parser.add_argument('--resume_from_checkpoint', default='None', type=str,
                        help='checkpoint path to last model iteration (usually last.ckpt)')


    parser.add_argument('--dataset', default="normal", type=str,
                        choices=['normal', 'sequence'],
                        help='which dataset to train on')
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loader workers')
    parser.add_argument('--warmup-steps', default=500, type=int,
                        help='number of learning rate warmup steps')
    parser.add_argument('--total-steps', default=1000, type=int,
                        help='total number of steps of minibatch gradient descent')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='mini-batch size')
    parser.add_argument('--weight-decay', default=1e-6, type=float,
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
    parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                        help='path to checkpoint directory')
    parser.add_argument('--checkpoint-prefix', default='channels',
                        help='prefix for local checkpoint')
    parser.add_argument('--device', default='cuda', type=str,
                        help='computational device')
    parser.add_argument('--model_option', default='transformer', type=str,
                        choices=['Unet', 'transformer'],
                        help='Choose model architecture')
    parser.add_argument('--download', default='True', type=str,
                        help='Download dataset')
    parser.add_argument('--swissprot_data_root', default='None', type=str,
                        help='path to SwissProt data')
    parser.add_argument('--pfam_data_root', default='None', type=str,
                        help='path to Pfam data')

    # Finetuning
    parser.add_argument('--finetune', default='False', type=str,
                        help='flag to run finetuning')
    parser.add_argument('--pretrained_checkpoint', default='None', type=str,
                        help='path to .bin checkpoint file containing model weights')
    parser.add_argument('--finetune_last_n_blocks', default=-1, type=int,
                        help='Number of last transformer blocks to finetune (0: finetune all layers, -1: no layers)')


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
    parser.add_argument('--start_pfam_trainer', default='False', type=str,
                        help='decide whether we are running script that transitions from swissprot to swissprot+pfam')


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
    parser.add_argument('--image-size', default=16, type=int,
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
    parser.add_argument('--output_hist_folder', default='./outputs/training_history', type=str,
                        help='Path to save model weights')
    parser.add_argument('--output_folder', default='./outputs/', type=str,
                        help='Folder containing all the necessary subfolders')
    parser.add_argument('--save_hist_path', default='./outputs/training_history/transformer_MNIST_train_hist.csv', type=str,
                        help='Path to save history training logs')
    parser.add_argument('--version_name', default=None, type=str,
                        help='Name of version sub-directory')
    parser.add_argument('--resume_from_checkpoint_state_dict_path', default=None, type=str,
                        help='Path to the deepspeed pytorch ckpt saved as state dict...')
    return parser


def get_wrapper_args(parser):
    """

    """
    parser.add_argument('--hydra', action="store_true", 
                        help='Whether to run with Hydra.')
    parser.add_argument('--wandb_name', type=str, default=None, 
                        help='Weights&Biases run name.')
    parser.add_argument('--wandb_entity', type=str, default=None, 
                        help='Weights&Biases entity.')
    parser.add_argument('--wandb_project', type=str, default=None, 
                        help='Weights&Biases project.')
    parser.add_argument('--wandb_logging_dir', type=str, default="./logs", 
                        help='Weights&Biases logging directory.')
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
        data_shape: tuple=(16,16),
        num_classes: int=3
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
    # model = mod.get_model(
    #         args=args,
    #         data_shape=data_shape,
    #         num_classes=num_classes
    # ).cpu()

    model = mod.get_model(
            args=args,
            data_shape=data_shape,
            num_classes=num_classes
    ).to("cuda")

    PL_model = PL_mod.PL_ProtARDM(
        args=args,
        model=model,
        #ema_model=ema_model,
    )
    return PL_model


def save_history_log(
        args: any,
        hist_log: dict
    ) -> None:
    """
    Save training history logs to a CSV file.
    
    This function converts a dictionary containing training history metrics
    into a pandas DataFrame and saves it to the path specified in the arguments.
    
    Args:
        args: Configuration object containing a 'save_hist_path' attribute
        hist_log: Dictionary containing training metrics and history
        
    Returns:
        None
    """
    # history dataframe
    hist_df = pd.DataFrame(hist_log)
    hist_df.to_csv(args.save_hist_path, index=False)
    return


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
    print('Total number of model parameters:', total_params)
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
    # save model
    convert_zero_checkpoint_to_fp32_state_dict(
        args.resume_from_checkpoint,
        args.resume_from_checkpoint + '/single_model.pth'
    )
    # load weights
    loaded_model = PL_model.load_from_checkpoint(
            args.resume_from_checkpoint + '/single_model.pth',
            args=args,
            model=PL_model.model
    )
    return loaded_model


def save_model(
        args: any,
        checkpoint_path: str,
        PL_model: pl.LightningModule,
        trainer: Trainer
    ) -> None:
    """
    Save a PyTorch Lightning model trained with DeepSpeed.
    
    This function handles the conversion of a DeepSpeed ZeRO checkpoint into 
    standard PyTorch state dictionaries. It saves the model both in Lightning 
    format and as a pure PyTorch state dictionary. If an EMA (Exponential Moving 
    Average) model is present, it will also be saved. The function additionally 
    generates parameter statistics and performs type checking to ensure consistency.
    
    Note: This function only executes operations on the global_zero process when 
    running in a distributed environment.
    
    Args:
        args: Configuration object containing model parameters
        checkpoint_path: Directory where checkpoint files will be saved
        PL_model: PyTorch Lightning model to be saved
        
    Returns:
        None
    """
    image_size = args.image_size
    num_classes = args.num_classes

    # once saved via the model checkpoint callback...
    # we have a saved folder containing the deepspeed checkpoint rather than a single file
    checkpoint_folder_path = checkpoint_path + '/last.ckpt'
    if trainer.is_global_zero:
        single_ckpt_path = checkpoint_path + '/single_model.pth'
        # magically convert the folder into a single lightning loadable pytorch file (works for ZeRO 1,2,3)
        convert_zero_checkpoint_to_fp32_state_dict(checkpoint_folder_path, single_ckpt_path)
        print('Save model')
        new_temp_model = mod.get_model(
            args=args,
            data_shape=(image_size, image_size),
            num_classes=num_classes
        ).cpu()
        loaded_model = PL_mod.PL_ProtARDM.load_from_checkpoint(single_ckpt_path, args=args, model=new_temp_model)
        # make sure that datatypes or the same ...
        if PL_model.dtype != loaded_model.dtype:
            msg = "Data types are not matching."
            msg += f" Expected {PL_model.dtype}. Got: {loaded_model.dtype} (from loaded)"
            assert False, msg
        else:
            # save model state dict without pytorch lightning wrapper
            torch.save(loaded_model.model.state_dict(), checkpoint_path + '/state_dict.pth')
            if hasattr(loaded_model, 'ema_model'):
                print('Also saving EMA model...')
                torch.save(loaded_model.ema_model.state_dict(), checkpoint_path + '/state_dict_ema.pth')
            # check model params
            get_model_params(
                    checkpoint_path + '/params.csv',
                    model=loaded_model.model
            )
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
    torch.cuda.empty_cache()
    # torch.xpu.empty_cache()
    gc.collect()
    return


def retrieve_all_args():
    """
    Collect and consolidate all command line arguments from various components.
    
    This function creates an ArgumentParser and populates it with arguments 
    from multiple sources including general parameters, model-specific settings,
    architecture-specific parameters, and file path configurations. It then
    parses the command line input and returns the complete argument object.
    
    This serves as the central point for collecting all configuration parameters
    needed across the entire application.
    
    Args:
        None
        
    Returns:
        argparse.Namespace: A namespace object containing all parsed arguments
    """
    parser = argparse.ArgumentParser(description='Stage 3: ProteoScribe')
    get_args(parser=parser)
    get_model_args(parser=parser)
    mod.add_model_args(parser=parser)
    get_path_args(parser=parser)
    get_wrapper_args(parser=parser)
    args = parser.parse_args()

    # Type conversions
    args.resume_from_checkpoint = nonestr_to_none(args.resume_from_checkpoint)
    args.download = str_to_bool(args.download)
    args.swissprot_data_root = nonestr_to_none(args.swissprot_data_root)
    args.pfam_data_root = nonestr_to_none(args.pfam_data_root)
    args.start_pfam_trainer = str_to_bool(args.start_pfam_trainer)
    args.finetune = str_to_bool(args.finetune)
    args.pretrained_checkpoint = nonestr_to_none(args.pretrained_checkpoint)
    
    return args


def load_data(
        args, *,
        swissprot_data_root,
        pfam_data_root,
        facilitator,
    ):
    """
    Initialize and prepare a data module for protein sequence datasets.
    
    This function creates a PyTorch Lightning data module that handles loading
    and preprocessing protein sequence data from HDF5 files. It supports both
    SwissProt and Pfam datasets, and can configure different data groupings
    based on the specified facilitator (MMD or MSE).
    
    Args:
        args: Configuration object containing dataset paths and parameters
              including swissprot_data_root, pfam_data_root, and facilitator
        
    Returns:
        A configured PyTorch Lightning data module ready for use in training
    """
    data_module = PL_mod.HDF5_PFamDataModule(
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            valid_size=args.valid_size,
                            seed=args.seed,
                            diffusion_steps=args.diffusion_steps,
                            image_size=args.image_size,
                            swissprot_path=swissprot_data_root,
                            pfam_path=pfam_data_root,
                            group_name=facilitator + '_data'  # Uses facilitator type as prefix
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
    num_classes = args.num_classes
    num_nodes = args.num_nodes
    batch_size = args.batch_size

    args.traindata_len = len(data_module.train_dataloader()) // gpu_devices // acc_grad_batches
    print('Length of dataloader:', len(data_module.train_dataloader()))
    print('Numer of devices:', gpu_devices)
    print('Number of nodes:', num_nodes)
    print('Batch size:', batch_size)
    print('Length of dataloader per device:', len(data_module.train_dataloader()) // gpu_devices)
    print(f'Length of a training epoch in batch gradient updates: {args.traindata_len}')
    w, h = image_size, image_size
    # Ensure diffusion steps are sufficient for data dimensions
    if diffusion_steps < int(w*h):
        print('Make sure that the number of diffusion steps is equal to or greather than the data cardinality')
    help_tools.print_gpu_initialization()
    # Compile model architecture
    PL_model = compile_model(
            args=args,
            data_shape=(image_size, image_size),
            num_classes=num_classes,
    )
    print('Model size:', sum(p.numel() for p in PL_model.model.parameters()))
    return PL_model


def load_pretrained_weights(
    PL_model,
    checkpoint_path: str
    ):
    """
    Load pretrained model weights from a .bin checkpoint file.
    
    This function loads model weights saved as a PyTorch state dictionary
    from a .bin file and applies them to the provided PyTorch Lightning model.
    It handles missing keys and extra keys gracefully, reporting any issues
    while still loading compatible weights.
    
    Args:
        PL_model: PyTorch Lightning model to load weights into
        checkpoint_path: Path to the .bin checkpoint file containing model weights
        
    Returns:
        The model with loaded pretrained weights
    """
    print(f"Loading pretrained weights from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load the state dictionary from the .bin file
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Load weights into the model, handling potential key mismatches
    missing_keys, unexpected_keys = PL_model.model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    
    print("Pretrained weights loaded successfully")
    return PL_model


def freeze_except_last_n_blocks(PL_model, n_blocks):
    """
    Freeze all model parameters except those in the last n transformer blocks.
    
    This enables efficient finetuning by only updating parameters in the last
    n transformer blocks while keeping all other layers frozen. This approach
    reduces computational cost and can help prevent catastrophic forgetting.
    
    Args:
        PL_model: PyTorch Lightning model with loaded pretrained weights
        n_blocks: Number of last transformer blocks to keep trainable (default: 1)
                  If n_blocks=0, all layers will be trainable (no freezing)
                  If n_blocks >= total blocks, all blocks will be trainable
        
    Returns:
        The model with frozen parameters (except last n blocks)
    """
    if n_blocks == 0:
        print("n_blocks=0: All parameters will remain trainable (no freezing)")
        return PL_model
    
    print(f"Freezing all parameters except the last {n_blocks} transformer block(s)...")
    
    # First, freeze all parameters
    for param in PL_model.model.parameters():
        param.requires_grad = False
    
    # Then unfreeze only the last n transformer blocks
    # The model structure is: PL_model.model.transformer.transformer_blocks
    if hasattr(PL_model.model, 'transformer'):
        transformer = PL_model.model.transformer
        if hasattr(transformer, 'transformer_blocks') and len(transformer.transformer_blocks) > 0:
            total_blocks = len(transformer.transformer_blocks)
            blocks_to_unfreeze = min(n_blocks, total_blocks)
            
            # Unfreeze the last n blocks
            for i in range(total_blocks - blocks_to_unfreeze, total_blocks):
                block = transformer.transformer_blocks[i]
                for param in block.parameters():
                    param.requires_grad = True
            
            print(f"Unfroze last {blocks_to_unfreeze} transformer block(s) out of {total_blocks} total blocks")
            if blocks_to_unfreeze < n_blocks:
                print(f"Note: Requested {n_blocks} blocks but only {total_blocks} exist in model")
        else:
            print("Warning: Could not find transformer_blocks in model")
    else:
        print("Warning: Could not find transformer attribute in model")
    
    # Also unfreeze the final output layers (norm and out) for better finetuning
    if hasattr(PL_model.model, 'transformer'):
        transformer = PL_model.model.transformer
        if hasattr(transformer, 'norm'):
            for param in transformer.norm.parameters():
                param.requires_grad = True
            print("Unfroze final LayerNorm")
        if hasattr(transformer, 'out'):
            for param in transformer.out.parameters():
                param.requires_grad = True
            print("Unfroze final output layer")
    
    # Count trainable vs frozen parameters
    trainable_params = sum(p.numel() for p in PL_model.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in PL_model.model.parameters())
    frozen_params = total_params - trainable_params
    
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Frozen parameters: {frozen_params:,} ({100 * frozen_params / total_params:.2f}%)")
    
    return PL_model


###########################
##  Training Entrypoint  ##
###########################

def train_model(
        args,
        PL_model,
        data_module,
        ds_config=None,
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
    # Note: Nested args must be accessed via dictionaries, not attributes.
    tb_logger_path = args.tb_logger_path
    tb_logger_folder = args.tb_logger_folder
    version_name = args.version_name
    log_every_n_steps = args.log_every_n_steps
    pfam_data_root = args.pfam_data_root
    gpu_devices = args.gpu_devices
    num_nodes = args.num_nodes
    acc_grad_batches = args.acc_grad_batches
    epochs = args.epochs
    start_pfam_trainer = args.start_pfam_trainer  # Expect bool
    assert isinstance(start_pfam_trainer, bool), "start_fpam_trainer not bool"
    max_steps = args.max_steps
    val_check_interval = args.val_check_interval
    limit_val_batches = args.limit_val_batches
    lr = args.lr
    resume_from_checkpoint = args.resume_from_checkpoint  # Expect str or None
    assert resume_from_checkpoint is None or (
            isinstance(resume_from_checkpoint, str) and 
            resume_from_checkpoint != "None"
        ), f"resume_from_checkpoint should be str or None (not str `None`)." + \
           f" Got {resume_from_checkpoint} (type {type(resume_from_checkpoint)})"
    precision = args.precision
    
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
    
    strategy = DeepSpeedStrategy(
        config=ds_config
    )

    # Set up TensorBoard logging
    tb_logger = TensorBoardLogger(
        os.path.join(tb_logger_path, tb_logger_folder), 
        version=version_name
    )

    # Set up Weights&Biases logging
    wandb_logger = WandbLogger(
        name=args.wandb_name,
        save_dir=args.wandb_logging_dir,
        project=args.wandb_project, 
        entity=args.wandb_entity, 
        config=args, 
        tags=args.wandb_tags,
        group=args.version_name,
        log_model="all",
    )  # TODO: investigate "all"

    # Monitor learning rate changes
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    # Monitor GPU usage
    gpu_logger = pl.callbacks.DeviceStatsMonitor()

    # Configure checkpoint strategy based on dataset type
    if pfam_data_root is None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(tb_logger_path, tb_logger_folder, 'checkpoints', version_name),
            save_top_k=2,
            verbose=True,
            monitor='val_loss',
            mode="min",
            save_last=True
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(tb_logger_path, tb_logger_folder, 'checkpoints', version_name),
            save_top_k=2,
            verbose=True,
            monitor='val_loss',
            mode="min",
            save_last=True,
            every_n_train_steps=log_every_n_steps  # Save checkpoints periodically by steps
        )

    # Define common trainer parameters 
    trainer_params = {
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'enable_checkpointing': True,
        'devices': gpu_devices,
        'num_nodes': num_nodes,
        'accelerator': 'cuda',
        'strategy': 'deepspeed_stage_2',  # TODO: use strategy defined above?
        'accumulate_grad_batches': acc_grad_batches,
        'logger': [tb_logger, wandb_logger],
        'log_every_n_steps': log_every_n_steps,
        'callbacks': [checkpoint_callback, lr_monitor, gpu_logger],
        # 'plugins': [MyClusterEnvironment()]  # added a la multinode_PL_train_stage3
    }

    # Configure training mode: epoch-based or step-based
    if pfam_data_root is None:
        trainer_params['max_epochs'] = epochs
    else:
        if start_pfam_trainer:
            print('load weights from swissprot phase...')
            PL_model = get_deepspeed_model(
                    args=args,
                    PL_model=PL_model
            )

        trainer_params['max_steps'] = max_steps
        trainer_params['val_check_interval'] = val_check_interval
        trainer_params['limit_val_batches'] = limit_val_batches

        trainer_params['accelerator'] = 'gpu'
        trainer_params['devices'] = gpu_devices
        # trainer_params['num_nodes'] = num_nodes
        trainer_params['precision'] = precision

    # Initialize trainer with configured parameters
    trainer = Trainer(**trainer_params)

    # wrap optimizer and model with intel extension for pytorch 
    optimizer = torch.optim.AdamW(PL_model.parameters(), lr=lr)
    # PL_model, optimizer = ipex.optimize(PL_model, optimizer=optimizer, dtype=torch.float32)

    # Handle different training scenarios
    if args.finetune:
        # Finetuning
        print("Start finetuning")
        trainer.fit(PL_model, data_module)
    else:
        # Pretraining
        if resume_from_checkpoint is None:
            print("Train from scratch")
            trainer.fit(PL_model, data_module)
        elif start_pfam_trainer:
            print('Start training Proteoscribe in phase 2 ...')
            trainer.fit(PL_model, data_module)
        else:
            print('Continue training Proteoscribe in phase 2 ...')
            trainer.fit(PL_model, data_module, ckpt_path=resume_from_checkpoint)

    help_tools.print_gpu_initialization()

    # Save trained model in multiple formats
    save_model(
            args=args,
            checkpoint_path=checkpoint_callback.dirpath,
            PL_model=PL_model,
            trainer=trainer,
    )


def main(args, use_hydra=False, ds_config=None,):

    SIZE = MPI.COMM_WORLD.Get_size() # Total number of processes
    RANK = MPI.COMM_WORLD.Get_rank() # Global rank of the process
    LOCAL_RANK = os.environ.get('PALS_LOCAL_RANKID', '0') # Local rank of the process on the node
    NODE_RANK = os.environ.get('PALS_NODE_RANKID', '0') # Node rank of the process
    
    # ----- Process passed parameters -----
    seed = args.seed
    swissprot_data_root = args.swissprot_data_root
    pfam_data_root = args.pfam_data_root
    facilitator = args.facilitator
    
    use_wandb = True
    wandb_entity = args.wandb_entity
    wandb_project = args.wandb_project
    wandb_dir = args.wandb_logging_dir
    wandb_tags = args.wandb_tags

    

    # ----- Clear the GPU cache -----
    clear_gpu_cache()

    # ----- For reproducibility -----
    if seed <= 0:
        # If the seed is not specified, generate a random seed and use this.
        seed = np.random.randint(2**32)
        args.seed = seed
    set_seed(seed)
     
    # ----- Load Data -----
    data_module = load_data(
        args=args,
        swissprot_data_root=swissprot_data_root,
        pfam_data_root=pfam_data_root,
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
        pretrained_checkpoint = args.pretrained_checkpoint
        if finetune_last_n_blocks < 0:
            # If flag is set to finetune and blocks not specified (default -1)
            # set to default actionable value 1
            finetune_last_n_blocks = 1
        if pretrained_checkpoint is None:
            msg = "Finetuning flag --finetune set to True but " \
                    "pretrained_checkpoint path not specified."
            print(msg)
            print("Proceeding with randomly initialized weights")
        elif os.path.exists(pretrained_checkpoint):
            PL_model = load_pretrained_weights(
                PL_model=PL_model,
                checkpoint_path=pretrained_checkpoint
            )
            # Freeze parameters based on user configuration
            PL_model = freeze_except_last_n_blocks(
                PL_model=PL_model,
                n_blocks=finetune_last_n_blocks
            )
        else:
            print(f"Warning: Pretrained checkpoint not found at {pretrained_checkpoint}")
            print("Proceeding with randomly initialized weights")
    else:
        pass

    # ----- Train Model -----
    train_model(
        args=args,
        PL_model=PL_model,
        data_module=data_module,
        ds_config=ds_config,
    )

    # ----- Clean up -----
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    args = retrieve_all_args()
    main(args)
