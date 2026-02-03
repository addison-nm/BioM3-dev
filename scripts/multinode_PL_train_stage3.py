#!/usr/bin/env python3
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
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.strategies import DeepSpeedStrategy

# DeepSpeed is needed for the DeepSpeedStrategy
import deepspeed

# Custom modules
import Stage3_source.preprocess as prep
import Stage3_source.cond_diff_transformer_layer as mod
import Stage3_source.helper_funcs as help_tools
import Stage3_source.PL_wrapper as PL_mod






class MyClusterEnvironment(ClusterEnvironment):
    @property
    def creates_processes_externally(self) -> bool:
        """Return True if the cluster is managed (you don't launch processes yourself)"""
        return True

    def world_size(self) -> int:
        return int(os.environ["WORLD_SIZE"])

    def global_rank(self) -> int:
        return int(os.environ["RANK"])

    def local_rank(self) -> int:
        return int(os.environ["LOCAL_RANK"])

    def node_rank(self) -> int:
        return int(os.environ["NODE_RANK"])

    @property
    def main_address(self) -> str:
        return os.environ["MASTER_ADDRESS"]

    @property
    def main_port(self) -> int:
        return int(os.environ["MASTER_PORT"])

    def set_world_size(self, size:int) -> None:
        pass

    def set_global_rank(self, rank:int) -> None:
        pass

    @staticmethod
    def detect() -> bool:
        """Detects the environment settings corresponding to this cluster and returns ``True`` if they match."""
        return True


def set_distribution():

    # torch.backends.cuda.flash_sdp_enabled()
    print('Retrieve rank information...')
    os.environ['RANK'] = os.environ['PMI_RANK']# global
    os.environ['LOCAL_RANK'] = os.environ['PMI_LOCAL_RANK'] # local
    os.environ['WORLD_SIZE'] = os.environ['PMI_SIZE']
    os.environ['MASTER_ADDRESS'] = os.environ['MASTER_ADDR']
    os.environ['NODE_RANK'] = str(int(os.environ['RANK']) // 4)

    # Print the new values of RANK, LOCAL_RANK, and WORLD_SIZE
    print(f"RANK: {os.environ['RANK']}")
    print(f"LOCAL_RANK: {os.environ['LOCAL_RANK']}")
    print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}")
    print(f"MASTER_ADDRESS: {os.environ['MASTER_ADDRESS']}")
    print(f"MASTER PORT: {os.environ['MASTER_PORT']}")

    return

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
                        help='path to dataset root director')

    parser.add_argument('--tb_logger_path', default=None, type=str,
                        help='checkpoint path to pretrained weights')
    parser.add_argument('--tb_logger_folder', default=None, type=str,
                        help='tensorboard path containing checkpoints')
    parser.add_argument('--resume_from_checkpoint', default=None, type=str,
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
    parser.add_argument('--download', default=True, type=bool,
                        help='Download dataset')
    parser.add_argument('--swissprot_data_root', default=None, type=str,
                        help='path to SwissProt data')
    parser.add_argument('--pfam_data_root', default=None, type=str,
                        help='path to Pfam data')


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
    parser.add_argument('--num_workers', default=4, type=int,
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

def set_seed(args: any):
    """
    Set random seeds for reproducibility across different libraries.
    
    This function ensures deterministic behavior by setting identical random
    seeds for PyTorch, NumPy, and Python's built-in random module based on
    the seed value specified in the arguments.
    
    Args:
        args: Configuration object containing a 'seed' attribute
        
    Returns:
        None
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
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
    model = mod.get_model(
            args=args,
            data_shape=data_shape,
            num_classes=num_classes
    ).cpu()
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
        PL_model: pl.LightningModule
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
            data_shape=(args.image_size, args.image_size),
            num_classes=args.num_classes
        ).cpu()
        loaded_model = PL_mod.PL_ProtARDM.load_from_checkpoint(single_ckpt_path, args=args, model=new_temp_model)
        # make sure that datatypes or the same ...
        if PL_model.dtype != loaded_model.dtype:
            assert False, "Data types are not matching"
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
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError("Input must be 'True' or 'False'")


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
    gc.collect()
    return

def retieve_all_args();
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
    args = parser.parse_args()
    return args

def load_data(args):
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
                            args=args,
                            swissprot_path=args.swissprot_data_root,
                            pfam_path=args.pfam_data_root,
                            group_name=args.facilitator + '_data'  # Uses facilitator type as prefix
    )
    data_module.setup()
    return data_module

def load_model(
    args,
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
    args.traindata_len = len(data_module.train_dataloader()) // args.gpu_devices // args.acc_grad_batches
    print(f'Length of a training epoch in batch gradient updates: {args.traindata_len}')
    w, h = args.image_size, args.image_size
    # Ensure diffusion steps are sufficient for data dimensions
    if args.diffusion_steps < int(w*h):
        print('Make sure that the number of diffusion steps is equal to or greather than the data cardinality')
    help_tools.print_gpu_initialization()
    # Compile model architecture
    PL_model = compile_model(
            args=args,
            data_shape=(args.image_size,args.image_size),
            num_classes=args.num_classes
    )
    print('Model size:', sum(p.numel() for p in PL_model.model.parameters()))
    return PL_model

def train_model(
        args,
        PL_model,
        data_module
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
        
    Returns:
        None - results are saved to the paths specified in args
    """

    # Configure DeepSpeed optimization settings
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
    logger = TensorBoardLogger(args.tb_logger_path + args.tb_logger_folder, version=args.version_name)

    # Monitor learning rate changes
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    # Configure checkpoint strategy based on dataset type
    if args.pfam_data_root != 'None':
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.tb_logger_path + args.tb_logger_folder + '/checkpoints/' + args.version_name,
            save_top_k=2,
            verbose=True,
            monitor='val_loss',
            mode="min",
            save_last=True,
            every_n_train_steps=args.log_every_n_steps  # Save checkpoints periodically by steps
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.tb_logger_path + args.tb_logger_folder + '/checkpoints/' + args.version_name,
            save_top_k=2,
            verbose=True,
            monitor='val_loss',
            mode="min",
            save_last=True
        )

    # Define common trainer parameters 
    trainer_params = {
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'enable_checkpointing': True,
        'devices': args.gpu_devices,
        'num_nodes': args.num_nodes,
        'accelerator': 'cuda',
        'strategy': 'deepspeed_stage_2',
        'accumulate_grad_batches': args.acc_grad_batches,
        'logger': logger,
        'log_every_n_steps': args.log_every_n_steps,
        'callbacks': [checkpoint_callback, lr_monitor],
        'plugins': [MyClusterEnvironment()]
    }

    # Configure training mode: epoch-based or step-based
    if args.pfam_data_root == 'None':
        trainer_params['max_epochs'] = args.epochs
    else:
        if args.start_pfam_trainer.lower() == 'true':
            print('load weights from swissprot phase...')
            PL_model = get_deepspeed_model(
                    args=args,
                    PL_model=PL_model
            )

        trainer_params['max_steps'] = args.max_steps
        trainer_params['val_check_interval'] = args.val_check_interval
        trainer_params['limit_val_batches'] = args.limit_val_batches

    # Initialize trainer with configured parameters
    trainer = Trainer(**trainer_params)

    # Handle different training scenarios
    if args.resume_from_checkpoint == "None":
        print("Train from scratch")
        trainer.fit(PL_model, data_module)
    elif args.resume_from_checkpoint != 'None' and args.start_pfam_trainer.lower() =='true':
        print('Start training Proteoscribe in phase 2 ...')
        trainer.fit(PL_model, data_module)
    else:
        print('Continue training Proteoscribe in phase 2 ...')
        trainer.fit(PL_model, data_module, ckpt_path=args.resume_from_checkpoint)

    help_tools.print_gpu_initialization()

    # Save trained model in multiple formats
    save_model(
            args=args,
            checkpoint_path=checkpoint_callback.dirpath,
            PL_model=PL_model
    )


if __name__ == '__main__':

    #######################
    # Clear the GPU cache #
    #######################
    clear_gpu_cache()
    
    ########################
    # Get input argumenmts #
    ########################
    args = retrieve_all_args()

    ######################
    # For reproducbility #
    ######################
    set_seed(args=args)
    
    ###############
    # Set Machine # 
    ###############
    set_distribution()


    #############
    # Load Data #
    #############
    data_module = load_data(args=args)

    ##############
    # Load Model #
    ##############
    PL_model = load_model(
            args=args,
            data_module
    )
    
    ###############
    # Train Model #
    ###############
    train_model(
            args=args,
            PL_model=PL_model,
            data_module=data_module
    )


