#!/usr/bin/env python3

"""Training script for BioM3 Stage 3

Support for PyTorch Lightning, Hydra, and Weights&Biases

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

# # lightning imports (from local installation)
# import lightning as pl
# from lightning import Trainer
# from lightning.fabric.strategies import DeepSpeedStrategy
# from lightning.fabric.loggers import TensorBoardLogger
# from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
# from lightning.pytorch.callbacks import ModelCheckpoint

# to make the model and pytorch lightning compatible with Intel hardware i) import the ipex module and ii) set the device to 'xpu' for the model
# import intel_extension_for_pytorch as ipex

# DeepSpeed is needed for the DeepSpeedStrategy
import deepspeed

# Hydra
import hydra
from omegaconf import DictConfig, OmegaConf

# WandB
import wandb


# Custom modules
import Stage3_source.preprocess as prep
import Stage3_source.cond_diff_transformer_layer as mod
import Stage3_source.helper_funcs as help_tools
import Stage3_source.PL_wrapper as PL_mod


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
    num_classes = args.dataset["num_classes"]

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
    # torch.xpu.empty_cache()
    gc.collect()
    return


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
    num_classes = args.dataset.num_classes
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


def train_model(
        args,
        PL_model,
        data_module,
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
    # Note: Nested args must be accessed via dictionaries, not attributes.
    tb_logger_path = args.tb_logger_path
    tb_logger_folder = args.tb_logger_folder
    version_name = args.version_name
    log_every_n_steps = args.log_every_n_steps
    pfam_data_root = args.dataset["pfam_data_root"]
    gpu_devices = args.gpu_devices
    num_nodes = args.num_nodes
    acc_grad_batches = args.acc_grad_batches
    epochs = args.epochs
    start_pfam_trainer = args.start_pfam_trainer
    max_steps = args.max_steps
    val_check_interval = args.val_check_interval
    limit_val_batches = args.limit_val_batches
    lr = args.optimizer["lr"]
    resume_from_checkpoint = args.resume_from_checkpoint
    precision = args.precision
    
    # Configure DeepSpeed optimization settings
    ds_config = args.deepspeed
    strategy = DeepSpeedStrategy(
        config=ds_config
    )

    # Set up TensorBoard logging
    tb_logger = TensorBoardLogger(
        os.path.join(tb_logger_path, tb_logger_folder), 
        version=version_name
    )

    # Set up Weights&Biases logging
    wandb_logger = WandbLogger(log_model="all")  # TODO: investigate "all"

    # Monitor learning rate changes
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    # Configure checkpoint strategy based on dataset type
    if pfam_data_root != 'None':
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(tb_logger_path, tb_logger_folder, 'checkpoints', version_name),
            save_top_k=2,
            verbose=True,
            monitor='val_loss',
            mode="min",
            save_last=True,
            every_n_train_steps=log_every_n_steps  # Save checkpoints periodically by steps
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(tb_logger_path, tb_logger_folder, 'checkpoints', version_name),
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
        'devices': gpu_devices,
        'num_nodes': num_nodes,
        'accelerator': 'cuda',
        'strategy': 'deepspeed_stage_2',
        'accumulate_grad_batches': acc_grad_batches,
        'logger': [tb_logger, wandb_logger],
        'log_every_n_steps': log_every_n_steps,
        'callbacks': [checkpoint_callback, lr_monitor],
    }

    # Configure training mode: epoch-based or step-based
    if pfam_data_root == 'None':
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

        trainer_params['accelerator'] = 'auto'
        trainer_params['devices'] = gpu_devices
        # trainer_params['num_nodes'] = num_nodes
        trainer_params['precision'] = precision

    # Initialize trainer with configured parameters
    trainer = Trainer(**trainer_params)

    # wrap optimizer and model with intel extension for pytorch 
    optimizer = torch.optim.AdamW(PL_model.parameters(), lr=lr)
    # PL_model, optimizer = ipex.optimize(PL_model, optimizer=optimizer, dtype=torch.float32)

    # Handle different training scenarios
    if resume_from_checkpoint == "None":
        print("Train from scratch")
        trainer.fit(PL_model, data_module)
    elif resume_from_checkpoint != 'None' and start_pfam_trainer:
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


def print_config(cfg):
    for k in sorted(list(cfg.keys())):
        if hasattr(cfg[k], "keys"):
            print(f"{k}:")
            for kk in sorted(list(cfg[k].keys())):
                print(f"\t{kk}: {cfg[k][kk]}")
        else:
            print(f"{k}: {cfg[k]}")


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    
    # ----- Process config parameters ----- #
    print_config(cfg)
    seed = cfg.seed
    swissprot_data_root = cfg.dataset.swissprot_data_root
    pfam_data_root = cfg.dataset.pfam_data_root
    facilitator = cfg.facilitator

    
    # ----- Clear the GPU cache ----- #
    clear_gpu_cache()

    # ----- For reproducibility ----- #
    set_seed(seed)
     
    # ----- Load Data ----- #
    data_module = load_data(
        args=cfg,
        swissprot_data_root=swissprot_data_root,
        pfam_data_root=pfam_data_root,
        facilitator=facilitator,
    )

    # ----- Load Model ----- #
    PL_model = load_model(
        args=cfg,
        data_module=data_module
    )
    
    # ----- Prepare Weights&Biases coverage ----- #
    # See: https://docs.wandb.ai/models/integrations/hydra and
    # issue solution at: https://github.com/wandb/docs/issues/1964
    cfg_dict = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    entity = cfg["wandb"]["entity"]
    project = cfg["wandb"]["project"]
    wandb_dir = cfg["wandb"]["wandb_logging_dir"]
    # del cfg_dict["wandb"]  # don't need to log these
    
    # ----- Train Model ----- #
    with wandb.init(entity, project, config=cfg_dict, dir=wandb_dir) as run:
        train_model(
            args=run.config,
            PL_model=PL_model,
            data_module=data_module
        )
    
    # train_model(
    #     args=cfg,
    #     PL_model=PL_model,
    #     data_module=data_module,
    # )


if __name__ == '__main__':
    main()
    


