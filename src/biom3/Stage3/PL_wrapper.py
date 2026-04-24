# ----- PyTorch Core -----
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import OneHotCategorical

# ----- Retrieve available device -----
from biom3.backend.device import BACKEND_NAME, _XPU

# ----- PyTorch Lightning Framework -----
if BACKEND_NAME == _XPU:
    # lightning imports (from local installation)
    import lightning as pl
    from lightning import Trainer, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping
else:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import EarlyStopping

# ----- Distributed Training -----
#from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP  # Old import
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # Current FSDP implementation
from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        enable_wrap,
        wrap
)
import deepspeed  # DeepSpeed integration

# ----- Data & File Handling -----
import h5py  # HDF5 file format support
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import train_test_split

# ----- Optimization & Scheduling -----
from transformers.optimization import Adafactor  # Efficient optimizer for transformers
from deepspeed.ops.adam import DeepSpeedCPUAdam  # Memory-efficient Adam implementation
from transformers import get_cosine_schedule_with_warmup  # Learning rate scheduler

# ----- Standard Libraries & Utilities -----
import functools
import math
import copy

# ----- Custom Modules & Project-Specific Imports -----
from biom3.Stage3.DSEma import moving_average, clone_zero_model  # EMA implementation
import biom3.Stage3.transformer_training_helper as trainer_tools
import biom3.Stage3.eval_metrics as eval_funcs
import biom3.Stage3.preprocess as prep
from biom3.backend.device import print_gpu_initialization, setup_logger

logger = setup_logger(__name__)


class PL_ProtARDM(pl.LightningModule):
    """
    PyTorch Lightning module for Protein Autoregressive Diffusion Models.
    
    This class implements a Lightning wrapper around the protein diffusion model,
    handling training, validation, optimization, and performance tracking. It follows
    the autoregressive diffusion process with conditional sampling for protein sequence
    generation guided by text embeddings.
    
    The model uses a masked autoregressive approach where sequence tokens are
    progressively generated in a randomized order, with each prediction conditioned
    on previously generated tokens and a text embedding that guides the generation.
    
    Key features:
    - Supports multiple optimizers (AdamW, Adafactor, Adam, DeepSpeedCPUAdam)
    - Configurable learning rate schedulers (cosine warmup, exponential decay)
    - Comprehensive metrics tracking during training and validation
    - Compatible with distributed training via DeepSpeed
    """

    def __init__(
            self,
            args: any,
            model: nn.Module,
            #ema_model: nn.Module,
        ):
        """
        Initialize the PyTorch Lightning module for protein diffusion.
        
        Args:
            args: Configuration object containing training and model parameters
            model: Neural network module implementing the diffusion model architecture
            #ema_model: Optional Exponential Moving Average model for improved inference stability
        """
        super().__init__()
        #self.save_hyperparameters()

        # arguments
        self.script_args = args

        # the whole model
        self.model = model
        #self.ema_model = ema_model

        #clone_zero_model(self.model, self.ema_model, zero_stage=3)
        ##self.ema_model = copy.deepcopy(self.model)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y_c: torch.Tensor,
            ema=False,
        ) -> torch.Tensor:
        """
        Perform a forward pass through the model.
        
        This method routes the input through either the primary model
        or the EMA model (if enabled) to generate logits for the masked 
        protein sequence positions.
        
        Args:
            x: Tensor of shape [batch_size, 1, seq_length] containing the partially
               masked protein sequence
            t: Tensor of shape [batch_size] containing timestep indices
            y_c: Tensor containing conditional text embeddings to guide generation
            ema: Whether to use the EMA model for inference (if available)
            
        Returns:
            Tensor of logits for each possible amino acid at each position
        """
        if ema:
            logits = self.ema_model(x=x, t=t.view(-1,), y_c=y_c)
        else:
            logits = self.model(x=x, t=t.view(-1,), y_c=y_c)
        return logits
        #return F.softmax(logits, dim=1)


    #def on_train_batch_end(self, *args, **kwargs):
    #    clone_zero_model(self.model, self.ema_model, zero_stage=3)
    #    #moving_average(self.model, self.ema_model, beta=0.0, zero_stage=3)



    def configure_optimizers(self, ):
        """
        Configure optimizers and learning rate schedulers.
        
        This method sets up the training optimization based on the arguments provided:
        - Chooses between different optimizers (AdamW, Adafactor, Adam, DeepSpeedCPUAdam)
        - Configures appropriate learning rate schedulers based on script_args
          including cosine warmup or exponential decay options
        
        Returns:
            dict or optimizer: Configuration dictionary containing optimizer and
                              scheduler settings, or just the optimizer if no
                              scheduler is specified
        """
        choose_optim = self.script_args.choose_optim
        lr = self.script_args.lr
        weight_decay = self.script_args.weight_decay
        scheduler_gamma = self.script_args.scheduler_gamma
        traindata_len = self.script_args.traindata_len
        epochs = self.script_args.epochs

        if choose_optim == 'AdamW':

            if isinstance(self, FSDP):
                logger.info("Enter FSDP")
                optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

            else:
                optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        elif choose_optim == 'AdaFactor':
            optimizer = Adafactor(self.parameters(), lr=lr, weight_decay=weight_decay, relative_step=False)

        elif choose_optim == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        elif choose_optim == 'DeepSpeedCPUAdam':
            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=lr, weight_decay=weight_decay)

        if scheduler_gamma is not None:
            if isinstance(scheduler_gamma, str):
                if 'coswarmup' == scheduler_gamma.lower():
                    logger.info('Using cossine warmup scheduler with decay')
                    num_warmup_steps=traindata_len
                    num_training_steps=traindata_len * epochs
                    logger.info('Num_warmup_steps=%s', num_warmup_steps)
                    logger.info('Num_training_steps=%s', num_training_steps)

                    def _get_cosine_schedule_with_warmup_lr_lambda(
                        current_step: int, num_warmup_steps: int, num_training_steps: int, num_cycles: float
                    ):
                        if current_step < num_warmup_steps:
                            return float(current_step) / float(max(1, num_warmup_steps))
                        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

                    lr_lambda = functools.partial(
                        _get_cosine_schedule_with_warmup_lr_lambda,
                        num_warmup_steps=num_warmup_steps,
                        num_training_steps=num_training_steps,
                        num_cycles=0.5,
                    )
                    return {
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            "scheduler": optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1),
                            "interval": "step",
                        },
                    }

                    #return {
                    #    "optimizer": optimizer,
                    #    "lr_scheduler": {
                    #        "scheduler": get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps),
                    #        "interval": "step",
                    #    },
                    #}
            else:
                logger.info('Using Exponential learning rate decay / epoch with factor: %s', scheduler_gamma)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma, verbose=True),
                        "interval": "epoch",
                    },
                }

        else:
            return optimizer

        #else:
        #    print("Please make choose_option variable from these options: 'AdamW', 'AdaFactor', 'Adam', 'DeepSpeedCPUAdam'")

    def common_step(
            self,
            realization: torch.Tensor,
            realization_idx: any,
            stage: str) -> dict:
        """
        Common processing step used in both training and validation.
        
        This method handles:
        1. Reshaping input data
        2. Computing loss via the ELBO objective
        3. Calculating and logging metrics
        4. Managing memory utilization
        
        Args:
            realization: Input protein sequences, either as a tensor or a list
                        containing [sequence, text_embedding]
            realization_idx: Index of the current batch
            stage: String indicating the current stage ('train', 'val', or 'EMA_val')
            
        Returns:
            dict: Dictionary containing the computed loss
        """
        if isinstance(realization, list):

            # class labels
            y_c = realization[1]#.long()

            # input samples
            realization = realization[0]
            batch_size, seq_length = realization.size()

        realization = realization.reshape(batch_size, 1, seq_length).long()

        train_tuple = self.cond_elbo_objective(
                realization=realization,
                y_c=y_c,
                realization_idx=realization_idx,
                stage=stage,
                ema=True if 'ema' in stage.lower() else False,
        )

        if len(train_tuple) == 1:
            loss = train_tuple[0]
        else:
            loss = train_tuple[0]
            metrics = train_tuple[1]

        # if realization_idx == 0:
        #     if self.global_rank == 0:
        #         gpu_memory_usage = print_gpu_initialization()
        #     else:
        #         gpu_memory_usage = 0.0
        #     self.log(f"{stage}_gpu_memory_usage", gpu_memory_usage, sync_dist=True)

        # For train: log per-step, no cross-rank sync (grad allreduce already syncs).
        # For val: skip per-step logging; accumulate locally and sync once at epoch end.
        # Per-step sync_dist=True during val races with DeepSpeed ZeRO's grad
        # allreduce on XPU/oneCCL and deadlocks (see docs/bug_reports/).
        is_val = 'val' in stage
        log_kwargs = dict(on_step=not is_val, on_epoch=True, sync_dist=is_val)

        self.log(f"{stage}_loss", loss, prog_bar=True, **log_kwargs)
        if len(train_tuple) > 1:
            self.log(f"{stage}_prev_hard_acc",    metrics[0], prog_bar=True, **log_kwargs)
            self.log(f"{stage}_prev_soft_acc",    metrics[1], **log_kwargs)
            self.log(f"{stage}_fut_hard_acc",     metrics[2], prog_bar=True, **log_kwargs)
            self.log(f"{stage}_fut_soft_acc",     metrics[3], **log_kwargs)
            self.log(f"{stage}_current_hard_acc", metrics[4], prog_bar=True, **log_kwargs)
            self.log(f"{stage}_current_soft_acc", metrics[5], **log_kwargs)
            self.log(f"{stage}_current_ppl",      metrics[6], **log_kwargs)
            self.log(f"{stage}_prev_ppl",         metrics[7], **log_kwargs)
            self.log(f"{stage}_fut_ppl",          metrics[8], **log_kwargs)
            self.log(f"{stage}_pos_entropy",      metrics[9], **log_kwargs)

        if BACKEND_NAME == _XPU:
            torch.xpu.memory.empty_cache()
        elif BACKEND_NAME == "cuda":
            torch.cuda.empty_cache()
        return {'loss': loss}

    def training_step(
            self,
            realization: torch.Tensor,
            realization_idx: any):
        """
        Perform a single training step.
        
        This method is called by PyTorch Lightning during training.
        It delegates to common_step with the 'train' stage flag.
        
        Args:
            realization: Input protein sequences
            realization_idx: Index of the current batch
            
        Returns:
            dict: Dictionary containing the computed loss
        """
        return self.common_step(realization, realization_idx, stage='train')

    def validation_step(
            self,
            realization: torch.Tensor,
            realization_idx: any):
        """
        Perform a single validation step.
        
        This method is called by PyTorch Lightning during validation.
        It delegates to common_step with the 'val' stage flag.
        
        Args:
            realization: Input protein sequences
            realization_idx: Index of the current batch
        """
        self.common_step(realization, realization_idx, stage='val')
        #self.common_step(realization, realization_idx, stage='EMA_val')

    def on_fit_start(self):
        # Diagnostic: confirm which torch.distributed backend is actually in use.
        # On Aurora frameworks/2025.3.1 this should print 'xccl'. If it prints
        # 'ccl' or 'gloo' the DeepSpeedStrategy process_group_backend kwarg
        # isn't taking effect and oneCCL collectives aren't going through the
        # native xccl path.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            backend = torch.distributed.get_backend()
            import logging
            logging.getLogger().info(
                f"[rank {self.global_rank}] torch.distributed backend = {backend}"
            )

    def on_before_optimizer_step(self, optimizer):
        norms = [p.grad.detach().norm(2) for p in self.parameters() if p.grad is not None]
        if norms:
            total_norm = torch.stack(norms).norm(2)
            self.log("train_grad_norm", total_norm, on_step=True, on_epoch=False)

    def apply_OneHotCat(self, probs: torch.Tensor) -> any:
        """
        Create a OneHotCategorical distribution from input probabilities.
        
        This helper method converts model output probabilities into a
        proper distribution object for sampling and probability calculations.
        
        Args:
            probs: Tensor of shape [batch_size, num_classes, seq_length] 
                  containing probabilities
                  
        Returns:
            OneHotCategorical distribution with appropriate shape
        """
        return OneHotCategorical(probs=probs.permute(0,2,1))
        #return OneHotCategorical(probs=F.softmax(probs.permute(0,2,1), dim=-1))

    def cond_elbo_objective(
            self,
            realization: torch.Tensor,
            y_c: torch.Tensor,
            realization_idx: any,
            stage: str,
            ema=False,
        ):
        """
        Calculate the conditional Evidence Lower Bound (ELBO) objective.
        
        This method implements the core diffusion training objective:
        1. Generates a random sampling path and timestep
        2. Creates masks for different parts of the sequence
        3. Computes model predictions on masked sequences
        4. Calculates log probabilities and loss
        5. Computes performance metrics
        
        Args:
            realization: Tensor of shape [batch_size, 1, seq_length] containing 
                        protein sequences
            y_c: Tensor containing conditional text embeddings
            realization_idx: Index of the current batch
            stage: String indicating the current stage ('train', 'val')
            ema: Whether to use the EMA model for inference
            
        Returns:
            tuple: (loss, metrics) where metrics is a tuple of performance measurements
        """
        # Use Lightning's rank-local device (xpu:N for rank N) rather than the
        # CLI --device string ("xpu"), which resolves to xpu:0 on every rank and
        # crashes under plain DDP with "Expected all tensors to be on the same
        # device". DeepSpeedStrategy previously hid this by moving tensors
        # transparently before use.
        device = self.device

        bs, channel, seq_length = realization.size()

        # get a batch of random sampling paths
        sampled_random_path = trainer_tools.sample_random_path(bs, seq_length, device=device)
        # sample a set of random smapling steps for each individual training sequences in the current batch
        idx = trainer_tools.sample_random_index_for_sampling(bs, seq_length, device=device, option='random')
        # we create a mask that masks the location were we've already sampled
        random_path_mask = trainer_tools.create_mask_at_random_path_index(sampled_random_path, idx, bs, seq_length)
        # create a mask that masks the location where we are currently sampling
        current_path_mask = trainer_tools.create_sampling_location_mask(sampled_random_path, idx, bs, seq_length)
        # future sampling locations (i.e. >t)
        future_path_mask = trainer_tools.create_mask_at_future_path_index(sampled_random_path, idx, bs, seq_length)
        # tokenize realization
        real_tokens, bs, seq_length = trainer_tools.create_token_labels(self.script_args, realization)
        #real_tokens = realization.clone().squeeze(1)
        # mask realizations
        real_token_masked = trainer_tools.mask_realizations(real_tokens, random_path_mask)
        # conditional probs
        #probs = self(x=real_token_masked, t=idx, y_c=y_c, ema=ema)
        logits = self(x=real_token_masked, t=idx, y_c=y_c, ema=ema)
        logits = logits.float() # TN: Cast to float32 before softmax, added to fix "Expected parameter probs (Tensor of shape (32, 1024, 29)) of distribution OneHotCategorical() to satisfy the constraint Simplex(), but found invalid values" error

        conditional_prob = OneHotCategorical(logits=logits.permute(0,2,1))
        #conditional_prob = self.apply_OneHotCat(probs=probs)
        # evaluate the value of the log prob for the given realization
        log_prob = trainer_tools.log_prob_of_realization(self.script_args, conditional_prob, real_tokens)

        # compute an average over all the unsampled
        #log_prob_unsampled = trainer_tools.log_prob_of_unsampled_locations(log_prob.to(device), real_token_masked.to(device))
        log_prob_unsampled = trainer_tools.log_prob_of_unsampled_locations(log_prob, real_token_masked)
        #log_prob_unsampled = trainer_tools.log_prob_of_unsampled_locations(log_prob, real_token_masked, real_tokens)


        # compute an average loss i.e. negative average log-likelihood over the batch elements
        log_prob_weighted = trainer_tools.weight_log_prob(log_prob_unsampled, idx, seq_length)
        # compute an average loss i.e. negative average log-likelihood over the batch elements
        loss = trainer_tools.compute_average_loss_for_batch(log_prob_weighted)

        #if 'val' in stage:
        probs = F.softmax(logits, dim=1)
        metrics = self.performance_step(
                    real_tokens=real_tokens.cpu(),
                    idx=idx.cpu(),
                    sampled_random_path=sampled_random_path.cpu().float(),
                    probs=probs.cpu().float(),
                    conditional_prob=conditional_prob)

        return loss, metrics


    @torch.no_grad()
    def performance_step(
                    self,
                    real_tokens: torch.Tensor,
                    idx: torch.Tensor,
                    sampled_random_path: torch.Tensor,
                    probs: torch.Tensor,
                    conditional_prob: torch.Tensor
                    ) -> tuple:
        """
        Calculate performance metrics for the current batch.
        
        This method computes various accuracy and perplexity metrics for
        different parts of the sequence (previous, current, and future tokens)
        based on the diffusion timestep.
        
        Note: This method is decorated with @torch.no_grad() to prevent
        gradient computation during metric calculation.
        
        Args:
            real_tokens: Ground truth protein sequences
            idx: Tensor containing timestep indices for each sequence
            sampled_random_path: Random sampling paths for the batch
            probs: Softmax probabilities from the model
            conditional_prob: OneHotCategorical distribution for sampling
            
        Returns:
            tuple: A tuple containing various performance metrics:
                - prev_B_hard_acc: Hard accuracy for previous tokens
                - prev_B_soft_acc: Soft accuracy for previous tokens
                - fut_B_hard_acc: Hard accuracy for future tokens
                - fut_B_soft_acc: Soft accuracy for future tokens
                - current_B_hard_acc: Hard accuracy for current tokens
                - current_B_soft_acc: Soft accuracy for current tokens
                - current_ppl: Perplexity for current tokens
                - prev_ppl: Perplexity for previous tokens
                - fut_ppl: Perplexity for future tokens
                - pos_entropy: Average positional entropy
        """
        # create numerical token sequence
        sample_seq = torch.argmax(trainer_tools.sample_from_conditional(conditional_prob).cpu(), dim=1)

        # eval prev positions in terms of time
        prev_B_hard_acc, prev_B_soft_acc, fut_B_hard_acc, fut_B_soft_acc, current_B_hard_acc, current_B_soft_acc = eval_funcs.compute_acc_given_time_pos(
                real_tokens=real_tokens,
                sample_seq=sample_seq,
                sample_path=sampled_random_path,
                idx=idx
        )

        # compute ppl given time position
        current_ppl, prev_ppl, fut_ppl = eval_funcs.compute_ppl_given_time_pos(
                probs=probs,
                sample_path=sampled_random_path,
                idx=idx
        )

        # average positional entropy
        pos_entropy = trainer_tools.compute_pos_entropy(probs=probs).mean().item()

        metric_evals = (
                prev_B_hard_acc,
                prev_B_soft_acc,
                fut_B_hard_acc,
                fut_B_soft_acc,
                current_B_hard_acc,
                current_B_soft_acc,
                current_ppl,
                prev_ppl,
                fut_ppl,
                pos_entropy
        )

        return metric_evals



class PFamDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for protein sequence data from PFam and/or SwissProt databases.

    This module handles data loading, preprocessing, train/validation splitting,
    and DataLoader creation for protein sequence modeling. It supports loading
    from multiple data sources (SwissProt and PFam) and can merge them automatically
    when both are provided.

    The module uses protein_dataset from the prep module to create PyTorch Dataset
    objects and handles the entire data pipeline from raw data to batched DataLoaders.
    """
    def __init__(self, args):
        """
        Initialize the PFam DataModule and prepare the datasets.

        This method:
        1. Loads protein data from specified sources
        2. Preprocesses sequences and embeddings
        3. Splits data into training and validation sets
        4. Creates dataset objects for both splits

        Args:
            args: Configuration object containing:
                - data paths (swissprot_data_root, pfam_data_root)
                - valid_size: Fraction of data to use for validation
                - seed: Random seed for reproducible splitting
                - batch_size: Number of samples per batch
                - num_workers: Number of worker processes for data loading
        """
        super().__init__()
        self.args = args
        #df = pd.read_csv(args.data_root)
        #data = torch.load(args.data_root)
        data = self.load_data()
        num_seq_list, text_emb_list = prep.prepare_protein_data(
                args=args,
                data_dict=data
        )
        logger.info('Performing 80/20 random train/val split')
        num_seq_list_train, num_seq_list_val, text_emb_train, text_emb_val = train_test_split(num_seq_list,
                                                                                            text_emb_list,
                                                                                            test_size=args.valid_size,
                                                                                            #stratify=class_label_list,
                                                                                            random_state=args.seed)
        logger.info('Number of training samples: %s', len(num_seq_list_train))
        logger.info('Number of validation samples: %s', len(num_seq_list_val))
        self.train_dataset = prep.protein_dataset(
            num_seq_list=num_seq_list_train,
            text_emb=text_emb_train
        )
        self.val_dataset = prep.protein_dataset(
            num_seq_list=num_seq_list_val,
            text_emb=text_emb_val
        )

    def load_data(self):
        """
        Load protein sequence data from specified sources.

        This method checks the provided paths for SwissProt and PFam data
        and loads them accordingly. If both sources are available, it merges
        the data from both sources.

        Returns:
            dict: Dictionary containing protein sequences and their associated
                  text embeddings

        Raises:
            FileNotFoundError: If specified data files cannot be found
            ValueError: If neither SwissProt nor PFam data is available
        """
        try:
            logger.debug("swissprot_data_root=%s, pfam_data_root=%s", self.args.swissprot_data_root, self.args.pfam_data_root)
            if self.args.swissprot_data_root != "None":
                swissprot_data = torch.load(self.args.swissprot_data_root)
            else:
                swissprot_data=None
            if self.args.pfam_data_root != "None":
                pfam_data = torch.load(self.args.pfam_data_root)
            else:
                pfam_data=None
            if (self.args.swissprot_data_root != "None") and (self.args.pfam_data_root != "None"):
                return self.merge_and_append_values(dict1=swissprot_data, dict2=pfam_data)
            elif self.args.swissprot_data_root == "None":
                return pfam_data
            elif self.args.pfam_data_root == "None":
                return swissprot_data
            else:
                raise ValueError('Both SwissProt and Pfam datasets are unavailable.')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data file not found: {e}")

    def merge_and_append_values(self, dict1, dict2):
        """
        Merge two dictionaries by concatenating their values.

        This utility method combines data from two sources (e.g., SwissProt and PFam)
        by merging corresponding values for each key. If the values are lists,
        they are concatenated. If not, they are converted to lists and then merged.

        Args:
            dict1: First dictionary (e.g., SwissProt data)
            dict2: Second dictionary (e.g., PFam data)

        Returns:
            dict: A merged dictionary with combined values from both sources
        """
        merged_dict = {}
        # Combine all keys from both dictionaries
        all_keys = set(dict1) | set(dict2)
        for key in all_keys:
            values = []
            if key in dict1:
                values.append(dict1[key])
            if key in dict2:
                values.append(dict2[key])
            # Merge values for each key
            # This merges lists or appends non-list values
            merged_dict[key] = [item for sublist in values for item in (sublist if isinstance(sublist, list) else [sublist])]
        return merged_dict

    def train_dataloader(self):
        """
        Create a DataLoader for the training dataset.

        The DataLoader applies batching and enables multi-process data loading
        with the specified number of worker processes.

        Returns:
            DataLoader: PyTorch DataLoader configured for the training dataset
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        """
        Create a DataLoader for the validation dataset.

        Similar to train_dataloader but configured for evaluation (no shuffling)
        to ensure reproducible validation results.

        Returns:
            DataLoader: PyTorch DataLoader configured for the validation dataset
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False
        )

class HDF5Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for protein sequences stored in HDF5 format.
    
    This dataset loads protein sequences and their corresponding text embeddings 
    from a specified group within an HDF5 file. The protein sequences undergo 
    preprocessing via the HDF5_prepare_protein_data function to convert them into 
    numerical representation suitable for model input.
    
    The dataset is designed to work with HDF5 files containing at least:
    - 'sequence': Protein sequences stored as byte strings
    - 'text_to_protein_embedding': Numerical embeddings for each sequence
    - 'sequence_length': Length information for each sequence (optional)
    
    Note: This dataset keeps the HDF5 file open during its lifetime, so it's
    important to call the close() method when done to release resources.
    """
    def __init__(self, args, file_path, group_name):
        """
        Initialize the HDF5 protein dataset.
        
        Args:
            args: Configuration object containing preprocessing parameters
            file_path: Path to the HDF5 file containing protein data
            group_name: Name of the group within the HDF5 file to access
                       (e.g., 'MMD_data' or 'MSE_data')
        """
        self.file_path = file_path
        self.f = h5py.File(file_path, 'r')
        self.group = self.f[group_name]
        self.transform = prep.HDF5_prepare_protein_data
        self.args = args
        # Store data as attributes for easier access
        self.tensor_data = self.group["text_to_protein_embedding"]
        self.sequences = self.group["sequence"]
        
    def get_sequence_lengths(self):
        """
        Retrieve the lengths of all sequences in the dataset.
        
        Returns:
            numpy.ndarray: Array containing the length of each sequence
        """
        # Assumes there is a dataset in your HDF5 file that stores sequence lengths
        return self.group['sequence_length'][:]
        
    def __len__(self):
        """
        Get the total number of samples in the dataset.
        
        Returns:
            int: Number of protein sequences in the dataset
        """
        return len(self.group['text_to_protein_embedding'])
        
    def __getitem__(self, idx):
        """
        Retrieve a single sample by index.
        
        This method:
        1. Loads the text embedding directly from the HDF5 file
        2. Processes the protein sequence through the transform function
        3. Returns both as a tuple
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            tuple: (
                x_p: Tensor containing the preprocessed protein sequence 
                z_c: Tensor containing the text-to-protein embedding
            )
        """
        z_c = torch.from_numpy(self.tensor_data[idx])
        x_p = torch.tensor(
            self.transform(
                args=self.args,
                sequences=self.sequences[idx]
            )[0]
        )
        # Return a dictionary containing your data
        return (
            x_p, # sequence
            z_c # text_to_protein_embedding
        )
        
    def close(self):
        """
        Close the HDF5 file.
        
        This method should be called when the dataset is no longer needed
        to properly release file handles and system resources.
        """
        self.f.close()



class HDF5DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for HDF5-formatted protein sequence datasets.

    Supports loading from a primary data source and optionally one or more
    secondary sources.  Each source is split into train/validation, filtered
    by sequence length, and the splits from all sources are concatenated.

    Key features:
    - Memory-efficient access to large sequence datasets via HDF5
    - Supports combining data from multiple sources
    - Automatic filtering of sequences that exceed maximum length
    - Proper resource management with HDF5 file closing
    """
    def __init__(
        self, *,
        batch_size,
        num_workers,
        valid_size,
        seed,
        diffusion_steps,
        image_size,
        primary_path=None,
        secondary_paths=None,
        group_name='data',
        # Deprecated aliases
        swissprot_path=None,
        pfam_path=None,
    ):
        super().__init__()
        # Handle deprecated aliases
        if swissprot_path is not None and primary_path is None:
            primary_path = swissprot_path
        if pfam_path is not None and secondary_paths is None:
            secondary_paths = [pfam_path]

        self.args = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "valid_size": valid_size,
            "seed": seed,
            "diffusion_steps": diffusion_steps,
            "image_size": image_size,
        }
        self.primary_path = primary_path
        self.secondary_paths = secondary_paths or []
        self.group_name = group_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.seed = seed
        self.min_seq_length = diffusion_steps - 2

    def setup(self, stage=None):
        train_datasets = []
        val_datasets = []

        all_paths = []
        if self.primary_path is not None:
            all_paths.append(self.primary_path)
        all_paths.extend(self.secondary_paths)

        self.split_info = []
        for path in all_paths:
            dataset = HDF5Dataset(
                args=self.args,
                file_path=path,
                group_name=self.group_name,
            )
            train_idx, val_idx = self.split_indices(dataset)
            train_idx = self.filter_by_sequence_length(dataset, train_idx)
            val_idx = self.filter_by_sequence_length(dataset, val_idx)
            train_datasets.append(Subset(dataset, train_idx))
            val_datasets.append(Subset(dataset, val_idx))
            self.split_info.append({
                "path": path,
                "train_indices": train_idx,
                "val_indices": val_idx,
            })
            logger.info("Loaded dataset from %s (%d train, %d val)",
                        path, len(train_idx), len(val_idx))

        if not train_datasets:
            raise ValueError("No dataset paths provided")

        # Combine training and validation subsets
        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets)

    def split_indices(self, dataset):
        """
        Create train and validation indices for a dataset.

        This utility method creates a random split of the dataset indices
        based on the validation size specified in the arguments.

        Args:
            dataset: Dataset to split

        Returns:
            tuple: (train_indices, validation_indices)
        """
        # Create indices for splitting
        train_size = int((1 - self.valid_size) * len(dataset))
        indices = np.arange(len(dataset))
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        return indices[:train_size], indices[train_size:]

    def filter_by_sequence_length(self, dataset, indices):
        """
        Filter indices to include only sequences within the length threshold.

        This method removes sequences that are too long for the model's
        context window (determined by diffusion_steps).

        Args:
            dataset: Dataset containing the sequences
            indices: Indices to filter

        Returns:
            list: Filtered indices containing only valid-length sequences
        """
        # Assumes 'get_sequence_lengths' method exists in HDF5Dataset
        sequence_lengths = dataset.get_sequence_lengths()
        filtered_indices = [idx for idx in indices if sequence_lengths[idx] <= self.min_seq_length]
        logger.info("Original indices count: %s, Filtered indices count: %s", len(indices), len(filtered_indices))
        return filtered_indices

    def train_dataloader(self):
        """
        Create a DataLoader for the training dataset.

        Returns:
            DataLoader: PyTorch DataLoader for the training dataset with
                       batching and shuffling enabled
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        """
        Create a DataLoader for the validation dataset.

        Returns:
            DataLoader: PyTorch DataLoader for the validation dataset
                       (without shuffling to ensure reproducible validation)
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def teardown(self, stage=None):
        """
        Clean up resources when the DataModule is no longer needed.

        This method properly closes all open HDF5 files to prevent resource
        leaks, especially important for large datasets.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or None)
        """
        # Close HDF5 files properly
        if stage == 'fit' or stage is None:
            for dataset in self.train_dataset.datasets:
                dataset.dataset.close()
            for dataset in self.val_dataset.datasets:
                dataset.dataset.close()


# Backward compatibility alias
HDF5_PFamDataModule = HDF5DataModule

