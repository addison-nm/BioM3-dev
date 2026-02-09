# pytorch fucntions
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.distributed as dist

# PL functions
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

# misc functions
import itertools
import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm
import time

# other learning packages
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# our packages
import Stage1_source.helper_funcs as helper_tools
import Stage1_source.preprocess as prep
import Stage1_source.model as mod


######################
# Default PL wrapper #
######################

class PL_PEN_CL(pl.LightningModule):


    def __init__(
            self,
            args: any,
            model: nn.Module,
            text_tokenizer: any,
            sequence_tokenizer: any
        ):

        super().__init__()
        # arguments
        self.script_args = args
        
        # model components
        self.model = model

        # tokenizers
        self.text_tokenizer = text_tokenizer
        self.sequence_tokenizer = sequence_tokenizer
        
        # validation tracker for outputs
        self.val_text_joint_latents = []
        self.val_seq_joint_latents = []

        # prediction tracker for outputs
        self.predict_text_joint_latents = []
        self.predict_seq_joint_latents = []
            
    def forward(
            self,
            x_t: torch.Tensor,
            x_s: torch.Tensor
        ) -> (
                torch.Tensor,
                torch.Tensor,
                torch.Tensor
        ):

        outputs = self.model(
                        x_t=x_t,
                        x_s=x_s
        )

        return (
                outputs['text_joint_latent'],
                outputs['seq_joint_latent'],
        )

    def training_step(
            self,
            batch: torch.Tensor,
            batch_idx: any,
        ) -> dict:
         
        if isinstance(batch, list):
            # split the 
            text_batch, protein_batch = batch
        
        # forward pass 
        z_t, z_s = self(
                    x_t=text_batch,
                    x_s=protein_batch
        )
        dist.barrier()

        # gather all tensors
        z_t_all = self.all_gather(z_t, sync_grads=True)
        dist.barrier()
        z_s_all = self.all_gather(z_s, sync_grads=True)
        
        # stack the embeddings
        z_t_all = z_t_all.view(-1, z_t.shape[-1])
        z_s_all = z_s_all.view(-1, z_s.shape[-1])
      
        # compute loss values
        loss, logits = self.model.compute_loss(
                protein_embeddings=z_s_all,
                text_embeddings=z_t_all
        )
        
        # track loss ...
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        # track metrics
        metric_dict = self.performance_metrics(logits=logits)
        for key in metric_dict:
            values = metric_dict[key]

            final_key = 'train_' + key
            self.log(final_key, metric_dict[key], prog_bar=True if 'f1' in key else False, on_step=True, on_epoch=True, sync_dist=True)
        
        if batch_idx == 0:
            gpu_memory_usage = helper_tools.print_gpu_initialization()
            self.log(f'gpu_memory_usage', gpu_memory_usage, sync_dist=True)

        return {'loss': loss}


    def validation_step(
            self,
            batch: list,
            batch_idx: any
        ) -> dict:

        # split the batch
        if isinstance(batch, list):
            # mean loss
            text_batch, protein_batch = batch
        
        # forward pass 
        z_t, z_s = self(
                    x_t=text_batch,
                    x_s=protein_batch
        )
        
        dist.barrier()
        # gather all tensors
        z_t_all = self.all_gather(z_t, sync_grads=True).view(-1, z_t.shape[-1])
        dist.barrier()
        z_s_all = self.all_gather(z_s, sync_grads=True).view(-1, z_s.shape[-1])
        
        # stack the embeddings
        z_t_all = z_t_all.view(-1, z_t.shape[-1])
        z_s_all = z_s_all.view(-1, z_s.shape[-1])

        # compute loss values
        loss, logits = self.model.compute_loss(
                protein_embeddings=z_s_all,
                text_embeddings=z_t_all
        )
        
        
        # track validation loss ...
        self.log('valid_loss', loss, prog_bar=True, sync_dist=True)

        # copmute validation metrics
        metric_dict = self.performance_metrics(logits=logits.detach().cpu())

        for key in metric_dict:
            values = metric_dict[key]
            final_key = 'valid_' + key
            self.log(final_key, metric_dict[key], prog_bar=True if 'f1' in key else False, sync_dist=True)
        
        # collect joint embedding
        self.val_text_joint_latents.append(z_t_all.detach().cpu())
        self.val_seq_joint_latents.append(z_s_all.detach().cpu())

        return {'valid_loss': loss}

    def on_validation_epoch_end(self):
        
        # collect and aggregate outputs from all validation steps
        val_z_t_joint = torch.cat(self.val_text_joint_latents, dim=0)
        val_z_s_joint = torch.cat(self.val_seq_joint_latents, dim=0)
        
        # compute singular values 
        text_log_sigma_k, S_text = self.compute_singular(val_z_t_joint.detach().cpu())
        protein_log_sigma_k, S_protein = self.compute_singular(val_z_s_joint.detach().cpu())
        
        # save image pngs for tracking dimensionality collapse
        self.save_png_to_tensorboard(
                data=text_log_sigma_k.numpy(),
                title='text',
        )
        self.save_png_to_tensorboard(
                data=protein_log_sigma_k.numpy(),
                title='protein'
        )

        # free memory
        self.val_text_joint_latents.clear()
        self.val_seq_joint_latents.clear()
        

        # compute effective rank (RankME):
        erank_text = self.compute_effective_rank(sigma_ks=S_text)
        erank_protein = self.compute_effective_rank(sigma_ks=S_protein)
        
        # log erank metrics
        self.log('valid_erank_text', erank_text, sync_dist=True)
        self.log('valid_erank_protein', erank_protein, sync_dist=True)


    def configure_optimizers(self,):

        params = [
                {"params": self.model.protein_encoder.parameters(), "lr": self.script_args.protein_encoder_lr},
                {"params": self.model.text_encoder.parameters(), "lr": self.script_args.text_encoder_lr},
                {"params": itertools.chain(
                    self.model.protein_projection.parameters(),
                    self.model.text_projection.parameters()
                    ),
                "lr": self.script_args.head_lr,
                "weight_decay": self.script_args.weight_decay}
        ]

        optimizer = torch.optim.AdamW(params, weight_decay=self.script_args.weight_decay)

        return {
                "optimizer": optimizer,
        }
    
    @torch.no_grad()
    def compute_class_metrics(
            self,
            outputs: torch.Tensor,
            targets: torch.Tensor,
            source: str
        ) -> dict:

        # convert torch tensors to numpy array
        outputs_np = outputs.numpy()
        targets_np = targets.numpy()

        # compute the metrics
        accuracy = accuracy_score(targets_np, outputs_np.round())
        precision = precision_score(targets_np, outputs_np.round(), average='micro')
        recall = recall_score(targets_np, outputs_np.round(), average='micro')
        f1 = f1_score(targets_np, outputs_np.round(), average='micro')

        return {
                f'{source}_accuracy': accuracy,
                f'{source}_precision': precision,
                f'{source}_recall': recall,
                f'{source}_f1': f1
        }

    @torch.no_grad()
    def performance_metrics(self, logits: torch.Tensor) -> tuple:

        logits = logits.cpu().float()

        # get probs
        p_text = F.softmax(logits, dim=-1) # prob of a given text captions aligning well with seq. pairs
        p_seq = F.softmax(logits.T, dim=-1) # prob of a given seq aligning well with text pairs
        p_tot = (p_seq + p_text) / 2 # total prob

        # get class labels
        y_pred_text = torch.argmax(p_text, dim=-1)
        y_pred_seq = torch.argmax(p_seq, dim=-1)
        y_pred = torch.argmax(p_tot, dim=-1)
        y_true = torch.arange(y_pred_text.shape[0])
        
        # compute class metrics
        text_metrics = self.compute_class_metrics(
                                    outputs=y_pred_text,
                                    targets=y_true,
                                    source='text'
        )
        seq_metrics = self.compute_class_metrics(
                                    outputs=y_pred_seq,
                                    targets=y_true,
                                    source='seq'
        )
        total_metrics = self.compute_class_metrics(
                                    outputs=y_pred,
                                    targets=y_true,
                                    source='total'
        )

        # combine dicts into one
        combined_dict = {}
        combined_dict.update(text_metrics)
        combined_dict.update(seq_metrics)
        combined_dict.update(total_metrics)

        return combined_dict
    
    @torch.no_grad()
    def compute_singular(self, inputs: torch.Tensor) -> (
            torch.Tensor,
            torch.Tensor
        ):

        # goal of this function: track for dimensionality collapse
        # inputs dim: (batch_size, emb_dim)

        mean_inputs = torch.mean(inputs, dim=0) # average over batch dimension
        norm_inputs = inputs - mean_inputs # normalize vectors
       
       # compute correlation matrix  #TODO: double check work...
        C = torch.zeros((norm_inputs.shape[-1], norm_inputs.shape[-1]))
        for sample_idx in range(norm_inputs.shape[0]):
            norm_vector = norm_inputs[sample_idx, :].unsqueeze(0)
            C += norm_vector.T @ norm_vector
        C *= 1/norm_vector.shape[0]

        _, S, _ = torch.linalg.svd(C, full_matrices=False)
        
        # return singular value indexes 
        log_sigma_k, _ = torch.sort(torch.log(S), descending=True)
        return (
                log_sigma_k,
                S
        )
        
    def compute_effective_rank(self, sigma_ks: torch.Tensor) -> torch.Tensor:
        """
        references:
            - Roy et al. The effective rank: a measure of effective dimensionality
            - Garrido et al. RankMe: Assessing the Downstream Performnace of Pretrained SS Reps by their Rank.
        """
        # sort the singular values
        sigma_ks, _ = torch.sort(sigma_ks, descending=True)
        
        # copute L1 norm for sing values.
        l1_norm_sigma = torch.norm(sigma_ks, p=1)
        
        # compute singular value distribution
        p_k = sigma_ks / l1_norm_sigma  + torch.finfo(torch.float).eps
        
        # compute Shannon entropy
        entropy = - torch.sum(p_k * torch.log(p_k))
    
        # get effective rank (RankME):
        erank = torch.exp(entropy)

        return erank

    def save_png_to_tensorboard(
            self,
            data: np.single,
            title: str,
            x_axis_label: str='Singular Value Rank Index',
            y_axis_label: str='Log of singular values',
            ):
    
        current_epoch = self.trainer.current_epoch
        
        # Plot the line
        fig, ax = plt.subplots(dpi=300)
        ax.plot(data)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_title(title)
        ax.set_ylim([-25,3])

        # Log the plot in TensorBoard 
        self.logger.experiment.add_figure(f'{title}_SingularValues_{current_epoch}', fig, current_epoch)

    def predict_step(
            self,
            batch: torch.Tensor,
            batch_idx: torch.Tensor,
            dataloder_idx: bool=False
        ) -> (  
                torch.Tensor,
                torch.Tensor
        ):


        if isinstance(batch, list):
            # mean loss
            text_batch, protein_batch = batch
            outputs = self(
                    x_t=text_batch,
                    x_s=protein_batch,
            )
        
        z_t_joint, z_p_joint = outputs

        self.predict_text_joint_latents.append(z_t_joint.detach().cpu())
        self.predict_seq_joint_latents.append(z_p_joint.detach().cpu())

        return outputs

    def on_predict_epoch_end(self, outputs=None):

        self.predict_text_joint_latents = torch.cat(self.predict_text_joint_latents).cpu()
        self.predict_seq_joint_latents = torch.cat(self.predict_seq_joint_latents).cpu()



##########################
# Masked-task PL wrapper #
##########################

class mask_PL_PEN_CL(pl.LightningModule):


    def __init__(
            self,
            args: any,
            model: nn.Module,
            text_tokenizer: any,
            sequence_tokenizer: any
    ):

        super().__init__()
        # arguments
        self.script_args = args
        
        # model components
        self.model = model
    
        # tokenizers
        self.text_tokenizer = text_tokenizer
        self.sequence_tokenizer = sequence_tokenizer
        
        # validation tracker for outputs
        self.val_text_joint_latents = []
        self.val_seq_joint_latents = []
    
        # prediction tracker for outputs
        self.predict_text_joint_latents = []
        self.predict_seq_joint_latents = []

    def forward(
            self,
            x_t: torch.Tensor,
            x_s: torch.Tensor,
            compute_masked_logits: bool=False
        ) -> (
                torch.Tensor,
                torch.Tensor,
                torch.Tensor
        ):

        outputs = self.model(
                        x_t=x_t,
                        x_s=x_s,
                        compute_masked_logits=compute_masked_logits
        )
        
        if compute_masked_logits:
            # forward pass for computing logits for masked language objective
            return (
                    outputs['text_masked_logits'],
                    outputs['protein_masked_logits']
            )
        else:
            # forward pass for computing latent embeddings in the joint space
            return (
                outputs['text_joint_latent'],
                outputs['seq_joint_latent'],
            )

    def training_step(
            self,
            batch: torch.Tensor,
            batch_idx: any,
        ) -> dict:
         
        if isinstance(batch, list):
            # split the data
            text_batch, protein_batch, text_mask_batch, protein_mask_batch = batch
        
        # forward pass 
        z_t, z_s = self(
                    x_t=text_batch,
                    x_s=protein_batch,
                    compute_masked_logits=False
        )
        dist.barrier()

        # gather all tensors
        z_t_all = self.all_gather(z_t, sync_grads=True)
        dist.barrier()
        z_s_all = self.all_gather(z_s, sync_grads=True)
        
        # stack the embeddings
        z_t_all = z_t_all.view(-1, z_t.shape[-1])
        z_s_all = z_s_all.view(-1, z_s.shape[-1])
      
        # compute loss values
        loss_align, logits = self.model.compute_loss(
                protein_embeddings=z_s_all,
                text_embeddings=z_t_all
        )
        
        # compute mask language model logits
        logits_t_mask, logits_s_mask = self(
                x_t=text_mask_batch,
                x_s=protein_mask_batch,
                compute_masked_logits=True
        )
        
        # compute mask language loss for biomedical expert model
        loss_text_mask = self.model.compute_masked_lang_loss(
                logits_masked=logits_t_mask,
                targets=text_batch,
                targets_masked=text_mask_batch,
                mask_token_id=self.text_tokenizer.mask_token_id
        )
        
        # compute mask language loss for protein expert model
        loss_sequence_mask = self.model.compute_masked_lang_loss(
                logits_masked=logits_s_mask,
                targets=protein_batch,
                targets_masked=protein_mask_batch,
                mask_token_id=self.sequence_tokenizer.mask_idx
        )
        
        
        # total loss
        loss = loss_align + loss_text_mask + loss_sequence_mask

        # track loss ...
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_loss_align', loss_align, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_loss_text_mask', loss_text_mask, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_loss_seq_mask', loss_sequence_mask, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        # track metrics
        metric_dict = self.performance_metrics(logits=logits)
        for key in metric_dict:
            values = metric_dict[key]

            final_key = 'train_' + key
            self.log(final_key, metric_dict[key], prog_bar=True if 'f1' in key else False, on_step=True, on_epoch=True, sync_dist=True)
        
        if batch_idx == 0:
            gpu_memory_usage = helper_tools.print_gpu_initialization()
            self.log(f'gpu_memory_usage', gpu_memory_usage, sync_dist=True)

        return {'loss': loss}


    def validation_step(
            self,
            batch: list,
            batch_idx: any
        ) -> dict:

        # split the batch
        if isinstance(batch, list):
            # mean loss
            text_batch, protein_batch, text_mask_batch, protein_mask_batch = batch
        
        # forward pass 
        z_t, z_s = self(
                    x_t=text_batch,
                    x_s=protein_batch
        )
        
        dist.barrier()
        # gather all tensors
        z_t_all = self.all_gather(z_t, sync_grads=True).view(-1, z_t.shape[-1])
        dist.barrier()
        z_s_all = self.all_gather(z_s, sync_grads=True).view(-1, z_s.shape[-1])
        
        # stack the embeddings
        z_t_all = z_t_all.view(-1, z_t.shape[-1])
        z_s_all = z_s_all.view(-1, z_s.shape[-1])

        # compute loss values
        loss_align, logits = self.model.compute_loss(
                protein_embeddings=z_s_all,
                text_embeddings=z_t_all
        )
        
        # compute mask language model logits
        logits_t_mask, logits_s_mask = self(
                x_t=text_mask_batch,
                x_s=protein_mask_batch,
                compute_masked_logits=True
        )
        
        # compute mask language loss for biomedical expert model
        loss_text_mask = self.model.compute_masked_lang_loss(
                logits_masked=logits_t_mask,
                targets=text_batch,
                targets_masked=text_mask_batch,
                mask_token_id=self.text_tokenizer.mask_token_id
        )
        
        # compute mask language loss for protein expert model
        loss_sequence_mask = self.model.compute_masked_lang_loss(
                logits_masked=logits_s_mask,
                targets=protein_batch,
                targets_masked=protein_mask_batch,
                mask_token_id=self.sequence_tokenizer.mask_idx
        )
        
        # total loss
        loss = loss_align + loss_text_mask + loss_sequence_mask

        # track validation loss ...
        self.log('valid_loss', loss, prog_bar=True, sync_dist=True)
        self.log('valid_loss_align', loss_align, prog_bar=True, sync_dist=True)
        self.log('valid_loss_text_mask', loss_text_mask, prog_bar=False, sync_dist=True)
        self.log('valid_loss_seq_mask', loss_sequence_mask, prog_bar=False, sync_dist=True)

        # copmute validation metrics
        metric_dict = self.performance_metrics(logits=logits.detach().cpu())

        for key in metric_dict:
            values = metric_dict[key]
            final_key = 'valid_' + key
            self.log(final_key, metric_dict[key], prog_bar=True if 'f1' in key else False, sync_dist=True)
        
        # collect joint embedding
        self.val_text_joint_latents.append(z_t_all.detach().cpu())
        self.val_seq_joint_latents.append(z_s_all.detach().cpu())

        return {'valid_loss': loss}

    def on_validation_epoch_end(self):
        
 #       # collect and aggregate outputs from all validation steps
 #      val_z_t_joint = torch.cat(self.val_text_joint_latents, dim=0)
 #       val_z_s_joint = torch.cat(self.val_seq_joint_latents, dim=0)
        
        # compute singular values 
 #       text_log_sigma_k, S_text = self.compute_singular(val_z_t_joint.detach().cpu())
 #       protein_log_sigma_k, S_protein = self.compute_singular(val_z_s_joint.detach().cpu())
        
        # save image pngs for tracking dimensionality collapse
 #       self.save_png_to_tensorboard(
 #               data=text_log_sigma_k.numpy(),
 #               title='text',
 #       )
 #       self.save_png_to_tensorboard(
 #               data=protein_log_sigma_k.numpy(),
 #               title='protein'
 #       )

        # free memory
        self.val_text_joint_latents.clear()
        self.val_seq_joint_latents.clear()
        

        # compute effective rank (RankME):
 #       erank_text = self.compute_effective_rank(sigma_ks=S_text)
 #       erank_protein = self.compute_effective_rank(sigma_ks=S_protein)
        
        # log erank metrics
 #       self.log('valid_erank_text', erank_text, sync_dist=True)
 #       self.log('valid_erank_protein', erank_protein, sync_dist=True)


    def configure_optimizers(self,):

        params = [
                {"params": self.model.protein_encoder.parameters(), "lr": self.script_args.protein_encoder_lr},
                {"params": self.model.text_encoder.parameters(), "lr": self.script_args.text_encoder_lr},
                {"params": itertools.chain(
                    self.model.protein_projection.parameters(),
                    self.model.text_projection.parameters()
                    ),
                "lr": self.script_args.head_lr,
                "weight_decay": self.script_args.weight_decay}
        ]

        optimizer = torch.optim.AdamW(params, weight_decay=self.script_args.weight_decay)

        return {
                "optimizer": optimizer,
        }
    
    @torch.no_grad()
    def compute_class_metrics(
            self,
            outputs: torch.Tensor,
            targets: torch.Tensor,
            source: str
        ) -> dict:

        # convert torch tensors to numpy array
        outputs_np = outputs.numpy()
        targets_np = targets.numpy()

        # compute the metrics
        accuracy = accuracy_score(targets_np, outputs_np.round())
        precision = precision_score(targets_np, outputs_np.round(), average='micro')
        recall = recall_score(targets_np, outputs_np.round(), average='micro')
        f1 = f1_score(targets_np, outputs_np.round(), average='micro')

        return {
                f'{source}_accuracy': accuracy,
                f'{source}_precision': precision,
                f'{source}_recall': recall,
                f'{source}_f1': f1
        }

    @torch.no_grad()
    def performance_metrics(self, logits: torch.Tensor) -> tuple:

        logits = logits.cpu().float()

        # get probs
        p_text = F.softmax(logits, dim=-1) # prob of a given text captions aligning well with seq. pairs
        p_seq = F.softmax(logits.T, dim=-1) # prob of a given seq aligning well with text pairs
        p_tot = (p_seq + p_text) / 2 # total prob

        # get class labels
        y_pred_text = torch.argmax(p_text, dim=-1)
        y_pred_seq = torch.argmax(p_seq, dim=-1)
        y_pred = torch.argmax(p_tot, dim=-1)
        y_true = torch.arange(y_pred_text.shape[0])
        
        # compute class metrics
        text_metrics = self.compute_class_metrics(
                                    outputs=y_pred_text,
                                    targets=y_true,
                                    source='text'
        )
        seq_metrics = self.compute_class_metrics(
                                    outputs=y_pred_seq,
                                    targets=y_true,
                                    source='seq'
        )
        total_metrics = self.compute_class_metrics(
                                    outputs=y_pred,
                                    targets=y_true,
                                    source='total'
        )

        # combine dicts into one
        combined_dict = {}
        combined_dict.update(text_metrics)
        combined_dict.update(seq_metrics)
        combined_dict.update(total_metrics)

        return combined_dict
    
    @torch.no_grad()
    def compute_singular(self, inputs: torch.Tensor) -> (
            torch.Tensor,
            torch.Tensor
        ):

        # goal of this function: track for dimensionality collapse
        # inputs dim: (batch_size, emb_dim)

        mean_inputs = torch.mean(inputs, dim=0) # average over batch dimension
        norm_inputs = inputs - mean_inputs # normalize vectors
       
       # compute correlation matrix  #TODO: double check work...
        C = torch.zeros((norm_inputs.shape[-1], norm_inputs.shape[-1]))
        for sample_idx in tqdm(range(norm_inputs.shape[0])):
            norm_vector = norm_inputs[sample_idx, :].unsqueeze(0)
            C += norm_vector.T @ norm_vector
        C *= 1/norm_vector.shape[0]

        _, S, _ = torch.linalg.svd(C, full_matrices=False)
        
        # return singular value indexes 
        log_sigma_k, _ = torch.sort(torch.log(S), descending=True)
        return (
                log_sigma_k,
                S
        )
        
    def compute_effective_rank(self, sigma_ks: torch.Tensor) -> torch.Tensor:
        """
        references:
            - Roy et al. The effective rank: a measure of effective dimensionality
            - Garrido et al. RankMe: Assessing the Downstream Performnace of Pretrained SS Reps by their Rank.
        """
        # sort the singular values
        sigma_ks, _ = torch.sort(sigma_ks, descending=True)
        
        # copute L1 norm for sing values.
        l1_norm_sigma = torch.norm(sigma_ks, p=1)
        
        # compute singular value distribution
        p_k = sigma_ks / l1_norm_sigma  + torch.finfo(torch.float).eps
        
        # compute Shannon entropy
        entropy = - torch.sum(p_k * torch.log(p_k))
    
        # get effective rank (RankME):
        erank = torch.exp(entropy)

        return erank

    def save_png_to_tensorboard(
            self,
            data: np.single,
            title: str,
            x_axis_label: str='Singular Value Rank Index',
            y_axis_label: str='Log of singular values',
            ):
    
        current_epoch = self.trainer.current_epoch
        
        # Plot the line
        fig, ax = plt.subplots(dpi=300)
        ax.plot(data)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_title(title)
        ax.set_ylim([-25,3])

        # Log the plot in TensorBoard 
        self.logger.experiment.add_figure(f'{title}_SingularValues_{current_epoch}', fig, current_epoch)

    def predict_step(
            self,
            batch: torch.Tensor,
            batch_idx: torch.Tensor,
            dataloder_idx: bool=False
        ) -> (  
                torch.Tensor,
                torch.Tensor
        ):


        if isinstance(batch, list):
            # mean loss
            text_batch, protein_batch = batch
            outputs = self(
                    x_t=text_batch,
                    x_s=protein_batch,
                    compute_masked_logits=False
            )
        
        z_t_joint, z_p_joint = outputs

        self.predict_text_joint_latents.append(z_t_joint.detach().cpu())
        self.predict_seq_joint_latents.append(z_p_joint.detach().cpu())

        return outputs

    def on_predict_epoch_end(self, outputs=None):

        self.predict_text_joint_latents = torch.cat(self.predict_text_joint_latents).cpu()
        self.predict_seq_joint_latents = torch.cat(self.predict_seq_joint_latents).cpu()



########################
# Pfam-task PL wrapper #
########################


class pfam_PL_PEN_CL(pl.LightningModule):


    def __init__(
            self,
            args: any,
            model: nn.Module,
            text_tokenizer: any,
            sequence_tokenizer: any
    ):

        super().__init__()
        # arguments
        self.script_args = args
        
        # model components
        self.model = model
    
        # tokenizers
        self.text_tokenizer = text_tokenizer
        self.sequence_tokenizer = sequence_tokenizer
        
        # validation tracker for outputs
        self.val_text_joint_latents = []
        self.val_seq_joint_latents = []
    
        # predictions...
        self.predict_text_joint_latents = [] 
        self.predict_seq_joint_latents = []

    def forward(
            self,
            x_t: torch.Tensor,
            x_p: torch.Tensor,
            compute_masked_logits: bool=False
        ) -> (
                torch.Tensor,
                torch.Tensor,
                torch.Tensor
        ):

        outputs = self.model(
                        x_t=x_t,
                        x_s=x_p,
                        compute_masked_logits=compute_masked_logits
        )
        
        if compute_masked_logits:
            # forward pass for computing logits for masked language objective
            return (
                    outputs['text_masked_logits'],
                    outputs['protein_masked_logits']
            )
        else:
            # forward pass for computing latent embeddings in the joint space
            return (
                outputs['text_joint_latent'],
                outputs['seq_joint_latent'],
            )


    def on_train_batch_start(self, batch, batch_idx):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_end_time = time.time()
        batch_time = batch_end_time - self.batch_start_time
        #print(f'Rank={dist.get_rank()}: time to process batch is {batch_time}')
        #self.log(f'batch_time_rank_{dist.get_rank()}', batch_time, on_step=True, on_epoch=False)

    def training_step(self, batch: torch.Tensor, batch_idx: any) -> dict:
        """
        Execute a single training step.

        Given a batch of data, this function processes both Swiss-Prot and Pfam data through the model, computes 
        various loss values including inter-modal, intra-modal, and masked language model losses for both text 
        and protein sequences. This function also computes and logs various metrics and GPU memory usage.

        Parameters:
        - batch: The input data batch. This can include multiple types of data.
        - batch_idx: Index of the current batch.

        Steps:
        1. Split the data into Swiss-Prot and Pfam batches, if the batch is a list.
        2. Forward pass the Swiss-Prot data through the model.
        3. Synchronize and gather embeddings from all GPUs.
        4. Forward pass the Pfam data through the model.
        5. Synchronize and gather Pfam embeddings from all GPUs.
        6. Concatenate Swiss-Prot and Pfam embeddings.
        7. Compute inter-modal and intra-modal loss values.
        8. Compute masked language model logits for the concatenated batch.
        9. Compute masked language loss for both text and protein sequences.
        10. Compute and log the total loss and individual loss components.
        11. Compute and log performance metrics.
        12. Log GPU memory usage at the start of training.

        Returns:
        - Dictionary containing the total loss value.

        Note: 
        This function is intended to be used within a distributed (multi-GPU) training context, as evident 
        from the use of barriers and gathering operations. It's designed to handle batches that contain both 
        Swiss-Prot and Pfam data, both being biological datasets used in multi-modal protein embeddings. 
        The function utilizes both inter-modal (between modalities) and intra-modal (within the same modality) 
        contrastive losses, as well as masked language modeling objectives similar to BERT's MLM objective.
        """

        # Check if the batch is a list and split data if so.
        if isinstance(batch, list):
            text_batch, protein_batch, text_mask_batch, protein_mask_batch, \
            pfam_text_batch, pfam_protein_batch, pfam_text_mask_batch, pfam_protein_mask_batch, \
            bool_pfam_vector = batch
    

        #print(f'rank={dist.get_rank()}: text size {text_batch.shape}')
        
        #start_time_forward_pass = time.time()
        # Forward pass with Swiss-Prot data.
        z_t_swiss, z_p_swiss = self(
            x_t=text_batch,
            x_p=protein_batch,
            compute_masked_logits=False
        )
        # Timer end and log
        #end_time_forward_pass = time.time()
        #print(f"Rank={dist.get_rank()}: Time taken for Swiss-Prot forward pass: {end_time_forward_pass - start_time_forward_pass} seconds.")

        # Ensure all GPUs are synchronized.
        dist.barrier()

        # Forward pass with Pfam data.
        z_t_pfam, z_p_pfam = self(
            x_t=pfam_text_batch,
            x_p=pfam_protein_batch,
            compute_masked_logits=False
        )
        dist.barrier()
        
        #Gather tensors from all GPUs.
        z_t_swiss_all = self.all_gather(z_t_swiss, sync_grads=True)
        dist.barrier()
        z_p_swiss_all = self.all_gather(z_p_swiss, sync_grads=True)

        # Reshape the embeddings.
        z_t_swiss_all = z_t_swiss_all.view(-1, z_t_swiss.shape[-1])
        z_p_swiss_all = z_p_swiss_all.view(-1, z_p_swiss.shape[-1])


        # Gather tensors from all GPUs.
        z_t_pfam_all = self.all_gather(z_t_pfam, sync_grads=True)
        dist.barrier()
        z_p_pfam_all = self.all_gather(z_p_pfam, sync_grads=True)

        # Reshape the embeddings.
        z_t_pfam_all = z_t_pfam_all.view(-1, z_t_pfam.shape[-1])
        z_p_pfam_all = z_p_pfam_all.view(-1, z_p_pfam.shape[-1])

        # Concatenate Swiss-Prot and Pfam embeddings.
        z_t_all = torch.cat((z_t_swiss_all, z_t_pfam_all), dim=0)
        z_p_all = torch.cat((z_p_swiss_all, z_p_pfam_all), dim=0)
        
        # Timer start
        #start_time_loss_computation = time.time()

        # Compute inter-modal loss.
        loss_align, logits = self.model.compute_inter_loss(
            protein_embeddings=z_p_all,
            text_embeddings=z_t_all,
            batch_size=z_p_all.shape[0] // 2
        )
        # Timer end and log
        #end_time_loss_computation = time.time()
        #print(f"Rank={dist.get_rank()}: Time taken for loss computation: {end_time_loss_computation - start_time_loss_computation} seconds.")


        # Compute intra-modal loss.
        loss_intra, cosine_similarity = self.model.compute_intra_loss(
            protein_embeddings=z_p_all,
            batch_size=z_p_all.shape[0] // 2
        )

        # Concatenate batches for masked language modeling.
        all_text_batch = torch.cat((text_batch, pfam_text_batch), dim=0)
        all_protein_batch = torch.cat((protein_batch, pfam_protein_batch), dim=0)
        all_text_mask_batch = torch.cat((text_mask_batch, pfam_text_mask_batch), dim=0)
        all_protein_mask_batch = torch.cat((protein_mask_batch, pfam_protein_mask_batch), dim=0)
        
        #TODO: timer start
        #start_time_mask_comp = time.time()

        # Compute masked language model logits.
        logits_t_mask, logits_s_mask = self(
            x_t=all_text_mask_batch,
            x_p=all_protein_mask_batch,
            compute_masked_logits=True
        )
        #end_time_mask_comp = time.time()
        #print(f"Rank={dist.get_rank()}: Time taken for mask predictions: {end_time_mask_comp - start_time_mask_comp} seconds.")


        # Compute masked language model loss for text data.
        loss_text_mask = self.model.compute_masked_lang_loss(
            logits_masked=logits_t_mask,
            targets=all_text_batch,
            targets_masked=all_text_mask_batch,
            mask_token_id=self.text_tokenizer.mask_token_id
        )

        # Compute masked language model loss for protein data.
        loss_sequence_mask = self.model.compute_masked_lang_loss(
            logits_masked=logits_s_mask,
            targets=all_protein_batch,
            targets_masked=all_protein_mask_batch,
            mask_token_id=self.sequence_tokenizer.mask_idx
        )


        if self.script_args.dataset_type == 'pfam':
            # Aggregate all computed losses.
            loss = loss_align + loss_intra + loss_text_mask + loss_sequence_mask
        
        elif self.script_args.dataset_type == 'pfam_ablated':
            # Aggregate all losses besides PFC.
            loss = loss_align + loss_text_mask + loss_sequence_mask
        else:
            # Add an assertion here
            assert self.script_args.dataset_type in ['pfam', 'pfam_ablated'], "Unexpected dataset_type value"
            sys.stderr.write("Unexpected dataset_type value\n")
            sys.exit(1)

        # Log the individual and total loss values.
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_loss_align', loss_align, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_loss_intra', loss_intra, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_loss_text_mask', loss_text_mask, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_loss_seq_mask', loss_sequence_mask, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        # Compute and log additional performance metrics.
        metric_dict = self.performance_metrics(logits=logits)
        for key in metric_dict:
            values = metric_dict[key]
            final_key = 'train_' + key
            self.log(final_key, metric_dict[key], prog_bar=True if 'f1' in key else False, on_step=True, on_epoch=True, sync_dist=True)

        # Log GPU memory usage at the beginning of the training.
        if batch_idx == 0:
            gpu_memory_usage = helper_tools.print_gpu_initialization()
            self.log(f'gpu_memory_usage', gpu_memory_usage, sync_dist=True)
        
        # log CPU memory
        memory_usage = helper_tools.print_memory_usage()
        self.log(f'memory_usage', memory_usage, sync_dist=True)
 
        return {'loss': loss}


    def validation_step(
            self,
            batch: torch.Tensor,
            batch_idx: any,
        ) -> dict:

        """
        `validation_step()`: Validates a single batch of data and computes loss and performance metrics.

        Parameters:
        - `self`: Reference to the current instance of the model or module.
        - `batch`: Input data, which might contain text and protein sequences, their corresponding masks, and additional data from both Swiss-Prot and Pfam datasets.
        - `batch_idx`: Identifier for the current batch.

        Functionality:
        1. Extracts and processes data from the given batch.
        2. Computes embeddings for Swiss-Prot and Pfam datasets.
        3. Concatenates these embeddings to form a unified representation.
        4. Computes various loss values: inter-modal, intra-modal, and masked language losses for both biomedical texts and protein sequences.
        5. Logs the computed loss values and other performance metrics, highlighting metrics such as F1-score.
        6. Collects and appends the joint embeddings of the batch for potential future use.

        Returns:
        - A dictionary with the total validation loss for the current batch.
        """

        if isinstance(batch, list):
            # split the data
            text_batch, protein_batch, text_mask_batch, protein_mask_batch, \
            pfam_text_batch, pfam_protein_batch, pfam_text_mask_batch, pfam_protein_mask_batch, \
            bool_pfam_vector = batch

        
        # forward pass over the swiss-prot data
        z_t_swiss, z_p_swiss = self(
                                  x_t=text_batch,
                                  x_p=protein_batch,
                                  compute_masked_logits=False
        )
        dist.barrier() # wait till all GPUs catch up...
     
        # gather all tensors
        z_t_swiss_all = self.all_gather(z_t_swiss, sync_grads=True)
        dist.barrier()    
        z_p_swiss_all = self.all_gather(z_p_swiss, sync_grads=True)

        # stack the embeddings
        z_t_swiss_all = z_t_swiss_all.view(-1, z_t_swiss.shape[-1])
        z_p_swiss_all = z_p_swiss_all.view(-1, z_p_swiss.shape[-1])

        # foward pass over the pfam data
        z_t_pfam, z_p_pfam = self(
                                x_t=pfam_text_batch,
                                x_p=pfam_protein_batch,
                                compute_masked_logits=False
        )
        dist.barrier() # wait till all GPUs catch up...
        
        # gather all tensors
        z_t_pfam_all = self.all_gather(z_t_pfam, sync_grads=True)
        dist.barrier()
        z_p_pfam_all = self.all_gather(z_p_pfam, sync_grads=True)
        
        # stack the embeddings
        z_t_pfam_all = z_t_pfam_all.view(-1, z_t_pfam.shape[-1])
        z_p_pfam_all = z_p_pfam_all.view(-1, z_p_pfam.shape[-1])
           
        # concatenate swiss-prot <> pfam embeddings
        z_t_all = torch.cat((z_t_swiss_all, z_t_pfam_all), dim=0)
        z_p_all = torch.cat((z_p_swiss_all, z_p_pfam_all), dim=0)

        # compute inter-modal loss values     
        loss_align, logits = self.model.compute_inter_loss(
                                            protein_embeddings=z_p_all,
                                            text_embeddings=z_t_all,
                                            batch_size=z_p_all.shape[0] // 2
        )

        # compute intra-modal loss values
        loss_intra, cosine_similarity = self.model.compute_intra_loss(
                                            protein_embeddings=z_p_all,
                                            batch_size=z_p_all.shape[0] // 2
        )

        # concatenate batch samples
        all_text_batch = torch.cat((text_batch, pfam_text_batch), dim=0)
        all_protein_batch = torch.cat((protein_batch, pfam_protein_batch), dim=0)
        all_text_mask_batch = torch.cat((text_mask_batch, pfam_text_mask_batch), dim=0)
        all_protein_mask_batch = torch.cat((protein_mask_batch, pfam_protein_mask_batch), dim=0)

        # compute mask language model logits
        logits_t_mask, logits_s_mask = self(
                    x_t=all_text_mask_batch,
                    x_p=all_protein_mask_batch,
                    compute_masked_logits=True
        )

        # compute mask language loss for biomedical expert model
        loss_text_mask = self.model.compute_masked_lang_loss(
                    logits_masked=logits_t_mask,
                    targets=all_text_batch,
                    targets_masked=all_text_mask_batch,
                    mask_token_id=self.text_tokenizer.mask_token_id
        )

        # compute mask language loss for protein expert model
        loss_sequence_mask = self.model.compute_masked_lang_loss(
                    logits_masked=logits_s_mask,
                    targets=all_protein_batch,
                    targets_masked=all_protein_mask_batch,
                    mask_token_id=self.sequence_tokenizer.mask_idx
        )


        # total loss
        #loss = loss_align + loss_intra + loss_text_mask + loss_sequence_mask
        
        if self.script_args.dataset_type == 'pfam':
            # Aggregate all computed losses.
            loss = loss_align + loss_intra + loss_text_mask + loss_sequence_mask
        
        elif self.script_args.dataset_type == 'pfam_ablated':
            # Aggregate all losses besides PFC.
            loss = loss_align + loss_text_mask + loss_sequence_mask
        else:
            # Add an assertion here
            assert self.script_args.dataset_type in ['pfam', 'pfam_ablated'], "Unexpected dataset_type value"
            sys.stderr.write("Unexpected dataset_type value\n")
            sys.exit(1)

     
        # track loss ...
        self.log('valid_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('valid_loss_align', loss_align, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('valid_loss_intra', loss_intra, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('valid_loss_text_mask', loss_text_mask, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log('valid_loss_seq_mask', loss_sequence_mask, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        # log CPU memory
        memory_usage = helper_tools.print_memory_usage()
        self.log(f'memory_usage', memory_usage, sync_dist=True)
 
        # track metrics
        metric_dict = self.performance_metrics(logits=logits.detach().cpu())
        for key in metric_dict:
            values = metric_dict[key]
            final_key = 'valid_' + key
            self.log(final_key, metric_dict[key], prog_bar=True if 'f1' in key else False, on_step=True, on_epoch=True, sync_dist=True)


        # collect joint embedding
        #self.val_text_joint_latents.append(z_t_all.detach().cpu())
        #self.val_seq_joint_latents.append(z_p_all.detach().cpu())

        return {'valid_loss': loss}


 #   def on_validation_epoch_end(self):
 #       print('Enter validation end of epoch analysis...')
#
 #       # collect and aggregate outputs from all validation steps
 #       val_z_t_joint = torch.cat(self.val_text_joint_latents, dim=0)
 #       val_z_s_joint = torch.cat(self.val_seq_joint_latents, dim=0)
 #       
 #       # compute singular values 
 #       print('Compute singular values...')
 #       text_log_sigma_k, S_text = self.compute_singular(val_z_t_joint.detach().cpu())
 #       protein_log_sigma_k, S_protein = self.compute_singular(val_z_s_joint.detach().cpu())
 #      
 #       # save image pngs for tracking dimensionality collapse
 #       self.save_png_to_tensorboard(
 #               data=text_log_sigma_k.numpy(),
 #               title='text',
 #       )
 #       self.save_png_to_tensorboard(
 #               data=protein_log_sigma_k.numpy(),
 #               title='protein'
 #       )
 #
 #       # free memory
 #       self.val_text_joint_latents.clear()
 #       self.val_seq_joint_latents.clear()
 #       
 #       
 #       # compute effective rank (RankME):
 #       print('Compute eranks')
 #       erank_text = self.compute_effective_rank(sigma_ks=S_text)
 #       erank_protein = self.compute_effective_rank(sigma_ks=S_protein)
 #       
 #       # log erank metrics
 #       self.log('valid_erank_text', erank_text, sync_dist=True)
 #       self.log('valid_erank_protein', erank_protein, sync_dist=True)

    def configure_optimizers(self,):

        params = [
                {"params": self.model.protein_encoder.parameters(), "lr": self.script_args.protein_encoder_lr},
                {"params": self.model.text_encoder.parameters(), "lr": self.script_args.text_encoder_lr},
                {"params": itertools.chain(
                    self.model.protein_projection.parameters(),
                    self.model.text_projection.parameters()
                    ),
                "lr": self.script_args.head_lr,
                "weight_decay": self.script_args.weight_decay}
        ]

        optimizer = torch.optim.AdamW(params, weight_decay=self.script_args.weight_decay)

        return {
                "optimizer": optimizer,
        }
    
    @torch.no_grad()
    def compute_class_metrics(
            self,
            outputs: torch.Tensor,
            targets: torch.Tensor,
            source: str
        ) -> dict:

        # convert torch tensors to numpy array
        outputs_np = outputs.numpy()
        targets_np = targets.numpy()

        # compute the metrics
        accuracy = accuracy_score(targets_np, outputs_np.round())
        precision = precision_score(targets_np, outputs_np.round(), average='micro')
        recall = recall_score(targets_np, outputs_np.round(), average='micro')
        f1 = f1_score(targets_np, outputs_np.round(), average='micro')

        return {
                f'{source}_accuracy': accuracy,
                f'{source}_precision': precision,
                f'{source}_recall': recall,
                f'{source}_f1': f1
        }

    @torch.no_grad()
    def performance_metrics(self, logits: torch.Tensor) -> tuple:

        logits = logits.cpu().float()

        # get probs
        p_text = F.softmax(logits, dim=-1) # prob of a given text captions aligning well with seq. pairs
        p_seq = F.softmax(logits.T, dim=-1) # prob of a given seq aligning well with text pairs
        p_tot = (p_seq + p_text) / 2 # total prob

        # get class labels
        y_pred_text = torch.argmax(p_text, dim=-1)
        y_pred_seq = torch.argmax(p_seq, dim=-1)
        y_pred = torch.argmax(p_tot, dim=-1)
        y_true = torch.arange(y_pred_text.shape[0])
        
        # compute class metrics
        text_metrics = self.compute_class_metrics(
                                    outputs=y_pred_text,
                                    targets=y_true,
                                    source='text'
        )
        seq_metrics = self.compute_class_metrics(
                                    outputs=y_pred_seq,
                                    targets=y_true,
                                    source='seq'
        )
        total_metrics = self.compute_class_metrics(
                                    outputs=y_pred,
                                    targets=y_true,
                                    source='total'
        )

        # combine dicts into one
        combined_dict = {}
        combined_dict.update(text_metrics)
        combined_dict.update(seq_metrics)
        combined_dict.update(total_metrics)

        return combined_dict
    
    @torch.no_grad()
    def compute_singular(self, inputs: torch.Tensor) -> (
            torch.Tensor,
            torch.Tensor
        ):

        # goal of this function: track for dimensionality collapse
        # inputs dim: (batch_size, emb_dim)

        mean_inputs = torch.mean(inputs, dim=0) # average over batch dimension
        norm_inputs = inputs - mean_inputs # normalize vectors
       
       # compute correlation matrix  #TODO: double check work...
        C = torch.zeros((norm_inputs.shape[-1], norm_inputs.shape[-1]))
        for sample_idx in range(norm_inputs.shape[0]):
            norm_vector = norm_inputs[sample_idx, :].unsqueeze(0)
            C += norm_vector.T @ norm_vector
        C *= 1/norm_vector.shape[0]

        _, S, _ = torch.linalg.svd(C, full_matrices=False)
        
        # return singular value indexes 
        log_sigma_k, _ = torch.sort(torch.log(S), descending=True)
        return (
                log_sigma_k,
                S
        )
        
    def compute_effective_rank(self, sigma_ks: torch.Tensor) -> torch.Tensor:
        """
        references:
            - Roy et al. The effective rank: a measure of effective dimensionality
            - Garrido et al. RankMe: Assessing the Downstream Performnace of Pretrained SS Reps by their Rank.
        """
        # sort the singular values
        sigma_ks, _ = torch.sort(sigma_ks, descending=True)
        
        # copute L1 norm for sing values.
        l1_norm_sigma = torch.norm(sigma_ks, p=1)
        
        # compute singular value distribution
        p_k = sigma_ks / l1_norm_sigma  + torch.finfo(torch.float).eps
        
        # compute Shannon entropy
        entropy = - torch.sum(p_k * torch.log(p_k))
    
        # get effective rank (RankME):
        erank = torch.exp(entropy)

        return erank

    def save_png_to_tensorboard(
            self,
            data: np.single,
            title: str,
            x_axis_label: str='Singular Value Rank Index',
            y_axis_label: str='Log of singular values',
            ):
    
        current_epoch = self.trainer.current_epoch
        
        # Plot the line
        fig, ax = plt.subplots(dpi=300)
        ax.plot(data)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_title(title)
        ax.set_ylim([-25,3])

        # Log the plot in TensorBoard 
        self.logger.experiment.add_figure(f'{title}_SingularValues_{current_epoch}', fig, current_epoch)
        
        # Close the figure to free up memory
        plt.close(fig)

    def predict_step(
            self,
            batch: torch.Tensor,
            batch_idx: torch.Tensor,
            dataloder_idx: bool=False
        ) -> (  
                torch.Tensor,
                torch.Tensor
        ):


        if isinstance(batch, list):
            # mean loss
            text_batch, protein_batch = batch
            outputs = self(
                    x_t=text_batch,
                    x_p=protein_batch,
                    compute_masked_logits=False
            )
        
        z_t_joint, z_p_joint = outputs

        self.predict_text_joint_latents.append(z_t_joint.detach().cpu())
        self.predict_seq_joint_latents.append(z_p_joint.detach().cpu())

        return outputs

    def on_predict_epoch_end(self, outputs=None):

        self.predict_text_joint_latents = torch.cat(self.predict_text_joint_latents).cpu()
        self.predict_seq_joint_latents = torch.cat(self.predict_seq_joint_latents).cpu()


##########################
# Facilitator PL wrapper #
##########################

class PL_Facilitator(pl.LightningModule):

    def __init__(
            self, 
            args: any
    ):

        super().__init__()

        # arguments
        self.args = args

        # model 
        self.model = mod.Facilitator(
                    in_dim=self.args.emb_dim,
                    hid_dim=self.args.hid_dim,
                    out_dim=self.args.emb_dim,
                    dropout=self.args.dropout
        )
        
        self.text_to_protein_joint_embeddings = []

    def forward(
            self,
            z_t: torch.Tensor,
    ) -> torch.Tensor:

        # reconfigure z_t to z_p (additional alignment)
        z_t_to_p = self.model(z_t)

        return z_t_to_p

    

    def training_step(self, batch: torch.Tensor, batch_id: any) -> dict:

        # check if the batch is a list and split data if so 
        if isinstance(batch, list):
            text_embeddings, protein_embeddings = batch

        # forward pass with the model
        z_t_to_p = self(z_t=text_embeddings)

        # compute loss
        loss = self.model.compute_loss(
                output=z_t_to_p,
                target=protein_embeddings,
                loss_option=self.args.loss_type
        )
    
        # log the total loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return {'loss': loss}

 
    def validation_step(self, batch: torch.Tensor, batch_id: any) -> dict:

        # check if the batch is a list and split data if so 
        if isinstance(batch, list):
            text_embeddings, protein_embeddings = batch

        # forward pass with the model
        z_t_to_p = self(z_t=text_embeddings)
    
        # compute loss
        loss = self.model.compute_loss(
                output=z_t_to_p,
                target=protein_embeddings,
                loss_option=self.args.loss_type
        )
    
        # log the total loss
        self.log('valid_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return {'loss': loss}

    
    def configure_optimizers(self,):

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
    
        return {
                "optimizer": optimizer
        }
   

    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = None) -> torch.Tensor:
        """
        Defines a single prediction (inference) step.
        """

        # Unpack the batch if it comes in a list format.
        # Here, we only take text embeddings for prediction as an example.
        if isinstance(batch, list):
            text_embeddings, _ = batch  # We ignore the second element (protein_embeddings)
        else:
            text_embeddings = batch

        # Perform forward pass to get transformed text embeddings (z_t_to_p)
        z_t_to_p = self(z_t=text_embeddings)
        self.text_to_protein_joint_embeddings.append(z_t_to_p.detach().cpu())

        return z_t_to_p

    def on_predict_epoch_end(self, outputs=None):
        
        self.text_to_protein_joint_embeddings = torch.cat(self.text_to_protein_joint_embeddings).cpu()
