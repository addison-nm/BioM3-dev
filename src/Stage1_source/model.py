import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertForMaskedLM
import torch.distributed as dist
import esm
from torch.nn.utils.weight_norm import weight_norm


"""
functions and classes adapted from the following:
    1. https://keras.io/examples/vision/nl_image_search/
    2. https://colab.research.google.com/drive/1hYHb0FTdKQCXZs3qCwVZnSuVGrZU2Z1w?usp=sharing
"""

class ProteinEncoder(nn.Module):
    """
    Encoder for protein sequence to a fixed size vector --> z_s
    """
    
    def __init__(self, args: any):
        super().__init__()

        #self.script_args = args
        self.seq_model_path = args.seq_model_path
        self.pretrained = args.pretrained_seq
        self.trainable = args.trainable_seq
        self.n_layers_to_finetune = args.pLM_n_layers_to_finetune
        self.rep_layer = args.rep_layer
        self.model, self.alphabet = self.get_ESM_model() # get model and alphabet (ESM)
        
        for p in self.model.parameters():
            if self.trainable and self.n_layers_to_finetune == 0:
                p.required_grad = True
            else:
                p.requires_grad = False
        
        # Make the last n_layers_to_finetune layers trainable
        if self.trainable and self.n_layers_to_finetune != 0:
            for layer in self.model.layers[-self.n_layers_to_finetune:]:
                for p in layer.parameters():
                    p.requires_grad = True

        # Use the [CLS] token hidden representation as the sentence's embedding
        # for the downstream latent alignment.
        self.target_token_idx = 0

    def get_ESM_model(self):

        return esm.pretrained.load_model_and_alphabet(
                os.path.expanduser(
                    self.seq_model_path
                )
        )
    
    def forward(self, x_s: torch.Tensor, compute_logits: bool=False):
        # drop channel depth
        x_s = x_s.squeeze(1)

        outputs = self.model(
                x_s,
                repr_layers=[self.rep_layer],
                return_contacts=False
        )
        
        # mask langauge model objective 
        if compute_logits:
            logits = outputs['logits']
            return logits
       
        # fine-tuning cls token for protein sequence alignment with biomedical text 
        cls_hidden = outputs['representations'][self.rep_layer][:,self.target_token_idx,:]
        return cls_hidden
    
class TextEncoder(nn.Module):

    """
    Encoder for protein's natural text to a fixed size vector --> z_t
    """

    def __init__(self, args: any):
        super().__init__()
 
        self.model_name = args.text_model_path
        self.pretrained = args.pretrained_text
        self.trainable = args.trainable_text
        self.n_layers_to_finetune = args.bLM_n_layers_to_finetune 
        self.tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)

        if self.pretrained:
            #self.model = AutoModel.from_pretrained(self.model_name)
            self.model = BertForMaskedLM.from_pretrained(self.model_name)

        else:
            #self.model = AutoModel.from_config(self.model_name)
            self.model = BertForMaskedLM.from_config(self.model_name)

        for p in self.model.parameters():
            if self.trainable and self.n_layers_to_finetune == 0:
                p.required_grad = True
            else:
                p.requires_grad = False
       
        # Make the last n_layers_to_finetune layers trainable
        if self.trainable and self.n_layers_to_finetune != 0:
            for layer in self.model.bert.encoder.layer[-self.n_layers_to_finetune:]:
                for p in layer.parameters():
                    p.requires_grad = True
        
        # Use the [CLS] token hidden representation as the sentence's embedding
        # for the downstream latent alignment.
        self.target_token_idx = 0

    def forward(self, inputs: torch.Tensor, compute_logits: bool=False) -> torch.Tensor:
        # drop channel depth
        inputs = inputs.squeeze(1)
        
        if compute_logits:
            # compute the masked language model logits
            #sequence_output = outputs.last_hidden_state
            outputs = self.model(inputs)
            logits = outputs.logits
            return logits
        
        else:
            outputs = self.model(inputs, output_hidden_states=True)
            # use the token representations...
            last_hidden_state = outputs.hidden_states[-1]
            return last_hidden_state[:, self.target_token_idx, :] # return [cls] token 



class ProjectionHead(nn.Module):
    """
    g(.) which maps z_t --> h_t or z_s --> h_s
    
    Note: h is the joint embedding representation, h_t
    is the joint embedding for the text caption, and
    h_s is the joint embedding for the protein sequence.
    """

    def __init__(self, embedding_dim: int, args: any):

        super().__init__()
        self.projection_dim = args.proj_embedding_dim
        self.dropout = args.dropout
        self.embedding_dim = embedding_dim

        # model graph
        self.projection = nn.Linear(self.embedding_dim, self.projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(self.projection_dim, self.projection_dim)
        self.dropout = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.projection_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        projection = self.projection(z)
        h = self.gelu(projection)
        h = self.fc(h)
        h = self.dropout(h)
        h = h + projection
        h = self.layer_norm(h)
        return h




#####################
# Pfam architecture #
#####################



class pfam_PEN_CL(nn.Module):

    """
    Protein Embeddings with Natural lanauge using Constrastive Learing (PEN-CL) while including pfam constrastive learning.
    """

    def __init__(self, args: any):

        super().__init__()

        self.protein_embedding = args.protein_encoder_embedding
        self.text_embedding = args.text_encoder_embedding
        self.temperature = args.temperature

        # protein sequence expert
        self.protein_encoder = ProteinEncoder(args=args)
        # natural text expert
        self.text_encoder = TextEncoder(args=args)
        
        # projection heads g_seq( . ) --> joint embedding space
        self.protein_projection = ProjectionHead(
                embedding_dim=self.protein_embedding,
                args=args
        )

        # projection heads g_text( . ) --> joint embedding space
        self.text_projection = ProjectionHead(
                embedding_dim=self.text_embedding,
                args=args
        )

    def forward(
            self,
            x_t: torch.Tensor,
            x_s: torch.Tensor,
            compute_masked_logits: bool=False
        ) -> dict:

        if compute_masked_logits:
            # forward pass for computing logits for masked langauge objective
            protein_logits = self.protein_encoder(x_s, compute_logits=True)
            text_logits = self.text_encoder(x_t, compute_logits=True)

            return {
                    'text_masked_logits': text_logits,
                    'protein_masked_logits': protein_logits
            }

        else:
            # split the tuple into 2 dicts... 
            # getting protein sequence and text inputs ...
            z_t = self.text_encoder(x_t, compute_logits=False)
            z_s = self.protein_encoder(x_s, compute_logits=False)

            # "joint" sequence and text embedding (with same dimension)
            z_t_joint = self.text_projection(z_t)
            z_s_joint = self.protein_projection(z_s)

            return {
                    'text_joint_latent': z_t_joint,
                    'seq_joint_latent': z_s_joint,
            }

    def compute_inter_loss(
        self,
        protein_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        batch_size: int
    ) -> (
            torch.Tensor,
            torch.Tensor
            ):
            
            """
            Compute the inter-modal contrastive InfoNCE loss between protein and text embeddings.

            Parameters:
            - protein_embeddings: A tensor representing the embeddings of the protein sequences.
            - text_embeddings: A tensor representing the embeddings of the text descriptions.
            - batch_size: The number of samples in the batch.

            Steps:
            1. Generate a masking matrix to identify off-diagonal elements.
            2. Compute cosine similarities (i.e., logits) between text and protein embeddings.
            3. Compute self-similarities for both protein and text embeddings.
            4. Mask off-diagonal elements between swiss-prot and pfam in the similarity matrices.
            5. Define ground truth by averaging the masked protein and text similarity matrices.
            6. Compute the contrastive loss for the protein and text embeddings using the ground truth.

            Returns:
            - Mean contrastive loss for the given batch of protein and text embeddings.
            - The logits (cosine similarity matrix between text and protein embeddings).

            Note: This function assumes a specific structure in the input batches, where corresponding positive samples 
            in the protein and text embeddings are arranged in a particular way, allowing for masking and contrastive loss calculation.
            """
            
            # get off-diagonal masking matrix
            mask = torch.zeros((2*batch_size, 2*batch_size))
            # mask the bottom left quadrant diagonal
            mask[batch_size:, :batch_size] = torch.eye(batch_size)
            # mask the top right quadrant
            mask[:batch_size, batch_size:] = torch.eye(batch_size)
            # convert to correct device and convert to boolean
            mask = mask.to(protein_embeddings.device).bool()

            # matrix multiplication between model embeddings
            logits = (text_embeddings @ protein_embeddings.T) / self.temperature
            protein_similarity = protein_embeddings @ protein_embeddings.T
            text_similarity = text_embeddings @ text_embeddings.T

            # mask the off-diagonal between swiss-prot and pfam
            mask_protein_similarity = self.set_inf(protein_similarity, mask)
            mask_text_similarity = self.set_inf(text_similarity, mask)
            mask_logits = self.set_inf(logits, mask)

            # ground truth
            targets = F.softmax(
                (mask_protein_similarity + mask_text_similarity) / (2 * self.temperature), dim=-1
            )

            # compute loss
            text_loss = self.cross_entropy(mask_logits, targets, reduction='none')
            protein_loss = self.cross_entropy(mask_logits.T, targets.T, reduction='none')
            loss = (protein_loss + text_loss) / 2.0

            return (
                loss.mean(),
                mask_logits.detach().cpu()
            )


    def compute_intra_loss(  
            self,
            protein_embeddings,
            batch_size
        ) -> (
                torch.Tensor,
                torch.Tensor,
        ):
        """
        Compute the intra-modal contrastive InfoNCE loss for protein embeddings.

        Parameters:
        - protein_embeddings: A tensor representing the embeddings of the protein sequences.
        - batch_size: Batch size used for training.

        Steps:
        1. Normalize the protein embeddings using L2 normalization.
        2. Compute the cosine similarity between the normalized embeddings.
        3. Mask the diagonal of the cosine similarity matrix to avoid using a protein's similarity with itself.
        4. Define positive examples by rolling the mask. The positive example for a given protein embedding is determined by an embedding half the batch size away.
        5. Compute the InfoNCE loss using the masked cosine similarity matrix.

        Returns:
        - Mean InfoNCE loss for the given batch of protein embeddings.
        - The cosine similarity matrix.

        Note: The underlying assumption is that in each batch, corresponding positive samples for a given protein embedding 
        lie half the batch size away. The function computes the negative log likelihood loss between these positive samples 
        and the entire batch.
        """
        
        # l2 normalization
        #norm_protein_embeddings = F.normalize(protein_embeddings, p=2, dim=1)
        norm_protein_embeddings = protein_embeddings

        # cosine similarity
        cosine_similarity = (norm_protein_embeddings @ norm_protein_embeddings.T) / self.temperature
        
        # mask cosine similarity matrix
        sample_size = protein_embeddings.shape[0]
        mask = torch.eye(sample_size, device=cosine_similarity.device, dtype=torch.bool)
        #cosine_similarity.masked_fill_(mask, float(-9e15))
        cosine_similarity = self.set_inf(cosine_similarity, mask)
        
        # Find positive example -> batch_size //2 away from the original example (swiss-prot<>pfam)
        pos_mask = mask.roll(shifts=mask.shape[0]//2, dims=0)

        # InfoNCE loss
        nll = -cosine_similarity[pos_mask] + torch.logsumexp(cosine_similarity, dim=-1)
        
        return (
            nll.mean(),
            cosine_similarity.cpu(),
        )

    def set_inf(
            self,
            tensor: torch.Tensor,
            mask: torch.Tensor
        ) -> torch.Tensor:
        # Determine replacement value based on tensor dtype
        if tensor.dtype == torch.float32:
            replace_value = -9e15
        elif tensor.dtype == torch.float16:
            replace_value = -1e4
        else:
            raise ValueError("Unsupported tensor dtype for this operation.")

        # Use masked_fill_ to replace positions in tensor where mask is True with the specified value
        tensor.masked_fill_(mask, replace_value)

        return tensor

    def cross_entropy(
            self,
            preds: torch.Tensor,
            targets: torch.Tensor,
            reduction: str='none'
        ) -> torch.Tensor:

        # compute categorical cross entropy
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)

        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        else:
            assert False, print('Choose either "none" or "mean" for reduction argument')

    def compute_masked_lang_loss(
            self,
            logits_masked: torch.Tensor,
            targets: torch.Tensor,
            targets_masked: torch.Tensor,
            mask_token_id: torch.Tensor
        ) -> torch.Tensor:
        
        """
        Compute the masked language model loss for BERT-like architectures.

        Given a batch of logits predicted for masked positions and their corresponding target tokens, this function 
        computes the cross-entropy loss between the predicted logits and the true labels, but only for positions 
        that have been masked in the input.

        Parameters:
        - logits_masked: Predicted token logits for masked positions from the model.
                         Shape: (batch_size, seq_len, vocab_size).
        - targets: True token IDs for each position in the input sequence.
                   Shape: (batch_size, seq_len).
        - targets_masked: Token IDs for the input sequence, including masked positions.
                          Shape: (batch_size, seq_len).
        - mask_token_id: The ID corresponding to the [MASK] token in the vocabulary.

        Steps:
        1. Compute the cross-entropy loss between predicted logits and true labels across all positions.
        2. For each sample in the batch, locate the positions that were masked.
        3. Extract the loss values corresponding to these masked positions.
        4. Compute and return the mean of these extracted loss values across the batch.

        Returns:
        - Mean cross-entropy loss for masked positions across the batch.

        Note: This function focuses exclusively on masked positions in the input, as is typical for the MLM objective 
        in BERT-like models. It disregards unmasked positions.
        """

        # compute the masked langauge objective loss for masked logits
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_mask = loss_func(
                            logits_masked.permute(0, 2, 1), # (batch_size, vocab_size, seq_len)
                            targets.squeeze(1) # (batch_size, seq_len)
        )

        # list to append loss values
        batch_loss = []

        for ii, target_mask_sample in enumerate(targets_masked):
            
            # locate mask positions 
            masked_positions = (target_mask_sample == mask_token_id).tolist()
            # extract the loss values at those masked positions
            loss_mask_sample = loss_mask[ii][masked_positions]
            
            # append mean loss value for a given batch sample
            if loss_mask_sample.numel() > 0:
                batch_loss.append(torch.mean(loss_mask_sample).unsqueeze(0))
        
        if len(loss_mask_sample) > 0:
            loss_mask_mean = torch.mean(torch.cat(batch_loss))
        else:
            # handle the case where there are no masked positions in any sample 
            loss_mask_mean = torch.tensor(0.0, device=logits_masked.device)

        return loss_mask_mean


###############
# Facilitator #
###############


class Facilitator(nn.Module):

    def __init__(self,
                 in_dim: int,  # Input dimension
                 hid_dim: int,  # Hidden layer dimension
                 out_dim: int,  # Output dimension
                 dropout: float = 0.  # Dropout rate
                 ):
        super().__init__()

        # Main neural network structure
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),  # Weight-normalized linear layer
            nn.GELU(),  # GELU activation function
            nn.Dropout(dropout, inplace=True),  # Dropout layer
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)  # Weight-normalized output layer
        )

    def forward(self, x):
        # Forward pass through the network
        return self.main(x)

    def compute_loss(self, output: torch.Tensor, target: torch.Tensor, loss_option='MSE') -> torch.Tensor:
        # Compute loss based on the chosen loss_option ('MSE' or 'MMD')
        if loss_option == 'MSE':
            return Facilitator.compute_MSE(output, target)
        elif loss_option == 'MMD':
            return Facilitator.compute_mmd(output, target)
        else:
            return ValueError("Invalid loss option")
    
    @staticmethod
    def compute_MSE(output, target):
        # Compute Mean Squared Error between output and target
        mse_loss = nn.MSELoss()
        loss = mse_loss(output, target)
        return loss

    @staticmethod
    def compute_kernel(
            x: torch.FloatTensor,
            y: torch.FloatTensor
        ) -> torch.FloatTensor:
        """
        Compute the Gaussian RBF kernel between tensors x and y
        """

        # Get the sizes of each mini-batch
        x_size, y_size = x.shape[0], y.shape[0]

        # Dimension based on z size
        dim = x.shape[1]

        x = x.view(x_size, 1, dim)
        y = y.view(1, y_size, dim)

        x_core = x.expand(x_size, y_size, dim)
        y_core = y.expand(x_size, y_size, dim)

        # Gaussian RBF kernel computation
        return torch.exp(-(x_core - y_core).pow(2).mean(2) / dim)

    @staticmethod
    def compute_mmd(
            x: torch.FloatTensor,
            y: torch.FloatTensor
        ) -> torch.FloatTensor:
        """
        Compute the Maximum Mean Discrepancy (MMD) between two distributions.
        Args:
            x: Samples from first distribution (z_t_to_p ~ q(z_p))
            y: Samples from second distribution (z_p ~ p(z_p))
        Returns:
            MMD_loss: The MMD loss between the sampled distributions
        """

        x_kernel = Facilitator.compute_kernel(x, x)
        y_kernel = Facilitator.compute_kernel(y, y)
        xy_kernel = Facilitator.compute_kernel(x, y)

        # Calculate MMD loss
        return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()


