""" 
text decoder with MLP flowing into a frozen BioGPT 
"""
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BioGptTokenizer, BioGptForCausalLM
from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

class PrefixHead(nn.Module):
    #MLP that maps z_p into a k prefix-token embeddings

    def __init__(self, 
        z_dim: int = 512, 
        prefix_length: int = 10,
        d_model: int = 1024,
        hidden_dim: int = 2048,
        dropout: float = 0.1, 
    ):
        super().__init__()

        self.projection_dim = d_model
        self.embedding_dim = z_dim

        self.projection = nn.Linear(self.embedding_dim, self.projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(self.projection_dim, prefix_length * self.projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.projection_dim)
        self.prefix_length = prefix_length

    def forward(self, z_p: torch.Tensor) -> torch.Tensor:
        #z_p (dim = 512) -> (prefix_length, d_model)
        projection = self.projection(z_p)              
        h = self.gelu(projection)                      
        h = self.fc(h)                                 
        h = self.dropout(h)
        h = h.view(z_p.size(0), self.prefix_length, self.projection_dim)  
        h = self.layer_norm(h)       

        return h
    

class ZpDecoder(nn.Module):
    #frozen BioGPT model conditioned on z_p prefixes 

    def __init__(self, args): 
        super().__init__()

        self.model = BioGptForCausalLM.from_pretrained(args.lm_path)
        self.hidden_dim = self.model.config.hidden_size
        self.head = PrefixHead(z_dim = args.z_dim, prefix_length = args.prefix_length, d_model = self.hidden_dim, hidden_dim = args.hidden_dim, dropout = args.dropout)
        self.prefix_length = args.prefix_length

        for p in self.model.parameters():
            p.requires_grad = False
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        
        return self

    def forward(self, z_p, input_ids, attention_mask, labels): 
        B = z_p.size(0)
        k = self.prefix_length

        prefix = self.head(z_p)
        token_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix, token_embeds], dim = 1)
        attn = torch.cat([torch.ones(B, k, dtype=attention_mask.dtype, device=attention_mask.device), attention_mask], dim = 1)
        labels = torch.cat([torch.full((B, k), -100, dtype=labels.dtype, device=labels.device), labels], dim = 1)
        out = self.model(inputs_embeds = inputs_embeds, attention_mask = attn, labels = labels)

        return out 

