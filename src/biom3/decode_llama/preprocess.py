"""dataset and collate_fn for the z_p decoder 
"""

import os
import numpy as np 
import pandas as pd 
import torch 
from torch.utils.data import Dataset 
from biom3.backend.device import setup_logger
from biom3.decode_llama.datasplit import load_split, decode_acc_id

logger = setup_logger(__name__)


class ZpEmbeddingDataset(Dataset):
    #pairs z_p vectors with captions, one split at a time 

    def __init__(self, embedding_path: str, split_path: str, split: str,):
        emb = torch.load(embedding_path, map_location = "cpu", weights_only = False)
        self.z_p = emb["z_p"]
        accessions = decode_acc_id(emb["acc_id"])
        splits = load_split(split_path)
        
        self.indices = []
        self.captions = []
        self.acc_ids = []
        s = set(splits.loc[splits["split"] == split, "acc_id"])
        captions_full = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in emb["text_prompts"]]

        for n, a in enumerate(accessions):
            if a in s:
                self.indices.append(n)
                self.captions.append(captions_full[n])
                self.acc_ids.append(a)
        

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx) -> tuple[torch.Tensor, str, str]:
        return (
            self.z_p[self.indices[idx]], self.captions[idx], self.acc_ids[idx]
        )

def collate_fn(batch, tokenizer, max_length: int = 1024): 
    #tokenizes captions and pads to max-in-batch 
    z_ps, captions, acc_ids = zip(*batch)
    z_p_batch = torch.stack(list(z_ps), dim=0)
    tokens = tokenizer(list(captions), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    labels = tokens["input_ids"].clone()
    labels[tokens["attention_mask"] == 0] = -100

    return {"z_p": z_p_batch, "input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"], "labels": labels, "acc_ids": list(acc_ids)}




