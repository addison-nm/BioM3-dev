"""
Splits embeddings into train, val, and test for the decoder 

"""

import os
from typing import Iterable
import numpy as np
import pandas as pd
import torch 
from biom3.backend.device import setup_logger

logger = setup_logger(__name__)
SPLIT_LABELS = ["train", "val", "test"]

def decode_acc_id(acc_id) -> list[str]:
    #decodes the accession id from embeddins into string list
    out = []
    for x in acc_id:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    
    return out 

def load_acc_id(embeddings_path: str) -> list[str]:
    #loads accession id from embeddings 
    emb = torch.load(embeddings_path, map_location = "cpu", weights_only = False)
    
    return decode_acc_id(emb["acc_id"])

def build_split_corpus(
    acc_ids: Iterable[str],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0
) -> pd.DataFrame: 
    #splits accession ids into train, val, and test set using seeded random shuffle 
    acc_ids = list(acc_ids)
    if len(set(acc_ids)) != len(acc_ids):
        raise ValueError(
            "split assumes unique acc_ids"
        )
    
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(acc_ids))
    n_train = int(round(fractions[0] * len(acc_ids)))
    n_val =  int(round(fractions[1] * len(acc_ids)))
    n_test =  int(round(fractions[2] * len(acc_ids)))

    splits = np.empty(len(acc_ids), dtype = object)
    splits[order[:n_train]] = "train"
    splits[order[n_train:n_train + n_val]] = "val"
    splits[order[n_train + n_val:]] = "test"

    df = pd.DataFrame({"acc_id": acc_ids, "split": splits})

    return df

def load_split(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"acc_id": str, "split": str}) 

    invalid = set(df["split"]) - set(SPLIT_LABELS)
    if invalid:
        raise ValueError(
            f"unexpected split label: {invalid}"
        )

    return df

def save_split(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
    counts = df["split"].value_counts().to_dict()

    logger.info("wrote split manifest %s (%s)", path, counts)

def parse_args(args):
    import argparse
    parser = argparse.ArgumentParser(
        description="build the decoder train/val/test split "
                    "from a pencl_embeddings.pt file."
    )
    parser.add_argument("--embeddings_path", "-e", type=str, required=True,
                        help="path to pencl_embeddings.pt")
    parser.add_argument("--output_path", "-o", type=str, required=True,
                        help="path to write the split csv")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed (pin once and never change)")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    return parser.parse_args(args)

def main(args):
    acc_ids = load_acc_id(args.embeddings_path)
    df = build_split_corpus(
        acc_ids,
        fractions=(args.train_frac, args.val_frac, args.test_frac),
        seed=args.seed,
    )

    save_split(df, args.output_path)

if __name__ == "__main__":
    import sys
    main(parse_args(sys.argv[1:]))
