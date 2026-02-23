"""
Helper script to compile Stage 2 outputs into a format ready for finetuning.

This script uses the output of the biom3_run_Facilitator_sample entrypoint, 
which saves as part of its output a .pt file <prefix>.Facilitator_emb.pt.
This file should contain the following key-accessed data:
    acc_id: Accession IDs
    sequence: Protein sequences
    facilitator_embedding: Facilitator text embeddings z_c
This data is loaded and converted into hdf5 format, under a specified group 
name, e.g. `MMD_data` or `MSE_data`.

The resulting hdf5 data should be immediately ready for the finetuning pipeline.
"""

import sys
import argparse
import pandas as pd
import h5py
import torch
import numpy as np

# Maps internal script keys to those used to access data in the INPUT file.
INPUT_KEYS = {
    "ACCESSION_ID": "acc_id",
    "SEQUENCE": "sequence",
    "FACILITATOR_EMBEDDING": "z_c",
}

# Maps internal script keys to those used to access data in the OUTPUT file.
OUTPUT_KEYS = {
    "ACCESSION_ID": "acc_id",
    "SEQUENCE": "sequence",
    "SEQUENCE_LENGTH": "sequence_length",
    "FACILITATOR_EMBEDDING": "text_to_protein_embedding",
}


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outfpath", type=str, required=True, 
                        help="output filepath")
    parser.add_argument("--dataset_key", type=str, required=True, 
                        help="key under which data will be saved in output dictionary. (e.g. MMD_data)")
    parser.add_argument("-fe", "--facilitator_embeddings", type=str, required=True,
                        help="stage 2 facilitator-embedded data (e.g. test_Facilitator_embeddings.pt)")
    return parser.parse_args(args)


def main(args):
    outfpath = args.outfpath
    dataset_key = args.dataset_key
    facilitator_embeddings_fpath = args.facilitator_embeddings
    
    input_data = torch.load(facilitator_embeddings_fpath)
    
    # TODO: key checks
    print(input_data)
    for k in input_data.keys():
        print(input_data[k][0])
    sequences = input_data[INPUT_KEYS["SEQUENCE"]]
    acc_ids = input_data[INPUT_KEYS["ACCESSION_ID"]]
    text_to_protein_embedding = input_data[INPUT_KEYS["FACILITATOR_EMBEDDING"]]
    sequence_length = np.array([len(seq) for seq in sequences], dtype=int)
    
    data_out = {
        "ACCESSION_ID": acc_ids,
        "SEQUENCE": sequences,
        "SEQUENCE_LENGTH": sequence_length,
        "FACILITATOR_EMBEDDING": text_to_protein_embedding
    }

    with h5py.File(outfpath, 'w') as outfile:
        grp = outfile.create_group(dataset_key)
        for k, output_key in OUTPUT_KEYS.items():
            outfile.create_dataset(
                f"{dataset_key}/{output_key}", 
                data=data_out[k]
            )


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
