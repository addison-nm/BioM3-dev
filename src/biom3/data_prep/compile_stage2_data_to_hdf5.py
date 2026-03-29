"""
Compile Stage 2 Facilitator outputs into HDF5 format for Stage 3 finetuning.

Takes the .pt output of biom3_Facilitator_sample (containing acc_id, sequence,
and z_c keys) and writes an HDF5 file with a named dataset group ready for
the ProteoScribe finetuning pipeline.
"""

import argparse
import h5py
import numpy as np
import torch


# Maps internal keys to those used in the Facilitator output (.pt file).
INPUT_KEYS = {
    "ACCESSION_ID": "acc_id",
    "SEQUENCE": "sequence",
    "FACILITATOR_EMBEDDING": "z_c",
}

# Maps internal keys to those written into the HDF5 output.
OUTPUT_KEYS = {
    "ACCESSION_ID": "acc_id",
    "SEQUENCE": "sequence",
    "SEQUENCE_LENGTH": "sequence_length",
    "FACILITATOR_EMBEDDING": "text_to_protein_embedding",
}


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="Compile Stage 2 Facilitator data to HDF5"
    )
    parser.add_argument(
        "-i", "--input_data_path", type=str, required=True,
        help="Path to Stage 2 Facilitator output (.pt file)"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True,
        help="Path to write the compiled HDF5 file"
    )
    parser.add_argument(
        "--dataset_key", type=str, default="MMD_data",
        help="Group name under which data is stored in the HDF5 file (default: MMD_data)"
    )
    return parser.parse_args(args)


def main(args):
    input_data = torch.load(args.input_data_path, weights_only=False)

    sequences = input_data[INPUT_KEYS["SEQUENCE"]]
    acc_ids = input_data[INPUT_KEYS["ACCESSION_ID"]]
    text_to_protein_embedding = input_data[INPUT_KEYS["FACILITATOR_EMBEDDING"]].cpu()
    sequence_length = np.array([len(seq) for seq in sequences], dtype=int)

    data_out = {
        "ACCESSION_ID": acc_ids,
        "SEQUENCE": sequences,
        "SEQUENCE_LENGTH": sequence_length,
        "FACILITATOR_EMBEDDING": text_to_protein_embedding,
    }

    with h5py.File(args.output_path, "w") as outfile:
        outfile.create_group(args.dataset_key)
        for k, output_key in OUTPUT_KEYS.items():
            outfile.create_dataset(
                f"{args.dataset_key}/{output_key}",
                data=data_out[k],
            )

    print(f"Compiled HDF5 saved to {args.output_path} "
          f"(group: {args.dataset_key}, {len(sequences)} samples)")
