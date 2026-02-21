"""BioM3 Stage 2: Facilitator sampling

Mimics the workflow described at
    https://huggingface.co/niksapraljak1/BioM3#stage-2-facilitator-sampling

Preparation:

    Follows execution of Stage 1: run_PenCL_inference

    Corresponding config file:
        configs/stage2_config_Facilitator_sample.json

Example usage:

biom3_run_Facilitator_sample.py \
    --input_data_path "outputs/test_PenCL_embeddings.pt" \
    --json_path "configs/stage2_config_Facilitator_sample.json" \
    --model_path "./weights/Facilitator/BioM3_Facilitator_epoch20.bin" \
    --output_data_path "outputs/test_Facilitator_embeddings.pt"

"""

import argparse
import yaml
from argparse import Namespace
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import biom3.Stage1.model as mod
import warnings


# Step 0: Argument Parser Function
def parse_arguments(args):
    parser = argparse.ArgumentParser(description="BioM3 Facilitator Model (Stage 2)")
    parser.add_argument('-i', '--input_data_path', type=str, required=True,
                        help="Path to the input embeddings (e.g., PenCL_test_outputs.pt)")
    parser.add_argument('-c', '--json_path', type=str, required=True,
                        help="Path to the JSON configuration file (stage2_config_Facilitator_sample.json)")
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help="Path to the Facilitator model weights (e.g., BioM3_Facilitator_epoch20.bin)")
    parser.add_argument('-o', '--output_data_path', type=str, required=True,
                        help="Path to save the output embeddings (e.g., Facilitator_test_outputs.pt)")
    return parser.parse_args(args)


# Step 1: Load JSON Configuration
def load_json_config(json_path):
    """Load JSON configuration file."""
    with open(json_path, "r") as f:
        config = json.load(f)
    warnings.warn("jajaja")
    return config


# Step 2: Convert JSON dictionary to Namespace
def convert_to_namespace(config_dict):
    """Recursively convert a dictionary to an argparse Namespace."""
    for key, value in config_dict.items():
        if isinstance(value, dict):
            config_dict[key] = convert_to_namespace(value)
    return Namespace(**config_dict)


# Step 3: Load Pre-trained Model
def prepare_model(config_args, model_path) -> nn.Module:
    model = mod.Facilitator(
        in_dim=config_args.emb_dim,
        hid_dim=config_args.hid_dim,
        out_dim=config_args.emb_dim,
        dropout=config_args.dropout
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("Model loaded successfully with weights!")
    return model


# Step 4: Compute MMD Loss
def compute_mmd_loss(x, y, kernel="rbf", sigma=1.0):
    def rbf_kernel(a, b, sigma):
        pairwise_distances = torch.cdist(a, b, p=2) ** 2
        return torch.exp(-pairwise_distances / (2 * sigma ** 2))

    K_xx = rbf_kernel(x, x, sigma)
    K_yy = rbf_kernel(y, y, sigma)
    K_xy = rbf_kernel(x, y, sigma)

    mmd_loss = K_xx.mean() - 2 * K_xy.mean() + K_yy.mean()
    return mmd_loss


def main(args):

    # Load configuration
    config_dict = load_json_config(args.json_path)
    config_args = convert_to_namespace(config_dict)

    # Load model
    model = prepare_model(
        config_args=config_args, 
        model_path=args.model_path
    )

    # Load input embeddings
    embedding_dataset = torch.load(args.input_data_path)

    # Run inference to get facilitated embeddings
    with torch.no_grad():
        z_t = embedding_dataset['z_t']
        z_p = embedding_dataset['z_p']
        z_c = model(z_t)
        embedding_dataset['z_c'] = z_c

    # Compute evaluation metrics
    # 1. MSE between embeddings
    mse_zc_zp = F.mse_loss(z_c, z_p)
    mse_zt_zp = F.mse_loss(z_t, z_p)

    # 2. Compute L2 norms for first batch
    batch_idx = 0
    norm_z_t = torch.norm(z_t[batch_idx], p=2).item()
    norm_z_p = torch.norm(z_p[batch_idx], p=2).item()
    norm_z_c = torch.norm(z_c[batch_idx], p=2).item()

    # 3. Compute MMD between embeddings
    mmd_zc_zp = model.compute_mmd(z_c, z_p)
    mmd_zp_zt = model.compute_mmd(z_p, z_t)

    # Print results
    print("\n=== Facilitator Model Output ===")
    print(f"Shape of z_t (Text Embeddings): {z_t.shape}")
    print(f"Shape of z_p (Protein Embeddings): {z_p.shape}")
    print(f"Shape of z_c (Facilitated Embeddings): {z_c.shape}\n")

    print("=== Norm (L2 Magnitude) Results for Batch Index 0 ===")
    print(f"Norm of z_t (Text Embedding): {norm_z_t:.6f}")
    print(f"Norm of z_p (Protein Embedding): {norm_z_p:.6f}")
    print(f"Norm of z_c (Facilitated Embedding): {norm_z_c:.6f}")

    print("\n=== Mean Squared Error (MSE) Results ===")
    print(f"MSE between Facilitated Embeddings (z_c) and Protein Embeddings (z_p): {mse_zc_zp:.6f}")
    print(f"MSE between Text Embeddings (z_t) and Protein Embeddings (z_p): {mse_zt_zp:.6f}")

    print("\n=== Max Mean Discrepancy (MMD) Results ===")
    print(f"MMD between Facilitated Embeddings (z_c) and Protein Embeddings (z_p): {mmd_zc_zp:.6f}")
    print(f"MMD between Text Embeddings (z_t) and Protein Embeddings (z_p): {mmd_zp_zt:.6f}")

    # Save output embeddings
    torch.save(embedding_dataset, args.output_data_path)
    print(f"\nFacilitator embeddings saved to {args.output_data_path}")


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
