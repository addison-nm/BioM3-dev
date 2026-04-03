"""BioM3 Stage 3: ProteoScribe sampling

Mimics the workflow described at
    https://huggingface.co/niksapraljak1/BioM3#stage-3-proteoscribe

Preparation:

    Corresponding config file:
        configs/stage3_config_ProteoScribe_sample.json

Example usage:

biom3_ProteoScribe_sample \
    --input_path "outputs/facilitator_embeddings.pt" \
    --json_path "configs/stage3_config_ProteoScribe_sample.json" \
    --model_path "./weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin" \
    --output_path "outputs/generated_sequences.pt"

Example usage (reproducible run with fixed seed, CPU):

biom3_ProteoScribe_sample \
    --input_path "outputs/facilitator_embeddings.pt" \
    --json_path "configs/stage3_config_ProteoScribe_sample.json" \
    --model_path "./weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin" \
    --output_path "outputs/generated_sequences.pt" \
    --seed 42 \
    --device cpu

"""

import copy
import os
import sys
from argparse import Namespace
from datetime import datetime
import random
import json
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn

import biom3.Stage3.sampling_analysis as Stage3_sample_tools
import biom3.Stage3.animation_tools as Stage3_ani_tools
from biom3.Stage3.io import prepare_model_ProteoScribe
from biom3.core.run_utils import (
    get_biom3_version,
    get_git_hash,
    setup_file_logging,
    teardown_file_logging,
    write_manifest,
)
from biom3.backend.device import setup_logger

logger = setup_logger(__name__)


# Step 0: Argument Parser Function
def parse_arguments(args):
    parser = argparse.ArgumentParser(description="BioM3 Inference Script (Stage 1)")
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help="Path to input embeddings")
    parser.add_argument('-c', '--json_path', type=str, required=True,
                        help="Path to the JSON configuration file (stage3_config_ProteoScribe_sample.json)")
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help="Path to the pre-trained model weights (pytorch_model.bin)")
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help="Path to save output embeddings")
    parser.add_argument('--seed', type=int, default=0,
                        help="seed for random number generation")
    parser.add_argument('--device', type=str, default="cuda",
                        choices=["cpu", "cuda", "xpu"], help="available device")
    return parser.parse_args(args)


# Step 1: Load JSON configuration
def load_json_config(json_path):
    """Load JSON configuration file."""
    with open(json_path, "r") as f:
        config = json.load(f)
    return config


# Step 2: Convert JSON dictionary to Namespace
def convert_to_namespace(config_dict):
    """Recursively convert a dictionary to an argparse Namespace."""
    for key, value in config_dict.items():
        if isinstance(value, dict):  # Recursively handle nested dictionaries
            config_dict[key] = convert_to_namespace(value)
    return Namespace(**config_dict)


# Step 3: load model with pretrained weights
def prepare_model(args, config_args) ->nn.Module:
    """
    Prepare the model and PyTorch Lightning Trainer using a flat args object.

    Supports raw state dicts (.bin, .pth, .pt), Lightning checkpoints (.ckpt),
    and sharded DeepSpeed checkpoint directories. Format is auto-detected.
    """
    model = prepare_model_ProteoScribe(
        config_args=config_args,
        model_fpath=args.model_path,
        device=config_args.device,
        strict=True,
        eval=True,
        attempt_correction=True,
    )
    logger.info("Stage 3 model loaded from: %s (loaded on %s)", args.model_path, config_args.device)
    return model


# Step 4: Sample sequences from the model
@torch.no_grad()
def batch_stage3_generate_sequences(
        args: any,
        model: nn.Module,
        z_t: torch.Tensor
    ) -> pd.Series:
    """
    Generates protein sequences in batches using a denoising model.

    Args:
        args (any): Configuration object containing model and sampling parameters.
        model (nn.Module): The pre-trained model used for denoising and generation.
        z_t (torch.Tensor): Input tensor representing initial samples for sequence generation.

    Returns:
        pd.Series: A dictionary containing generated sequences for each replica.
    """

    # Handle z_t if passed as a list of tensors
    if isinstance(z_t, list) and all(isinstance(item, torch.Tensor) for item in z_t):
        logger.info("z_t is a list of tensors with %s tensors.", len(z_t))
        z_t = torch.stack(z_t)

    # Move model and inputs to the target device (CPU or CUDA)
    model.to(args.device)
    z_t = z_t.to(args.device)

    # Amino acid tokenization including special characters
    tokens = [
        '-', '<START>', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
        'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '<END>', '<PAD>',
        'X', 'U', 'Z', 'B', 'O'  # Special characters
    ]

    # Build flat work list of (prompt_idx, replica_idx) pairs to parallelize
    # across both prompts and replicates
    num_prompts = z_t.size(0)
    work_items = [
        (p_idx, r_idx)
        for p_idx in range(num_prompts)
        for r_idx in range(args.num_replicas)
    ]

    # Pre-allocate output: design_sequences[replica_idx][prompt_idx]
    design_sequences = [[None] * num_prompts for _ in range(args.num_replicas)]

    # Process all (prompt, replica) pairs in batches
    for batch_start in range(0, len(work_items), args.batch_size_sample):
        batch = work_items[batch_start : batch_start + args.batch_size_sample]
        current_batch_size = len(batch)

        # Gather conditioning vectors for this batch (different prompts per item)
        batch_prompt_indices = [p_idx for p_idx, r_idx in batch]
        batched_z_text_sample = z_t[batch_prompt_indices]

        # Generate random permutations for each sample in the batch
        batch_perms = torch.stack([torch.randperm(args.diffusion_steps) for _ in range(current_batch_size)])

        # Generate denoised samples using the model
        mask_realization_list, _ = Stage3_sample_tools.batch_generate_denoised_sampled(
            args=args,
            model=model,
            extract_digit_samples=torch.zeros(current_batch_size, args.diffusion_steps),
            extract_time=torch.zeros(current_batch_size).long(),
            extract_digit_label=batched_z_text_sample,
            sampling_path=batch_perms
        )

        # Unpack results into the correct (replica, prompt) slots
        for i, (p_idx, r_idx) in enumerate(batch):
            mask_realization = mask_realization_list[-1][i]
            design_sequence = Stage3_ani_tools.convert_num_to_char(tokens, mask_realization[0])
            clean_sequence = design_sequence.replace('<START>', '').replace('<END>', '').replace('<PAD>', '')
            design_sequences[r_idx][p_idx] = clean_sequence

    assert all(seq is not None for row in design_sequences for seq in row), \
        "Not all (prompt, replica) pairs were generated"

    design_sequence_dict = {
        f'replica_{r_idx}': design_sequences[r_idx]
        for r_idx in range(args.num_replicas)
    }

    return design_sequence_dict


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
    np.random.seed(seed + 1)
    random.seed(seed + 2)
    return


def main(args, _setup_logging=True):
    # Parse arguments
    config_args_parser = args

    # Set up dual logging (console + file)
    outdir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(outdir, exist_ok=True)
    file_handler = None
    if _setup_logging:
        log_path, file_handler = setup_file_logging(outdir)
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("ProteoScribe sampling (Stage 3)")
    logger.info("biom3 version: %s (git: %s)", get_biom3_version(), get_git_hash())
    logger.info("Command:     %s", " ".join(sys.argv))
    logger.info("=" * 60)

    seed = config_args_parser.seed

    if seed <= 0:
        seed = np.random.randint(2**32)

    set_seed(seed)
    logger.info("Seed: %s", seed)

    # Load and convert JSON config
    config_dict = load_json_config(config_args_parser.json_path)
    raw_config = copy.deepcopy(config_dict)
    config_args = convert_to_namespace(config_dict)

    config_args.device = config_args_parser.device

    # load test dataset
    embedding_dataset = torch.load(config_args_parser.input_path)

    # load model
    model = prepare_model(args=config_args_parser, config_args=config_args)

    # sample sequences
    design_sequence_dict = batch_stage3_generate_sequences(
            args=config_args,
            model=model,
            z_t=embedding_dataset['z_c']
    )

    logger.info("design_sequence_dict=%s", design_sequence_dict)

    torch.save(design_sequence_dict, f"{config_args_parser.output_path}")

    # Write manifest and clean up logging
    elapsed = datetime.now() - start_time
    z_c = embedding_dataset['z_c']
    num_prompts = z_c.size(0) if isinstance(z_c, torch.Tensor) else len(z_c)
    if _setup_logging:
        write_manifest(
            args, outdir, start_time, elapsed,
            outputs={
                "num_prompts": num_prompts,
                "num_replicas": config_args.num_replicas,
                "seed": seed,
                "total_sequences": num_prompts * config_args.num_replicas,
                "output_file": os.path.abspath(args.output_path),
            },
            resolved_paths={
                "input_path": os.path.abspath(args.input_path),
                "model_path": os.path.abspath(args.model_path),
                "json_config": os.path.abspath(args.json_path),
            },
            config_contents=raw_config,
        )
        logger.info("Done in %s", elapsed)
        teardown_file_logging("biom3", file_handler)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)    
