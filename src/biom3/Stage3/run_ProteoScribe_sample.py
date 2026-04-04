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
import tqdm as tqdm

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
from biom3.backend.device import setup_logger, get_backend_name

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
    parser.add_argument('--unmasking_order', type=str, default=None,
                        choices=["random", "confidence", "confidence_no_pad"],
                        help="Position unmasking order: 'random' (default), "
                             "'confidence' (most-confident first), or "
                             "'confidence_no_pad' (most-confident first, "
                             "skipping positions predicted as PAD)")
    parser.add_argument('--token_strategy', type=str, default=None,
                        choices=["sample", "argmax"],
                        help="Token selection: 'sample' (Gumbel-max, default) or 'argmax' (deterministic)")
    parser.add_argument('--animate_prompts', type=str, nargs='+', default=None,
                        metavar='IDX',
                        help="Prompt indices to animate (e.g. 0 1 2), 'all', or 'none'. "
                             "Animation is disabled by default (omit this flag).")
    parser.add_argument('--animate_replicas', type=str, default='1',
                        metavar='N',
                        help="Replicas to animate: integer i animates range(0, i), "
                             "'all' animates every replica, 'none' disables. Default: 1.")
    parser.add_argument('--animation_dir', type=str, default=None,
                        help="Output directory for GIF animations. "
                             "Default: <output_dir>/animations/")
    parser.add_argument('--store_probabilities', action='store_true', default=False,
                        help="Store per-step conditional probabilities for each "
                             "(prompt, replica) pair as .npz files. "
                             "Shapes: probs[steps, seq_len, num_classes]. "
                             "Memory-intensive for long sequences or many replicas.")
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


def parse_animate_prompts(values):
    """Parse --animate_prompts raw values. Returns None, 'all', or a list of ints."""
    if values is None:
        return None
    if len(values) == 1 and values[0].lower() == 'none':
        return None
    if len(values) == 1 and values[0].lower() == 'all':
        return 'all'
    return [int(v) for v in values]


def parse_animate_replicas(value):
    """Parse --animate_replicas raw value. Returns None, 'all', or an int."""
    if value is None or value.lower() == 'none':
        return None
    if value.lower() == 'all':
        return 'all'
    return int(value)


def resolve_animate_prompts(parsed, num_prompts):
    """Resolve parsed prompt spec to a set of indices, or None if animation is off."""
    if parsed is None:
        return None
    if parsed == 'all':
        return set(range(num_prompts))
    invalid = [i for i in parsed if not (0 <= i < num_prompts)]
    if invalid:
        raise ValueError(
            f"--animate_prompts indices out of range: {invalid} "
            f"(valid range: 0–{num_prompts - 1})"
        )
    return set(parsed)


def resolve_animate_replicas(parsed, num_replicas):
    """Resolve parsed replica spec to a set of indices, or None if animation is off."""
    if parsed is None:
        return None
    if parsed == 'all':
        return set(range(num_replicas))
    if parsed > num_replicas:
        logger.warning(
            "--animate_replicas=%d exceeds num_replicas=%d; clamping to %d",
            parsed, num_replicas, num_replicas,
        )
        parsed = num_replicas
    return set(range(parsed))


# Step 4: Sample sequences from the model
@torch.no_grad()
def batch_stage3_generate_sequences(
        args: any,
        model: nn.Module,
        z_t: torch.Tensor,
        animate_prompts: set = None,
        animate_replicas: set = None,
        store_probabilities: bool = False,
    ) -> tuple:
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

    if '<PAD>' not in tokens:
        raise ValueError("Token vocabulary is missing '<PAD>' — cannot resolve pad_token_id")
    args.pad_token_id = tokens.index('<PAD>')

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

    animate = animate_prompts is not None and animate_replicas is not None
    animation_frames = {}  # (p_idx, r_idx) -> list of numpy arrays, one per diffusion step
    stored_probs = {}      # (p_idx, r_idx) -> np.ndarray [steps, seq_len, num_classes]

    # Process all (prompt, replica) pairs in batches
    num_batches = (len(work_items) + args.batch_size_sample - 1) // args.batch_size_sample
    logger.info(
        "Prompts: %d | Reps/prompt: %d | Total reps: %d | Batch size: %d | Batches: %d",
        num_prompts, args.num_replicas, num_prompts * args.num_replicas, 
        args.batch_size_sample, num_batches
    )
    for batch_start in tqdm.trange(0, len(work_items), args.batch_size_sample, desc="batch"):
        batch = work_items[batch_start : batch_start + args.batch_size_sample]
        current_batch_size = len(batch)

        # Gather conditioning vectors for this batch (different prompts per item)
        batch_prompt_indices = [p_idx for p_idx, r_idx in batch]
        batched_z_text_sample = z_t[batch_prompt_indices]

        # Generate denoised samples using the model
        unmasking_order = getattr(args, 'unmasking_order', 'random')
        if unmasking_order in ('confidence', 'confidence_no_pad'):
            mask_realization_list, _, batch_probs = Stage3_sample_tools.batch_generate_denoised_sampled_confidence(
                args=args,
                model=model,
                extract_digit_samples=torch.zeros(current_batch_size, args.diffusion_steps),
                extract_time=torch.zeros(current_batch_size).long(),
                extract_digit_label=batched_z_text_sample,
                store_probabilities=store_probabilities,
                skip_pad=(unmasking_order == 'confidence_no_pad'),
            )
        else:
            batch_perms = torch.stack([torch.randperm(args.diffusion_steps) for _ in range(current_batch_size)])
            mask_realization_list, _, batch_probs = Stage3_sample_tools.batch_generate_denoised_sampled(
                args=args,
                model=model,
                extract_digit_samples=torch.zeros(current_batch_size, args.diffusion_steps),
                extract_time=torch.zeros(current_batch_size).long(),
                extract_digit_label=batched_z_text_sample,
                sampling_path=batch_perms,
                store_probabilities=store_probabilities,
            )

        # Unpack results into the correct (replica, prompt) slots
        for i, (p_idx, r_idx) in enumerate(batch):
            mask_realization = mask_realization_list[-1][i]
            design_sequence = Stage3_ani_tools.convert_num_to_char(tokens, mask_realization[0])
            clean_sequence = design_sequence.replace('<START>', '').replace('<END>', '').replace('<PAD>', '')
            design_sequences[r_idx][p_idx] = clean_sequence

            if animate and p_idx in animate_prompts and r_idx in animate_replicas:
                animation_frames[(p_idx, r_idx)] = [
                    mask_realization_list[step][i][0].copy()
                    for step in range(args.diffusion_steps)
                ]

            if batch_probs is not None:
                # batch_probs shape: [steps, batch, seq_len, num_classes]
                stored_probs[(p_idx, r_idx)] = batch_probs[:, i]

    assert all(seq is not None for row in design_sequences for seq in row), \
        "Not all (prompt, replica) pairs were generated"

    design_sequence_dict = {
        f'replica_{r_idx}': design_sequences[r_idx]
        for r_idx in range(args.num_replicas)
    }

    return design_sequence_dict, animation_frames, tokens, stored_probs


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

    # Merge CLI overrides for sampling parameters (fall back to JSON, then defaults)
    for attr, default in [('unmasking_order', 'random'), ('token_strategy', 'sample')]:
        cli_val = getattr(config_args_parser, attr, None)
        if cli_val is not None:
            setattr(config_args, attr, cli_val)
        elif not hasattr(config_args, attr):
            setattr(config_args, attr, default)

    logger.info("Unmasking order: %s", config_args.unmasking_order)
    logger.info("Token strategy: %s", config_args.token_strategy)

    # load test dataset
    embedding_dataset = torch.load(config_args_parser.input_path)

    # load model
    model = prepare_model(args=config_args_parser, config_args=config_args)

    # TODO: re-enable torch.compile once Triton supports this GPU's compute
    # capability (sm_121a / CUDA 12.1). See ptxas "gpu-name" error in tests.
    # if get_backend_name() == "cuda":
    #     model = torch.compile(model)
    #     logger.info("Model compiled with torch.compile (inductor)")

    # Resolve animation targets
    z_c = embedding_dataset['z_c']
    num_prompts = z_c.size(0) if isinstance(z_c, torch.Tensor) else len(z_c)
    animate_prompts_set = resolve_animate_prompts(
        parse_animate_prompts(config_args_parser.animate_prompts),
        num_prompts,
    )
    animate_replicas_set = resolve_animate_replicas(
        parse_animate_replicas(config_args_parser.animate_replicas),
        config_args.num_replicas,
    )

    # sample sequences
    store_probs = getattr(config_args_parser, 'store_probabilities', False)
    design_sequence_dict, animation_frames, tokens, stored_probs = batch_stage3_generate_sequences(
            args=config_args,
            model=model,
            z_t=z_c,
            animate_prompts=animate_prompts_set,
            animate_replicas=animate_replicas_set,
            store_probabilities=store_probs,
    )

    logger.info("design_sequence_dict=%s", design_sequence_dict)

    torch.save(design_sequence_dict, f"{config_args_parser.output_path}")

    # Generate GIF animations
    if animation_frames:
        animation_dir = config_args_parser.animation_dir or os.path.join(outdir, "animations")
        os.makedirs(animation_dir, exist_ok=True)
        logger.info("Saving %d animation(s) to %s", len(animation_frames), animation_dir)
        for (p_idx, r_idx), frames in animation_frames.items():
            gif_path = os.path.join(animation_dir, f"prompt_{p_idx}_replica_{r_idx}.gif")
            Stage3_ani_tools.generate_sequence_animation(
                frames=frames,
                tokens=tokens,
                output_path=gif_path,
                title=f"Prompt {p_idx} \u00b7 Replica {r_idx}",
            )
            logger.info("Animation saved: %s", gif_path)

    # Save per-step conditional probabilities
    if stored_probs:
        probs_dir = os.path.join(outdir, "probabilities")
        os.makedirs(probs_dir, exist_ok=True)
        logger.info("Saving probabilities for %d (prompt, replica) pairs to %s",
                     len(stored_probs), probs_dir)
        for (p_idx, r_idx), probs_array in stored_probs.items():
            # probs_array shape: [steps, seq_len, num_classes]
            npz_path = os.path.join(probs_dir, f"prompt_{p_idx}_replica_{r_idx}.npz")
            np.savez_compressed(npz_path, probs=probs_array, tokens=tokens)
            logger.info("Probabilities saved: %s (shape=%s)", npz_path, probs_array.shape)

    # Write manifest and clean up logging
    elapsed = datetime.now() - start_time
    if _setup_logging:
        write_manifest(
            args, outdir, start_time, elapsed,
            outputs={
                "num_prompts": num_prompts,
                "num_replicas": config_args.num_replicas,
                "seed": seed,
                "unmasking_order": config_args.unmasking_order,
                "token_strategy": config_args.token_strategy,
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
