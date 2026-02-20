from argparse import Namespace
import json
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import Stage3_source.PL_wrapper as Stage3_PL_mod
import Stage3_source.cond_diff_transformer_layer as Stage3_mod
import Stage3_source.sampling_analysis as Stage3_sample_tools
import Stage3_source.animation_tools as Stage3_ani_tools


# Step 1: Load JSON configuration
def load_json_config(json_path):
    """
    Load JSON configuration file.
    """
    with open(json_path, "r") as f:
        config = json.load(f)
    # print("Loaded JSON config:", config)
    return config

# Step 2: Convert JSON dictionary to Namespace
def convert_to_namespace(config_dict):
    """
    Recursively convert a dictionary to an argparse Namespace.
    """
    for key, value in config_dict.items():
        if isinstance(value, dict):  # Recursively handle nested dictionaries
            config_dict[key] = convert_to_namespace(value)
    return Namespace(**config_dict)


# Step 3: load model with pretrained weights
def prepare_model(args, config_args) ->nn.Module:
    """
    Prepare the model and PyTorch Lightning Trainer using a flat args object.
    """

    # Initialize the model graph
    model = Stage3_mod.get_model(
        args=config_args,
        data_shape=(config_args.image_size, config_args.image_size),
        num_classes=config_args.num_classes
    )
    
    # Load state_dict into the model with map_location="cpu"
    model.load_state_dict(torch.load(args.model_path, map_location=config_args.device))
    model.eval()
    
    print(f"Stage 3 model loaded from: {args.model_path} (loaded on {config_args.device})")
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
        print(f"z_t is a list of tensors with {len(z_t)} tensors.")
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

    # Initialize a dictionary to store generated sequences for each replica
    design_sequence_dict = {f'replica_{ii}': [] for ii in range(args.num_replicas)}

    # Loop over input samples (each z_t) and generate sequences
    for idx_sample, z_text_sample in enumerate(z_t):

        # Process in batches to optimize memory and speed
        for batch_start in range(0, args.num_replicas, args.batch_size_sample):
            current_batch_size = min(args.batch_size_sample, args.num_replicas - batch_start)

            # Prepare batched input for current batch
            batched_z_text_sample = z_text_sample.unsqueeze(0).repeat(current_batch_size, 1)

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

            # Convert generated numeric sequences to amino acid sequences
            for i, mask_realization in enumerate(mask_realization_list[-1]):
                design_sequence = Stage3_ani_tools.convert_num_to_char(tokens, mask_realization[0])
                clean_sequence = design_sequence.replace('<START>', '').replace('<END>', '').replace('<PAD>', '')
                design_sequence_dict[f'replica_{batch_start + i}'].append(clean_sequence)

    return design_sequence_dict



# Step 5: Argument Parser Function
def parse_arguments():
    
    parser = argparse.ArgumentParser(description="BioM3 Inference Script (Stage 1)")
    parser.add_argument('--json_path', type=str, required=True,
                                    help="Path to the JSON configuration file (stage1_config.json)")
    parser.add_argument('--model_path', type=str, required=True,
                                    help="Path to the pre-trained model weights (pytorch_model.bin)")
    parser.add_argument('--input_path', type=str, required=True,
                                    help="Path to save input embeddings")
    parser.add_argument('--output_path', type=str, required=True,
                                    help="Path to save output embeddings")
    return parser.parse_args()


if __name__ == '__main__':
   
    # Parse arguments
    config_args_parser = parse_arguments()

    # Load and convert JSON config
    config_dict = load_json_config(config_args_parser.json_path)
    config_args = convert_to_namespace(config_dict) 

    # Set device if not specified in config
    config_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    
    print(f'{design_sequence_dict=}')
