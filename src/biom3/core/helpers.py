"""
Common helper functions

"""

import json
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn


def load_json_config(json_path: str) -> dict:
    """Load JSON configuration file."""
    with open(json_path, "r") as f:
        config = json.load(f)
    return config


def convert_to_namespace(config_dict: dict) -> Namespace:
    """Recursively convert a dictionary to an argparse Namespace."""
    for key, value in config_dict.items():
        if isinstance(value, dict):  # Recursively handle nested dictionaries
            config_dict[key] = convert_to_namespace(value)
    return Namespace(**config_dict)


def get_num_named_weights(model: nn.Module) -> int:
    """Retrieve total number of model weights associated with named parameters."""
    num = np.sum([v.numel() for k, v in model.named_parameters()])
    return num


def load_state_dict_with_correction_attempt(
        model,
        state_dict, 
        verbosity=2,
):
    """Load a state_dict and attempt to infer missing keys."""
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        if verbosity > 2:
            print("Encountered error loading state_dict:")
            print(e)
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, strict=False
        )
        if verbosity > 1 and missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if verbosity > 1 and unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
        for k in unexpected_keys:
            new_k = k.replace("weights_", "weights.")
            if new_k in missing_keys:
                state_dict[new_k] = state_dict.pop(k)
                if verbosity > 1:
                    print(f"Replaced state_dict key `{k}` with `{new_k}`")
        # Reload model with updated parameter names
        model.load_state_dict(
            state_dict, strict=True
        )


def compare_model_params(
        model1: nn.Module,
        model2: nn.Module,
        verbosity=1,
) -> dict:
    num_weights1 = get_num_named_weights(model1)
    num_weights2 = get_num_named_weights(model2)
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    set1 = set(params1.keys())
    set2 = set(params2.keys())
    intersection = set1.intersection(set2)
    set1_not_set2 = set1 - set2
    set2_not_set1 = set2 - set1
    max_differences = {}
    with torch.no_grad():
        for name in sorted(list(intersection)):
            vals1 = params1[name]
            vals2 = params2[name]
            print(type(vals1), type(vals2))
            if vals1.shape == vals2.shape:
                max_differences[name] = torch.max(torch.abs(vals1 - vals2))
            else:                
                max_differences[name] = np.inf # indicate mismatch in shape
    return {
        "num_weights1": num_weights1,
        "num_weights2": num_weights2,
        "model1_only": sorted(list(set1_not_set2)),
        "model2_only": sorted(list(set2_not_set1)),
        "common_names": sorted(list(intersection)),
        "differences": max_differences,
        "diff_shape_key": np.inf,
    }
