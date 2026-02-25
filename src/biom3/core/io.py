"""
I/O utilities

"""

import torch
import torch.nn as nn
import argparse
import warnings
from typing import Optional

from biom3.core.helpers import load_state_dict_with_correction_attempt


def load_and_prepare_model(
    model: nn.Module,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    strict: bool = True,
    eval_mode: bool = False,
    attempt_correction: bool = False,
    verbosity: int = 1,
) -> nn.Module:
    """Wrapper to load and attach weights, and specify model device/mode."""

    state_dict = None
    if weights_path:
        state_dict = load_state_dict(weights_path, device=device)

    return prepare_model(
        model=model,
        state_dict=state_dict,
        device=device,
        strict=strict,
        eval_mode=eval_mode,
        attempt_correction=attempt_correction,
        verbosity=verbosity,
    )


def load_state_dict(weights_path: str, device=None) -> dict:
    """Load a state_dict from disk, handling different checkpoint formats."""
    checkpoint = torch.load(weights_path, map_location=device)
    # Handle lightning-style checkpoints
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def prepare_model(
    model: nn.Module,
    state_dict: Optional[dict] = None,
    device: Optional[str] = None,
    strict: bool = True,
    eval_mode: bool = False,
    attempt_correction: bool = False,
    verbosity: int = 1,
) -> nn.Module:
    """Attach weights, move to device, and set mode."""
    
    if state_dict is not None:
        if attempt_correction:
            load_state_dict_with_correction_attempt(
                model,
                state_dict,
                verbosity=verbosity,
            )
        else:
            model.load_state_dict(state_dict, strict=strict)
    
    if device is not None:
        model.to(device)
    
    if eval_mode:
        model.eval()

    if verbosity:
        print("Model prepared successfully.")
    return model




# def load_weights_into_model(
#         model: nn.Module,
#         config_args: argparse.Namespace, 
#         weights_fpath=None, 
#         device=None, 
#         strict=True,
#         eval=False,
#         attempt_correction=False,
#         verbosity=2,
# ) -> nn.Module:
#     """Load model weights into the given model.

#     Args:
#         model (nn.Module): _description_
#         config_args (argparse.Namespace): _description_
#         weights_fpath (_type_, optional): _description_. Defaults to None.
#         device (_type_, optional): _description_. Defaults to None.
#         strict (bool, optional): _description_. Defaults to True.
#         eval (bool, optional): _description_. Defaults to False.
#         attempt_correction (bool, optional): _description_. Defaults to False.
#         verbosity (int, optional): _description_. Defaults to 2.

#     Returns:
#         nn.Module: _description_
#     """
#     if "device" in config_args:
#         if device is None and config_args.device is not None:
#             # Indicates that the device specified in the config wasn't passed.
#             msg = f"Loading weights into model without specifying device."
#             msg += f" config_args specifies device '{config_args.device}'"
#             msg += " but this value is not passed to prepare_model_ProteoScribe"
#             warnings.warn(msg)
#     if weights_fpath:
#         state_dict = torch.load(
#             weights_fpath, 
#             map_location=device,
#         )
#         if attempt_correction:
#             # Attempts to infer any missing keys. Ignores value of `strict`.
#             load_state_dict_with_correction_attempt(
#                 model,
#                 state_dict,
#                 verbosity=2,
#             )
#         else:
#             model.load_state_dict(state_dict, strict=strict)    
        
#     if device:
#         model.to(device)
#     if eval:
#         model.eval()
#     if verbosity:
#         print("Model loaded successfully with weights!")
#     return model
