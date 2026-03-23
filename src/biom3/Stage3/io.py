"""
I/O module for Stage 3 ProteoScribe

"""

import os
import tempfile
import torch
import torch.nn as nn
import argparse

from biom3.core.io import prepare_model
from biom3.backend.device import BACKEND_NAME, _XPU

if BACKEND_NAME == _XPU:
    from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
else:
    from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

import biom3.Stage3.cond_diff_transformer_layer as mod
import biom3.Stage3.PL_wrapper as PL_mod


_DEFAULT_SUBS = {
    "axial_pos_emb.weights_": "axial_pos_emb.weights.",
    "axial_pos_emb.weights.": "axial_pos_emb.weights_",
}

# Prefix used by PL_ProtARDM when saving state_dict (self.model = model)
_PL_MODEL_PREFIX = "model."


def build_model_ProteoScribe(
        config_args: argparse.Namespace
) -> nn.Module:
    return mod.get_model(
        args=config_args,
        data_shape=(config_args.image_size, config_args.image_size),
        num_classes=config_args.num_classes,
    )


def _load_state_dict_from_file(path: str, device=None) -> dict:
    """Load a state_dict from a single file, handling raw and PL checkpoint formats.

    - Raw state dict (.bin, .pt): returned as-is.
    - PL checkpoint (.ckpt): extracts ``checkpoint["state_dict"]`` and strips
      the ``model.`` prefix added by PL_ProtARDM.
    """
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        pl_state_dict = checkpoint["state_dict"]
        return {
            (k[len(_PL_MODEL_PREFIX):] if k.startswith(_PL_MODEL_PREFIX) else k): v
            for k, v in pl_state_dict.items()
        }
    return checkpoint


def _load_state_dict_from_sharded_dir(checkpoint_dir: str, device=None) -> dict:
    """Convert a DeepSpeed ZeRO sharded checkpoint directory to a single state_dict.

    Uses ``convert_zero_checkpoint_to_fp32_state_dict`` to merge shards into a
    temporary file, then loads and returns the result.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir, tmp_path)
        return _load_state_dict_from_file(tmp_path, device=device)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def prepare_model_ProteoScribe(
    config_args: argparse.Namespace,
    model_fpath=None,
    device=None,
    strict=True,
    eval=False,
    attempt_correction=False,
    substitutions=_DEFAULT_SUBS,
    verbosity=2,
) -> nn.Module:
    """Build and optionally load weights into the ProteoScribe model.

    ``model_fpath`` can be:
    - ``None``: returns a randomly-initialised model.
    - A file path to a raw state dict (``.bin``, ``.pt``) or a PyTorch Lightning
      checkpoint (``.ckpt``). The format is detected automatically.
    - A directory path containing a sharded DeepSpeed ZeRO checkpoint.
    """
    model = build_model_ProteoScribe(config_args)

    state_dict = None
    if model_fpath is not None:
        if os.path.isdir(model_fpath):
            if verbosity:
                print(f"Detected sharded checkpoint directory: {model_fpath}")
            state_dict = _load_state_dict_from_sharded_dir(model_fpath, device=device)
        else:
            if verbosity:
                print(f"Loading weights from file: {model_fpath}")
            state_dict = _load_state_dict_from_file(model_fpath, device=device)

    return prepare_model(
        model=model,
        state_dict=state_dict,
        device=device,
        strict=strict,
        eval_mode=eval,
        attempt_correction=attempt_correction,
        substitutions=substitutions,
        verbosity=verbosity,
    )
