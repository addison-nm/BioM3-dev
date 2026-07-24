"""
I/O utilities

"""

import os
import tempfile
import torch
import torch.nn as nn
import argparse
import warnings
from typing import Optional

from biom3.backend.device import setup_logger
from biom3.core.helpers import load_state_dict_with_correction_attempt

logger = setup_logger(__name__)

# Prefix PyTorch Lightning wrappers add to every key via ``self.model = model``.
_PL_MODEL_PREFIX = "model."


def load_and_prepare_model(
    model: nn.Module,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    strict: bool = True,
    eval_mode: bool = False,
    attempt_correction: bool = False,
    substitutions: dict = {},
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
        substitutions=substitutions,
        verbosity=verbosity,
    )


def load_state_dict(weights_path: str, device=None) -> dict:
    """Load a state_dict from disk, handling different checkpoint formats."""
    # Map to CPU when no device is given so checkpoints saved on another backend
    # (e.g. Aurora/XPU) deserialize on CUDA/CPU; prepare_model moves to the
    # target device afterward. Mirrors the Stage 1 inference fix (eb2920b).
    checkpoint = torch.load(weights_path, map_location=device or "cpu")
    # Handle lightning-style checkpoints
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def strip_pl_model_prefix(state_dict: dict) -> dict:
    """Strip the ``model.`` prefix that PL wrappers add to every key.

    A no-op when no key carries the prefix, so raw state dicts (``.bin``/``.pt``
    saved from a bare ``nn.Module``) pass through unchanged.
    """
    if not any(k.startswith(_PL_MODEL_PREFIX) for k in state_dict):
        return state_dict
    return {
        (k[len(_PL_MODEL_PREFIX):] if k.startswith(_PL_MODEL_PREFIX) else k): v
        for k, v in state_dict.items()
    }


def _resolve_lightning_ckpt_in_dir(checkpoint_dir: str) -> str:
    """Pick a single Lightning .ckpt file from a directory.

    Prefers ``last.ckpt``; otherwise the lexically last ``.ckpt`` (which sorts
    ``epoch=NN-step=...`` correctly).
    """
    last = os.path.join(checkpoint_dir, "last.ckpt")
    if os.path.isfile(last):
        return last
    candidates = sorted(
        f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .ckpt files in {checkpoint_dir} (and no DeepSpeed `latest` marker)."
        )
    return os.path.join(checkpoint_dir, candidates[-1])


def load_state_dict_unwrap_pl(weights_path: str, device=None) -> dict:
    """Load a state_dict from a raw file, Lightning checkpoint, or DeepSpeed dir.

    Unlike :func:`load_state_dict`, this also strips the ``model.`` prefix that
    PL wrappers add. Stage 1 (``PL_PEN_CL``, ``pfam_PL_PEN_CL``), Stage 2
    (``PL_Facilitator``) and Stage 3 (``PL_ProtARDM``) all set
    ``self.model = model``, so their ``.ckpt`` state dicts are keyed
    ``model.<submodule>....``. Loading one into a bare ``nn.Module`` without
    stripping makes every key unexpected and every parameter missing.
    """
    if os.path.isdir(weights_path):
        if os.path.exists(os.path.join(weights_path, "latest")):
            from biom3.backend.device import BACKEND_NAME, _XPU
            if BACKEND_NAME == _XPU:
                from lightning.pytorch.utilities.deepspeed import (
                    convert_zero_checkpoint_to_fp32_state_dict)
            else:
                from pytorch_lightning.utilities.deepspeed import (
                    convert_zero_checkpoint_to_fp32_state_dict)
            logger.info("Detected DeepSpeed ZeRO sharded directory: %s", weights_path)
            fd, tmp_path = tempfile.mkstemp(suffix=".pt")
            os.close(fd)
            try:
                convert_zero_checkpoint_to_fp32_state_dict(weights_path, tmp_path)
                checkpoint = torch.load(tmp_path, map_location=device or "cpu")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            inner = _resolve_lightning_ckpt_in_dir(weights_path)
            logger.info("Detected Lightning checkpoint directory; using %s", inner)
            checkpoint = torch.load(inner, map_location=device or "cpu")
    else:
        checkpoint = torch.load(weights_path, map_location=device or "cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return strip_pl_model_prefix(checkpoint["state_dict"])
    if isinstance(checkpoint, dict):
        return strip_pl_model_prefix(checkpoint)
    return checkpoint


def prepare_model(
    model: nn.Module,
    state_dict: Optional[dict] = None,
    device: Optional[str] = None,
    strict: bool = True,
    eval_mode: bool = False,
    attempt_correction: bool = False,
    substitutions: dict = {},
    verbosity: int = 1,
) -> nn.Module:
    """Attach weights, move to device, and set mode."""
    
    if state_dict is not None:
        if attempt_correction:
            load_state_dict_with_correction_attempt(
                model,
                state_dict,
                verbosity=verbosity,
                substitutions=substitutions,
            )
        else:
            model.load_state_dict(state_dict, strict=strict)
    
    if device is not None:
        model.to(device)
    
    if eval_mode:
        model.eval()

    if verbosity:
        logger.info("Model prepared successfully.")
    return model
