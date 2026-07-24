"""
I/O module for Stage 3 ProteoScribe

"""

import os
import torch.nn as nn
import argparse

from biom3.core.io import (
    prepare_model, load_state_dict_unwrap_pl, strip_pl_model_prefix,
)
from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

import biom3.Stage3.cond_diff_transformer_layer as mod
import biom3.Stage3.PL_wrapper as PL_mod


_DEFAULT_SUBS = {
    "axial_pos_emb.weights_": "axial_pos_emb.weights.",
    "axial_pos_emb.weights.": "axial_pos_emb.weights_",
}


def build_model_ProteoScribe(
        config_args: argparse.Namespace
) -> nn.Module:
    return mod.get_model(
        args=config_args,
        data_shape=(config_args.image_size, config_args.image_size),
        num_classes=config_args.num_classes,
    )


def _strip_pl_model_prefix(state_dict: dict) -> dict:
    """Strip the ``model.`` prefix added by PL_ProtARDM (``self.model = model``).

    A no-op when no key carries the prefix, so raw state dicts saved as
    ``PL_module.model.state_dict()`` pass through unchanged.
    """
    return strip_pl_model_prefix(state_dict)


def _load_state_dict_from_file(path: str, device=None) -> dict:
    """Load a state_dict from a single file, handling raw and PL checkpoint formats.

    - Raw state dict (.bin, .pt): returned as-is, stripping any ``model.``
      prefix left over from saving ``PL_module.state_dict()`` directly.
    - PL checkpoint (.ckpt): extracts ``checkpoint["state_dict"]`` and strips
      the ``model.`` prefix added by PL_ProtARDM.
    """
    return load_state_dict_unwrap_pl(path, device=device)


def _load_state_dict_from_sharded_dir(checkpoint_dir: str, device=None) -> dict:
    """Merge a DeepSpeed ZeRO sharded checkpoint directory into one state_dict.

    Also accepts a plain Lightning checkpoint directory, in which case
    ``last.ckpt`` (or the lexically last ``.ckpt``) is loaded instead.
    """
    return load_state_dict_unwrap_pl(checkpoint_dir, device=device)


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
    - A directory path holding either a sharded DeepSpeed ZeRO checkpoint or a
      Lightning checkpoint.
    """
    model = build_model_ProteoScribe(config_args)

    state_dict = None
    if model_fpath is not None:
        if os.path.isdir(model_fpath):
            logger.info("Detected checkpoint directory: %s", model_fpath)
            state_dict = _load_state_dict_from_sharded_dir(model_fpath, device=device)
        else:
            logger.info("Loading weights from file: %s", model_fpath)
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
