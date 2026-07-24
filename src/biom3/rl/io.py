"""Model loading helpers for GRPO.

Wraps the existing per-stage I/O so the GRPO trainer can build the frozen
Stage 1 + Stage 2 conditioning chain and a trainable Stage 3 policy from a
single config.
"""

from argparse import Namespace
from typing import Optional

import torch.nn as nn

import biom3.Stage1.model as S1mod
from biom3.Stage3.io import prepare_model_ProteoScribe
from biom3.core.io import load_state_dict_unwrap_pl as _load_state_dict_unwrap_pl
from biom3.backend.device import setup_logger

logger = setup_logger(__name__)


def _attach(model: nn.Module, sd: Optional[dict], device, eval_mode: bool):
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            logger.warning("missing keys (%d): %s ...", len(missing), missing[:3])
        if unexpected:
            logger.warning("unexpected keys (%d): %s ...", len(unexpected), unexpected[:3])
    if device is not None:
        model.to(device)
    if eval_mode:
        model.eval()
    return model


def load_pencl_frozen(
    cfg: Namespace,
    weights_path: Optional[str],
    device: Optional[str] = None,
) -> nn.Module:
    model = S1mod.pfam_PEN_CL(args=cfg)
    sd = _load_state_dict_unwrap_pl(weights_path, device=device) if weights_path else None
    model = _attach(model, sd, device=device, eval_mode=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_facilitator_frozen(
    cfg: Namespace,
    weights_path: Optional[str],
    device: Optional[str] = None,
) -> nn.Module:
    model = S1mod.Facilitator(
        in_dim=cfg.emb_dim,
        hid_dim=cfg.hid_dim,
        out_dim=cfg.emb_dim,
        dropout=cfg.dropout,
    )
    sd = _load_state_dict_unwrap_pl(weights_path, device=device) if weights_path else None
    model = _attach(model, sd, device=device, eval_mode=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_proteoscribe_trainable(
    cfg: Namespace,
    weights_path: Optional[str],
    device: Optional[str] = None,
) -> nn.Module:
    model = prepare_model_ProteoScribe(
        config_args=cfg,
        model_fpath=weights_path,
        device=device,
        strict=True,
        eval=False,
        attempt_correction=True,
        verbosity=2
    )
    return model
