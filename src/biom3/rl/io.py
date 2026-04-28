"""Model loading helpers for GRPO.

Wraps the existing per-stage I/O so the GRPO trainer can build the frozen
Stage 1 + Stage 2 conditioning chain and a trainable Stage 3 policy from a
single config.
"""

import os
import tempfile
from argparse import Namespace
from typing import Optional

import torch
import torch.nn as nn

import biom3.Stage1.model as S1mod
from biom3.Stage3.io import prepare_model_ProteoScribe
from biom3.backend.device import BACKEND_NAME, _XPU, setup_logger

if BACKEND_NAME == _XPU:
    from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
else:
    from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

logger = setup_logger(__name__)


_PL_MODEL_PREFIX = "model."


def _resolve_lightning_ckpt_in_dir(checkpoint_dir: str) -> str:
    """Pick a single Lightning .ckpt file from a directory.

    Prefers `last.ckpt` (Lightning convention). If absent, falls back to
    the lexically last `.ckpt` (which sorts `epoch=NN-step=...` correctly).
    Raises FileNotFoundError if no .ckpt files are found.
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


def _strip_pl_prefix(sd: dict) -> dict:
    return {
        (k[len(_PL_MODEL_PREFIX):] if k.startswith(_PL_MODEL_PREFIX) else k): v
        for k, v in sd.items()
    }


def _load_state_dict_unwrap_pl(weights_path: str, device=None) -> dict:
    """Load a state_dict from a raw file, Lightning checkpoint, or DeepSpeed
    ZeRO sharded directory. Strips the ``model.`` prefix that PL wrappers
    add when ``self.model = model``.

    Both Stage 1 (``PL_PEN_CL``, ``pfam_PL_PEN_CL``) and Stage 2
    (``PL_Facilitator``) follow that pattern, so their .ckpt state-dicts
    are keyed as ``model.<encoder>....``. Loading them into a raw
    nn.Module requires stripping that prefix. (Stage 3 has its own
    ``io.py`` that does the same; this mirrors that logic for stages
    whose checkpoints aren't routed through ``prepare_model_ProteoScribe``.)
    """
    if os.path.isdir(weights_path):
        if os.path.exists(os.path.join(weights_path, "latest")):
            logger.info("Detected DeepSpeed ZeRO sharded directory: %s", weights_path)
            fd, tmp_path = tempfile.mkstemp(suffix=".pt")
            os.close(fd)
            try:
                convert_zero_checkpoint_to_fp32_state_dict(weights_path, tmp_path)
                checkpoint = torch.load(tmp_path, map_location=device)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            inner = _resolve_lightning_ckpt_in_dir(weights_path)
            logger.info("Detected Lightning checkpoint directory; using %s", inner)
            checkpoint = torch.load(inner, map_location=device)
    else:
        checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return _strip_pl_prefix(checkpoint["state_dict"])
    if isinstance(checkpoint, dict):
        return _strip_pl_prefix(checkpoint)
    return checkpoint


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
    return prepare_model_ProteoScribe(
        config_args=cfg,
        model_fpath=weights_path,
        device=device,
        strict=False,
        eval=False,
    )
