"""Frozen text -> z_c embedding front-end for Stage 3 finetuning.

ProteoScribe is conditioned on z_c, the facilitated text embedding. Only the
*text* branch of PenCL is needed to produce it:

    z_t  = text_projection(text_encoder(input_ids))      # Stage 1 (PenCL)
    z_c  = facilitator(z_t)                               # Stage 2 (Facilitator)

The protein branch (ESM-2) participates only in Stage 1's contrastive loss and
is never used for conditioning, so it is not instantiated here. This keeps the
front-end light enough (BioBERT + two small heads) to run batched on-device
during finetuning.
"""

import torch
import torch.nn as nn

import biom3.Stage1.model as stage1_mod
from biom3.core.io import load_state_dict_unwrap_pl
from biom3.backend.device import setup_logger

logger = setup_logger(__name__)


TEXT_BRANCH_PREFIXES = ("text_encoder.", "text_projection.")


class TextToZcEmbedder(nn.Module):
    """Maps tokenized text (BioBERT input_ids) to ProteoScribe's z_c condition.

    Holds only PenCL's text branch (``text_encoder`` + ``text_projection``) and
    the Stage 2 ``facilitator``. Attribute names match PenCL so its checkpoint
    loads with ``strict=False`` (the protein-branch keys are ignored).
    """

    def __init__(self, stage1_args, stage2_args):
        super().__init__()
        self.text_encoder = stage1_mod.TextEncoder(args=stage1_args)
        self.text_projection = stage1_mod.ProjectionHead(
            embedding_dim=stage1_args.text_encoder_embedding,
            args=stage1_args,
        )
        self.facilitator = stage1_mod.Facilitator(
            in_dim=stage2_args.emb_dim,
            hid_dim=stage2_args.hid_dim,
            out_dim=stage2_args.emb_dim,
            dropout=stage2_args.dropout,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        z_t = self.text_projection(self.text_encoder(input_ids))
        z_c = self.facilitator(z_t)
        return z_c


def _load_frozen_weights(module, weights_path, required_prefixes, label, device=None):
    """Load weights into ``module``, raising if required parameters stay unloaded.

    ``strict=False`` is unavoidable here: a PenCL checkpoint carries the protein
    branch, which this module does not instantiate. But a bare ``strict=False``
    load also swallows the case where *nothing* matches (e.g. a ``model.``
    -prefixed Lightning checkpoint loaded into a bare nn.Module), leaving a
    randomly initialised encoder that trains without any error. Restricting the
    strictness to the keys under ``required_prefixes`` keeps the intended
    tolerance while making a no-op load fail loudly.
    """
    state_dict = load_state_dict_unwrap_pl(weights_path, device=device)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)

    param_names = {name for name, _ in module.named_parameters()}
    required_missing = [
        k for k in missing if any(k.startswith(p) for p in required_prefixes)
    ]
    unloaded_params = [k for k in required_missing if k in param_names]
    unloaded_buffers = [k for k in required_missing if k not in param_names]

    if unloaded_params:
        raise RuntimeError(
            f"{label} weights at {weights_path} did not populate "
            f"{len(unloaded_params)}/{len(param_names)} required parameters "
            f"(e.g. {unloaded_params[:5]}). The module would train against "
            f"randomly initialised weights. Checkpoint keys look like "
            f"{sorted(state_dict)[:3]}."
        )
    if unloaded_buffers:
        logger.warning(
            "%s: %d buffer(s) not present in the checkpoint: %s",
            label, len(unloaded_buffers), unloaded_buffers[:5],
        )
    logger.info(
        "%s: loaded %d tensors from %s (%d unrelated checkpoint keys ignored)",
        label, len(state_dict) - len(unexpected), weights_path, len(unexpected),
    )


def build_text_to_zc_embedder(
        stage1_args,
        stage2_args,
        pencl_weights: str,
        facilitator_weights: str,
        device=None,
    ) -> TextToZcEmbedder:
    """Build the frozen text->z_c embedder and load PenCL + Facilitator weights.

    Both paths accept a raw state dict (``.bin``/``.pt``), a Lightning
    ``.ckpt``, a checkpoint directory, or a DeepSpeed ZeRO shard directory. The
    PenCL load ignores the protein-branch keys but requires every text-branch
    parameter to be populated; the Facilitator load requires all of its own. The
    returned module is frozen and in eval mode so its embeddings are
    deterministic and match inference.
    """
    if not pencl_weights:
        raise ValueError(
            "--pencl_weights is required for finetuning: without it the frozen "
            "text->z_c embedder runs on a randomly initialised projection head."
        )
    if not facilitator_weights:
        raise ValueError(
            "--facilitator_weights is required for finetuning: without it z_c is "
            "produced by a randomly initialised Facilitator."
        )

    embedder = TextToZcEmbedder(stage1_args, stage2_args)

    logger.info("Loading PenCL text-branch weights from: %s", pencl_weights)
    _load_frozen_weights(
        embedder, pencl_weights,
        required_prefixes=TEXT_BRANCH_PREFIXES,
        label="PenCL text branch",
        device=device,
    )

    logger.info("Loading Facilitator weights from: %s", facilitator_weights)
    _load_frozen_weights(
        embedder.facilitator, facilitator_weights,
        required_prefixes=("",),
        label="Facilitator",
        device=device,
    )

    for p in embedder.parameters():
        p.requires_grad = False
    embedder.eval()
    if device is not None:
        embedder.to(device)
    return embedder

class ProteintoZpEmbedder(nn.Module):
    """Maps tokenized proteins to ProteoScribe's z_p condition.

    Mirrors TexttoZcEmbedder but holds protein_encoder (ESM-2) +
    protein_projection. PenCL so its checkpoint loads with 
    ``strict=False`` so the text-branch keys are ignored.
    """

    def __init__(self, stage1_args):
        super().__init__()
        self.protein_encoder = stage1_mod.ProteinEncoder(args=stage1_args)
        self.protein_projection = stage1_mod.ProjectionHead(
            embedding_dim=stage1_args.protein_encoder_embedding,
            args=stage1_args,
        )

    @torch.no_grad()
    def embed_protein(self, sequences, device=None, batch_size=64):
        """get z_p for a list of raw aa sequences (compute once)"""
        device = device or next(self.parameters()).device
        convert = self.protein_encoder.alphabet.get_batch_converter()
        chunks = []
        for i in range(0, len(sequences), batch_size):
            batch = [(str(j), s) for j, s in enumerate(sequences[i:i +batch_size])]
            _, _, tokens = convert(batch)
            z_p = self.protein_projection(self.protein_encoder(tokens.to(device)))
            chunks.append(z_p.float().cpu())
            
        return torch.cat(chunks, dim=0)


def build_protein_to_zp_embedder(
        stage1_args,
        pencl_weights: str,
        device=None,
    ) -> ProteintoZpEmbedder:
    """Build the frozen protein - >z_p embedder that mirrors the above 
    text -> z_c embedder
    """
    embedder = ProteintoZpEmbedder(stage1_args)

    logger.info("Loading PenCL protein-branch weights from: %s", pencl_weights)
    load_and_prepare_model(
        embedder,
        pencl_weights,
        device=None,
        strict=False,
        eval_mode=False,
        attempt_correction=False,
    )

    for p in embedder.parameters():
        p.requires_grad = False
    embedder.eval()
    if device is not None:
        embedder.to(device)
    return embedder