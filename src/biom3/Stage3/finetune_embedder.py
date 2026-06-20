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
from biom3.core.io import load_and_prepare_model
from biom3.backend.device import setup_logger

logger = setup_logger(__name__)


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


def build_text_to_zc_embedder(
        stage1_args,
        stage2_args,
        pencl_weights: str,
        facilitator_weights: str,
        device=None,
    ) -> TextToZcEmbedder:
    """Build the frozen text->z_c embedder and load PenCL + Facilitator weights.

    PenCL weights load with ``strict=False`` (protein-branch keys absent from
    this module are ignored, mirroring Stage 1 inference). Facilitator weights
    load into the ``facilitator`` submodule with the same key substitution used
    by Stage 2 sampling. The returned module is frozen and in eval mode so its
    embeddings are deterministic and match inference.
    """
    embedder = TextToZcEmbedder(stage1_args, stage2_args)

    logger.info("Loading PenCL text-branch weights from: %s", pencl_weights)
    load_and_prepare_model(
        embedder,
        pencl_weights,
        device=None,
        strict=False,
        eval_mode=False,
        attempt_correction=False,
    )

    logger.info("Loading Facilitator weights from: %s", facilitator_weights)
    load_and_prepare_model(
        embedder.facilitator,
        facilitator_weights,
        device=None,
        strict=True,
        eval_mode=False,
        attempt_correction=True,
        substitutions={"model.main.": "main."},
    )

    for p in embedder.parameters():
        p.requires_grad = False
    embedder.eval()
    if device is not None:
        embedder.to(device)
    return embedder
