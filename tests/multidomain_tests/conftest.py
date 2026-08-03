"""Shared fixtures for the multidomain tests.

Everything runs on CPU against a tiny randomly-initialised composed model, so the
suite needs no weights and no GPU.
"""

import argparse

import pytest
import torch

from biom3.Stage3.io import build_model_ProteoScribe
from biom3.Stage3.multidomain import build_multidomain_model


TINY_CONFIG = {
    "model_option": "transformer",
    "num_classes": 29,
    "num_y_class_labels": 6,
    "diffusion_steps": 64,
    "image_size": 8,
    "num_steps": 1,
    "actnorm": False,
    "perm_channel": "none",
    "perm_length": "reverse",
    "input_dp_rate": 0.0,
    "transformer_dim": 32,
    "transformer_heads": 4,
    "transformer_depth": 2,
    "transformer_blocks": 1,
    "transformer_dropout": 0.0,
    "transformer_reversible": False,
    "transformer_local_heads": 2,
    "transformer_local_size": 16,
    "text_emb_dim": 32,
    "facilitator": "MMD",
}
SEQ_LEN = TINY_CONFIG["diffusion_steps"]
EMB_DIM = TINY_CONFIG["text_emb_dim"]
NUM_DOMAINS = 2
BATCH = 3


@pytest.fixture
def tiny_args():
    return argparse.Namespace(**TINY_CONFIG)


def make_experts(tiny_args, num_domains=NUM_DOMAINS, seed=0):
    """Independently initialised experts, so a mix-up between them is detectable."""
    experts = []
    for d in range(num_domains):
        torch.manual_seed(seed + d)
        expert = build_model_ProteoScribe(tiny_args)
        expert.eval()
        experts.append(expert)
    return experts


def make_composed(tiny_args, num_domains=NUM_DOMAINS, seed=0):
    torch.manual_seed(seed + 100)
    model = build_multidomain_model(
        tiny_args, num_domains, experts=make_experts(tiny_args, num_domains, seed)
    )
    model.eval()
    return model


@pytest.fixture
def composed(tiny_args):
    return make_composed(tiny_args)


def make_batch(num_domains=NUM_DOMAINS, batch=BATCH, seed=7, n_real=20):
    """Token grids with a real prefix and a PAD tail, plus timesteps and z_c."""
    from biom3.Stage3.multidomain.model import PAD_ID

    torch.manual_seed(seed)
    x = torch.full((batch, num_domains, SEQ_LEN), PAD_ID, dtype=torch.long)
    for d in range(num_domains):
        # Vary the real length per domain, as real canvases do.
        length = n_real - 4 * d
        x[:, d, :length] = torch.randint(1, 23, (batch, length))
    t = torch.randint(0, SEQ_LEN, (batch,)).float()
    y_c = torch.randn(batch, num_domains, EMB_DIM)
    return x, t, y_c


@pytest.fixture
def batch():
    return make_batch()
