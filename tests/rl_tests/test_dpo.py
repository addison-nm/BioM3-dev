"""CPU tests for biom3.rl.dpo and biom3.rl.preference_data.

Exercises the DPO building blocks without real Stage 1/2 weights or a GPU:

- preference-data ingestion (Case B grouping by prompt, Case C default caption),
  paired/weighted batch shapes, tokenization, invalid-sequence filtering;
- the Paired and Weighted ELBO-DPO losses on the mini Stage 3 fixture, with the
  exact log-ratio-is-zero-at-init identity (pi_theta == pi_ref) that pins the
  loss to log 2 (paired) / log K (weighted), plus gradient flow.
"""

import copy
import math
import os

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from tests.conftest import DATDIR
from biom3.core.helpers import convert_to_namespace, load_json_config
from biom3.Stage3.io import build_model_ProteoScribe
from biom3.Stage3.preprocess import encode_protein_sequence
from biom3.rl.dpo import DPOConfig, _paired_elbos, _weighted_elbos
from biom3.rl.gdpo import _build_grid
from biom3.rl.preference_data import (
    DEFAULT_CAPTION_KEY,
    PreferenceSampler,
    load_groups,
    valid_length,
)

MINI_WEIGHTS = os.path.join(DATDIR, "models/stage3/weights/minimodel1_ds128_weights1.pth")
MINI_CONFIG = os.path.join(DATDIR, "configs/test_stage3_config_v2.json")
CPU = torch.device("cpu")


@pytest.fixture
def mini_s3():
    cfg = convert_to_namespace(load_json_config(MINI_CONFIG))
    cfg.device = "cpu"
    model = build_model_ProteoScribe(cfg)
    model.load_state_dict(torch.load(MINI_WEIGHTS, map_location="cpu"), strict=True)
    return model, cfg


# ─────────────────────────────────────────────────────────────────────────────
# preference_data
# ─────────────────────────────────────────────────────────────────────────────


def _write_csv(tmp_path):
    rows = []
    # Case B: two prompts, each with functional + nonfunctional members.
    for p, cap in [(0, "prompt zero text"), (1, "prompt one text")]:
        for i in range(6):
            func = 1 if i < 3 else 0
            rows.append(dict(dataset="biom3", source="biom3", prompt=p,
                             prompt_text=cap, functional=func,
                             sequence="ACDEFGHIKL"[: 4 + i % 5] + "MNPQ",
                             score=0.9 - 0.1 * i))
    df = pd.DataFrame(rows)
    path = os.path.join(tmp_path, "pref.csv")
    df.to_csv(path, index=False)
    return path


def test_load_groups_case_b(tmp_path):
    groups = load_groups(_write_csv(tmp_path), group_by="prompt_text", min_group_size=2)
    assert len(groups) == 2
    caps = sorted(g.caption for g in groups)
    assert caps == ["prompt one text", "prompt zero text"]
    for g in groups:
        assert g.functional is not None
        assert len(g.seqs) == len(g.scores) == 6


def test_load_groups_case_c_default_caption(tmp_path):
    # group_by names a column that doesn't exist -> single default group.
    groups = load_groups(_write_csv(tmp_path), group_by="no_such_col",
                         default_caption="SH3 domain.")
    assert len(groups) == 1
    assert groups[0].caption_key == DEFAULT_CAPTION_KEY
    assert groups[0].caption == "SH3 domain."
    assert groups[0].functional is None


def test_invalid_sequences_dropped(tmp_path):
    df = pd.DataFrame([
        dict(dataset="vae", source="s", prompt=np.nan, prompt_text=np.nan,
             functional=np.nan, sequence="ACDEFG", score=1.0),
        dict(dataset="vae", source="s", prompt=np.nan, prompt_text=np.nan,
             functional=np.nan, sequence="ACDJEF", score=0.0),  # 'J' not in vocab
        dict(dataset="vae", source="s", prompt=np.nan, prompt_text=np.nan,
             functional=np.nan, sequence="", score=0.0),         # empty
    ])
    path = os.path.join(tmp_path, "v.csv")
    df.to_csv(path, index=False)
    groups = load_groups(path, group_by="prompt_text", min_group_size=1)
    assert len(groups) == 1
    assert groups[0].seqs == ["ACDEFG"]


def test_paired_and_weighted_batch_shapes(tmp_path):
    groups = load_groups(_write_csv(tmp_path), group_by="prompt_text")
    sampler = PreferenceSampler(groups, image_size=4, seed=0)  # L = 16
    assert len(sampler.captions) == 2

    pb = sampler.sample_paired_batch(5, pairing="label")
    assert pb["w_ids"].shape == (5, 16) and pb["l_ids"].shape == (5, 16)
    assert len(pb["caption_keys"]) == 5
    assert pb["w_ids"].dtype == torch.long

    mb = sampler.sample_paired_batch(4, pairing="margin", gap_level=0.5)
    assert mb["w_ids"].shape == (4, 16)

    wb = sampler.sample_weighted_batch(3, K=4)
    assert wb["ids"].shape == (3, 4, 16)
    assert wb["scores"].shape == (3, 4)


def test_encode_and_valid_length():
    ids = torch.tensor(encode_protein_sequence("ACDE", image_size=4))  # L=16
    # <START>=1, A=2,C=3,D=4,E=5, <END>=22, then '-'=23 padding.
    assert ids.tolist()[:6] == [1, 2, 3, 4, 5, 22]
    # valid length counts the 6 non-PAD tokens (START + 4 residues + END).
    assert int(valid_length(ids.unsqueeze(0))[0].item()) == 6


# ─────────────────────────────────────────────────────────────────────────────
# DPO losses (mini Stage 3)
# ─────────────────────────────────────────────────────────────────────────────


def test_paired_logratio_zero_at_init_and_loss_log2(mini_s3):
    s3, cfg = mini_s3
    s3.eval()
    ref = copy.deepcopy(s3).eval()
    L, B = cfg.diffusion_steps, 3
    dpo_cfg = DPOConfig(n_quadrature=2, beta=1.0, inner_mc=1,
                        gradient_checkpoint=False, length_normalize=True)
    grid = _build_grid(dpo_cfg, L, CPU)
    w_ids = torch.randint(1, cfg.num_classes, (B, L))
    l_ids = torch.randint(1, cfg.num_classes, (B, L))
    z_c = torch.randn(B, cfg.text_emb_dim)

    rho_w, rho_l = _paired_elbos(s3, ref, w_ids, l_ids, z_c, grid, cfg, dpo_cfg, L)
    # pi_theta == pi_ref with shared corruptions -> log-ratio is exactly 0.
    assert torch.allclose(rho_w, torch.zeros_like(rho_w), atol=1e-4)
    assert torch.allclose(rho_l, torch.zeros_like(rho_l), atol=1e-4)
    loss = -F.logsigmoid(rho_w - rho_l).mean()
    assert loss.item() == pytest.approx(math.log(2), abs=1e-3)


def test_paired_loss_grad_flows(mini_s3):
    s3, cfg = mini_s3
    s3.train()
    ref = copy.deepcopy(s3).eval()
    L, B = cfg.diffusion_steps, 2
    dpo_cfg = DPOConfig(n_quadrature=2, beta=1.0, inner_mc=1, gradient_checkpoint=False)
    grid = _build_grid(dpo_cfg, L, CPU)
    w_ids = torch.randint(1, cfg.num_classes, (B, L))
    l_ids = torch.randint(1, cfg.num_classes, (B, L))
    z_c = torch.randn(B, cfg.text_emb_dim)
    rho_w, rho_l = _paired_elbos(s3, ref, w_ids, l_ids, z_c, grid, cfg, dpo_cfg, L)
    loss = -F.logsigmoid(rho_w - rho_l).mean()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in s3.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
               for g in grads)


def test_weighted_logratio_zero_at_init_and_loss_logK(mini_s3):
    s3, cfg = mini_s3
    s3.eval()
    ref = copy.deepcopy(s3).eval()
    L, B, K = cfg.diffusion_steps, 2, 4
    dpo_cfg = DPOConfig(loss_type="weighted", n_quadrature=2, beta=1.0,
                        inner_mc=1, gradient_checkpoint=False, temperature=0.1)
    grid = _build_grid(dpo_cfg, L, CPU)
    ids = torch.randint(1, cfg.num_classes, (B, K, L))
    z_c = torch.randn(B, K, cfg.text_emb_dim)
    rho = _weighted_elbos(s3, ref, ids, z_c, grid, cfg, dpo_cfg, L)
    assert rho.shape == (B, K)
    assert torch.allclose(rho, torch.zeros_like(rho), atol=1e-4)

    scores = torch.randn(B, K)
    target = F.softmax(scores / dpo_cfg.temperature, dim=1)
    loss = -(target * F.log_softmax(rho, dim=1)).sum(dim=1).mean()
    # rho == 0 -> uniform model dist -> cross-entropy = log K for any target.
    assert loss.item() == pytest.approx(math.log(K), abs=1e-3)


def test_dpo_module_imports():
    import biom3.rl.dpo as dpo
    import biom3.rl.run_dpo_train as run
    from biom3.rl.__main__ import run_dpo_train
    assert hasattr(dpo, "dpo_train") and hasattr(dpo, "DPOConfig")
    assert callable(run.main) and callable(run_dpo_train)
