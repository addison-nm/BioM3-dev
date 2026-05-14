"""CPU smoke tests for biom3.rl.gdpo.

Mirrors tests/rl_tests/test_grpo_smoke.py — exercises the GDPO building
blocks (quadrature, SDMC ELBO, sequence-level PPO step) without
requiring real Stage 1/Stage 2 weights, ESMFold, or a GPU. Uses the
mini Stage 3 fixture under tests/_data/models/stage3/.
"""

import copy
import os

import numpy as np
import pytest
import torch

from tests.conftest import DATDIR
from biom3.core.helpers import convert_to_namespace, load_json_config
from biom3.Stage3.io import build_model_ProteoScribe
from biom3.rl.gdpo import (
    GDPOConfig,
    MASK_ID,
    _build_grid,
    _build_shared_corruptions,
    _elbo_sdmc,
    _tokenwise_k3_kl,
)
from biom3.rl.grpo import PAD_ID, decode_tokens, diffusion_rollout
from biom3.rl.rewards import StubReward


MINI_WEIGHTS = os.path.join(DATDIR, "models/stage3/weights/minimodel1_ds128_weights1.pth")
MINI_CONFIG = os.path.join(DATDIR, "configs/test_stage3_config_v2.json")


@pytest.fixture
def mini_s3():
    cfg = convert_to_namespace(load_json_config(MINI_CONFIG))
    cfg.device = "cpu"
    model = build_model_ProteoScribe(cfg)
    sd = torch.load(MINI_WEIGHTS, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    return model, cfg


def test_build_grid_uniform_midpoints():
    cfg = GDPOConfig(n_quadrature=3, quadrature_grid="uniform")
    L = 128
    idx, t, w = _build_grid(cfg, L, torch.device("cpu"))
    assert idx.shape == (3,)
    assert torch.allclose(t, torch.tensor([1.0 / 6, 0.5, 5.0 / 6]))
    assert torch.allclose(w, torch.full((3,), 1.0 / 3))
    # idx_n = round((1 - t_n) * L)
    expected_idx = torch.tensor([round((1 - 1.0 / 6) * L),
                                  round((1 - 0.5) * L),
                                  round((1 - 5.0 / 6) * L)], dtype=torch.long)
    assert torch.equal(idx, expected_idx)
    # Clamp keeps idx in [0, L-1]
    assert int(idx.min()) >= 0 and int(idx.max()) < L


def test_build_grid_explicit_with_weights():
    cfg = GDPOConfig(
        quadrature_grid="explicit",
        quadrature_points=[0.25, 0.75],
        quadrature_weights=[0.4, 0.6],
    )
    idx, t, w = _build_grid(cfg, L=64, device=torch.device("cpu"))
    assert torch.allclose(t, torch.tensor([0.25, 0.75]))
    assert torch.allclose(w, torch.tensor([0.4, 0.6]))


def test_build_grid_explicit_default_weights():
    cfg = GDPOConfig(quadrature_grid="explicit", quadrature_points=[0.1, 0.5, 0.9])
    _, _, w = _build_grid(cfg, L=64, device=torch.device("cpu"))
    assert torch.allclose(w, torch.full((3,), 1.0 / 3))


def test_build_grid_rejects_out_of_range():
    cfg = GDPOConfig(quadrature_grid="explicit", quadrature_points=[0.0, 0.5])
    with pytest.raises(ValueError):
        _build_grid(cfg, L=32, device=torch.device("cpu"))
    cfg2 = GDPOConfig(quadrature_grid="explicit", quadrature_points=[0.5, 1.5])
    with pytest.raises(ValueError):
        _build_grid(cfg2, L=32, device=torch.device("cpu"))


def test_elbo_sdmc_shape_grad_finiteness(mini_s3):
    s3, cfg = mini_s3
    s3.train()
    BG, L = 2, cfg.diffusion_steps
    ids = torch.randint(1, cfg.num_classes, (BG, L), dtype=torch.int64)
    z_c = torch.randn(BG, cfg.text_emb_dim, requires_grad=False)

    gdpo_cfg = GDPOConfig(n_quadrature=2, quadrature_grid="uniform", inner_mc=1)
    idx_grid, t_floats, weights = _build_grid(gdpo_cfg, L, torch.device("cpu"))
    corruptions = _build_shared_corruptions(
        ids=ids,
        idx_grid=idx_grid,
        t_floats=t_floats,
        weights=weights,
        inner_mc=gdpo_cfg.inner_mc,
        device=torch.device("cpu"),
    )
    elbo = _elbo_sdmc(
        model=s3,
        ids=ids,
        z_c_rep=z_c,
        corruptions=corruptions,
        args_namespace=cfg,
        eps_t=gdpo_cfg.eps_t,
        inner_mc=gdpo_cfg.inner_mc,
    )
    assert elbo.shape == (BG,)
    assert torch.isfinite(elbo).all()
    # ELBO should be a sum of negative log-prob contributions ⇒ ≤ 0
    # (each per-token log-prob is ≤ 0; multiplied by positive 1/t and weights).
    assert (elbo <= 0).all()
    # Autograd should reach Stage 3 parameters.
    elbo.sum().backward()
    grads = [p.grad for p in s3.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
               for g in grads), "No finite, non-zero grad reached Stage 3 params"


def test_shared_corruptions_use_mask_token(mini_s3):
    """All masked positions in the corrupted x_t must contain MASK_ID = 0."""
    _, cfg = mini_s3
    BG, L = 2, cfg.diffusion_steps
    ids = torch.randint(1, cfg.num_classes, (BG, L), dtype=torch.int64)
    cfg_g = GDPOConfig(n_quadrature=2, quadrature_grid="uniform", inner_mc=1)
    idx_grid, t_floats, weights = _build_grid(cfg_g, L, torch.device("cpu"))
    corruptions = _build_shared_corruptions(
        ids=ids,
        idx_grid=idx_grid,
        t_floats=t_floats,
        weights=weights,
        inner_mc=1,
        device=torch.device("cpu"),
    )
    for c in corruptions:
        x_t = c["x_t"]
        idx_n = int(c["idx"][0, 0].item())
        # Exactly idx_n positions per row should be revealed (== ids); the rest masked (== MASK_ID).
        revealed = (x_t == ids)
        masked = (x_t == MASK_ID)
        assert (revealed | masked).all()
        # Mask count per row = L - idx_n.
        assert int(masked.sum(dim=1)[0].item()) == L - idx_n


def test_seq_log_ratio_zero_at_init(mini_s3):
    """If π_new == π_old, the sequence-level log-ratio is exactly 0.

    This is a sanity check: the importance ratio cannot drift on step 1
    when both policies share weights (regardless of mask sampling).
    """
    s3, cfg = mini_s3
    s3.eval()
    old_s3 = copy.deepcopy(s3).eval()
    for p in old_s3.parameters():
        p.requires_grad_(False)

    BG, L = 2, cfg.diffusion_steps
    ids = torch.randint(1, cfg.num_classes, (BG, L), dtype=torch.int64)
    z_c = torch.randn(BG, cfg.text_emb_dim)

    cfg_g = GDPOConfig(n_quadrature=2, quadrature_grid="uniform", inner_mc=1)
    idx_grid, t_floats, weights = _build_grid(cfg_g, L, torch.device("cpu"))
    corruptions = _build_shared_corruptions(
        ids=ids, idx_grid=idx_grid, t_floats=t_floats, weights=weights,
        inner_mc=1, device=torch.device("cpu"),
    )
    with torch.no_grad():
        elbo_old = _elbo_sdmc(old_s3, ids, z_c, corruptions, cfg, cfg_g.eps_t, 1)
        elbo_new = _elbo_sdmc(s3, ids, z_c, corruptions, cfg, cfg_g.eps_t, 1)
    assert torch.allclose(elbo_old, elbo_new, atol=1e-5)


def test_tokenwise_k3_kl_zero_when_policies_match(mini_s3):
    s3, cfg = mini_s3
    ref_s3 = copy.deepcopy(s3).eval()
    for p in ref_s3.parameters():
        p.requires_grad_(False)
    BG, L = 2, cfg.diffusion_steps
    ids = torch.randint(1, cfg.num_classes, (BG, L), dtype=torch.int64)
    z_c = torch.randn(BG, cfg.text_emb_dim)
    s3.eval()
    with torch.no_grad():
        kl = _tokenwise_k3_kl(s3, ref_s3, ids, z_c)
    assert torch.isfinite(kl)
    assert kl.abs().item() < 1e-5


def test_gdpo_inner_step_updates_params(mini_s3):
    """End-to-end: one GDPO step with N=2 modifies Stage 3 params and produces finite loss."""
    s3, cfg = mini_s3
    cfg.device = "cpu"
    device = torch.device("cpu")

    ref_s3 = copy.deepcopy(s3).eval()
    for p in ref_s3.parameters():
        p.requires_grad_(False)
    old_s3 = copy.deepcopy(s3).eval()
    for p in old_s3.parameters():
        p.requires_grad_(False)

    s3.train()
    optimizer = torch.optim.AdamW(s3.parameters(), lr=1e-3)
    reward_fn = StubReward(target_length=30)

    B, G = 1, 4
    BG = B * G
    z_cs = torch.randn(B, cfg.text_emb_dim)
    z_cs_rep = z_cs.repeat_interleave(G, dim=0)

    with torch.no_grad():
        ids_all = diffusion_rollout(old_s3, cfg, z_cs[0:1], G, device)
        seqs = [decode_tokens(ids_all[i]) for i in range(BG)]

    cfg_g = GDPOConfig(n_quadrature=2, quadrature_grid="uniform", inner_mc=1, eps=0.2)
    idx_grid, t_floats, weights = _build_grid(cfg_g, cfg.diffusion_steps, device)
    corruptions = _build_shared_corruptions(
        ids_all, idx_grid, t_floats, weights, inner_mc=1, device=device,
    )

    with torch.no_grad():
        elbo_old = _elbo_sdmc(old_s3, ids_all, z_cs_rep, corruptions, cfg, cfg_g.eps_t, 1)

    R = torch.tensor(reward_fn(seqs), dtype=torch.float32)
    Rg = R.view(B, G)
    adv = (Rg - Rg.mean(dim=-1, keepdim=True)).view(BG)        # unnormalized

    elbo_new = _elbo_sdmc(s3, ids_all, z_cs_rep, corruptions, cfg, cfg_g.eps_t, 1)
    log_ratio = elbo_new - elbo_old.detach()
    ratio = torch.exp(log_ratio)
    seq_lengths = (ids_all != PAD_ID).float().sum(dim=1).clamp(min=1.0)
    pg1 = -adv * ratio
    pg2 = -adv * ratio.clamp(1 - cfg_g.eps, 1 + cfg_g.eps)
    pg_loss = (torch.max(pg1, pg2) / seq_lengths).mean()
    kl_loss = _tokenwise_k3_kl(s3, ref_s3, ids_all, z_cs_rep)
    loss = pg_loss + 0.01 * kl_loss
    assert torch.isfinite(loss)

    p0 = next(s3.parameters()).detach().clone()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    p1 = next(s3.parameters()).detach()
    assert not torch.allclose(p0, p1), "GDPO step did not modify Stage 3 params"
