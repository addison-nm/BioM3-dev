"""Regression tests for GDPO's pre_unmask time-conditioning offset.

The Stage 3 sampler fix (commit 4b13045) shifted the pre_unmask sampling
path up by ``offset = sequence_length - D`` but left GDPO passing
``extract_time=zeros``. The two must agree: the sampler selects the
position to unmask with ``(sampling_path == temp_idx)``, so a mismatch
silently collapses every step onto column 0 instead of raising. Nothing
in tests/rl_tests exercised pre_unmask, which is how that slipped by.

See docs/bug_reports/2026-06-17_pre_unmask_time_offset.md.
"""

import os

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
    _gdpo_rollout,
    _time_offset,
)
from biom3.rl.grpo import PAD_ID, TOKENS


MINI_WEIGHTS = os.path.join(DATDIR, "models/stage3/weights/minimodel1_ds128_weights1.pth")
MINI_CONFIG = os.path.join(DATDIR, "configs/test_stage3_config_v2.json")

L_TOTAL = 128      # the mini model's architectural length
D = 32             # diffusion budget → offset 96
K = 3


def _mini_cfg(pre_unmask: bool):
    """Config only. The model must be built *before* diffusion_steps is
    overwritten with the budget — mirrors gdpo_train, which snapshots
    sequence_length and then narrows diffusion_steps to D.
    """
    cfg = convert_to_namespace(load_json_config(MINI_CONFIG))
    cfg.device = "cpu"
    cfg._silent_tqdm = True
    cfg.sequence_length = L_TOTAL
    cfg.pre_unmask = pre_unmask
    return cfg


def _build_mini(cfg):
    model = build_model_ProteoScribe(cfg)     # built at diffusion_steps = L_TOTAL
    model.load_state_dict(torch.load(MINI_WEIGHTS, map_location="cpu"), strict=True)
    return model.eval()


def _narrow_to_budget(cfg):
    cfg.diffusion_steps = D
    cfg.pre_unmask_strategy = "last_k"
    cfg.pre_unmask_fill_with = "PAD"
    return cfg


@pytest.fixture
def mini_s3_pre_unmask():
    cfg = _mini_cfg(pre_unmask=True)
    model = _build_mini(cfg)
    return model, _narrow_to_budget(cfg)


def _spy_on_sampler(monkeypatch):
    """Capture the (extract_time, sampling_path) the rollout hands the sampler."""
    import biom3.Stage3.sampling_analysis as S3sample

    seen = {}
    real_fn = S3sample.batch_generate_denoised_sampled

    def _spy(*args, **kwargs):
        seen["extract_time"] = kwargs["extract_time"].clone()
        seen["sampling_path"] = kwargs["sampling_path"].clone()
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(S3sample, "batch_generate_denoised_sampled", _spy)
    return seen


def _replayed_write_locations(seen, steps):
    """Columns the sampler writes, from its own position-selection rule.

    ``current_location = (sampling_path == temp_idx).long().argmax(-1)`` —
    replayed here because the property under test is index arithmetic, and
    the untrained mini model can legitimately emit MASK as a token, which
    makes the decoded output an unreliable witness.
    """
    path, temp_idx = seen["sampling_path"], seen["extract_time"].clone()
    locations = []
    for _ in range(steps):
        locations.append((path == temp_idx.unsqueeze(-1)).long().argmax(dim=-1))
        temp_idx += 1
    return torch.stack(locations, dim=1)       # (K, steps)


def test_time_offset_is_zero_without_pre_unmask():
    assert _time_offset(_mini_cfg(pre_unmask=False)) == 0


def test_time_offset_counts_the_pad_tail():
    assert _time_offset(_narrow_to_budget(_mini_cfg(pre_unmask=True))) == L_TOTAL - D


def test_rollout_unmasks_every_content_position(mini_s3_pre_unmask, monkeypatch):
    """The regression guard: with extract_time=zeros this collapsed to one."""
    model, cfg = mini_s3_pre_unmask
    seen = _spy_on_sampler(monkeypatch)

    with torch.no_grad():
        ids = _gdpo_rollout(model, cfg, torch.randn(1, cfg.text_emb_dim), K,
                            torch.device("cpu"))

    assert ids.shape == (K, L_TOTAL)
    locations = _replayed_write_locations(seen, D)
    for k in range(K):
        written = set(locations[k].tolist())
        assert written == set(range(D)), (
            f"row {k} wrote {len(written)} distinct positions, expected {D} — "
            "the sampling path and the time index disagree, so "
            "(path == temp_idx) never matched and argmax fell through to 0"
        )
    assert (ids[:, D:] == PAD_ID).all(), "pre-filled tail was overwritten"


def test_rollout_time_index_spans_the_true_revealed_count(mini_s3_pre_unmask, monkeypatch):
    """t must run [L_total - D, L_total), not [0, D)."""
    model, cfg = mini_s3_pre_unmask
    seen = _spy_on_sampler(monkeypatch)

    with torch.no_grad():
        _gdpo_rollout(model, cfg, torch.randn(1, cfg.text_emb_dim), K, torch.device("cpu"))

    offset = L_TOTAL - D
    assert (seen["extract_time"] == offset).all()
    # The clock starts where the path starts, and ends at L_total - 1.
    assert int(seen["sampling_path"].min()) == offset
    assert int(seen["extract_time"][0]) + D - 1 == L_TOTAL - 1


def test_rollout_default_path_is_unshifted(monkeypatch):
    """Offset-0 (no pre_unmask) behaviour must be unchanged."""
    cfg = _mini_cfg(pre_unmask=False)
    model = _build_mini(cfg)
    seen = _spy_on_sampler(monkeypatch)

    with torch.no_grad():
        ids = _gdpo_rollout(model, cfg, torch.randn(1, cfg.text_emb_dim), K,
                            torch.device("cpu"))

    assert (seen["extract_time"] == 0).all()
    assert ids.shape == (K, L_TOTAL)
    locations = _replayed_write_locations(seen, L_TOTAL)
    for k in range(K):
        assert set(locations[k].tolist()) == set(range(L_TOTAL))


def test_corruptions_shift_model_time_but_not_the_reveal_count():
    """idx does double duty; only the model-facing copy may be shifted."""
    offset = L_TOTAL - D
    ids = torch.full((4, L_TOTAL), 5, dtype=torch.long)
    ids[:, D:] = PAD_ID

    cfg_g = GDPOConfig(n_quadrature=3, quadrature_grid="uniform", inner_mc=1)
    idx_grid, t_floats, weights = _build_grid(cfg_g, D, torch.device("cpu"))

    corruptions = _build_shared_corruptions(
        ids=ids, idx_grid=idx_grid, t_floats=t_floats, weights=weights,
        inner_mc=1, device=torch.device("cpu"),
        diffusion_budget=D, time_offset=offset,
    )

    assert len(corruptions) == idx_grid.numel()
    for n, c in enumerate(corruptions):
        reveal_count = int(idx_grid[n])
        # Model sees the true revealed count: content revealed + PAD tail.
        assert (c["idx"] == reveal_count + offset).all()
        # But the masking still reveals exactly `reveal_count` content positions.
        revealed = (c["x_t"][:, :D] != MASK_ID).sum(dim=1)
        assert (revealed == reveal_count).all()
        # The tail is never masked.
        assert (c["x_t"][:, D:] == PAD_ID).all()


def test_corruptions_default_offset_is_unshifted():
    ids = torch.full((2, L_TOTAL), 5, dtype=torch.long)
    cfg_g = GDPOConfig(n_quadrature=2, quadrature_grid="uniform", inner_mc=1)
    idx_grid, t_floats, weights = _build_grid(cfg_g, L_TOTAL, torch.device("cpu"))

    corruptions = _build_shared_corruptions(
        ids=ids, idx_grid=idx_grid, t_floats=t_floats, weights=weights,
        inner_mc=1, device=torch.device("cpu"),
    )
    for n, c in enumerate(corruptions):
        assert (c["idx"] == int(idx_grid[n])).all()
