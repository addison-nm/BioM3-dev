"""Unit tests for the pre-unmask feature in Stage 3 ProteoScribe sampling."""

import argparse
import json

import pytest
import torch

from biom3.Stage3.run_ProteoScribe_sample import (
    _build_initial_mask_state,
    _resolve_fill_token_id,
    load_pre_unmask_config,
)


TOKENS = [
    '-', '<START>', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '<END>', '<PAD>',
    'X', 'U', 'Z', 'B', 'O',
]
PAD_ID = TOKENS.index('<PAD>')


def _make_args(**kwargs):
    defaults = {"diffusion_steps": 1024}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_default_is_all_masked():
    """With pre_unmask disabled, the init tensor is all zeros of shape
    (batch, diffusion_steps)."""
    args = _make_args(diffusion_steps=1024, pre_unmask=False)
    init, sampling_path = _build_initial_mask_state(args, batch_size=3, tokens=TOKENS)
    assert init.shape == (3, 1024)
    assert torch.all(init == 0)
    assert sampling_path.shape == (3, 1024)
    for row in sampling_path:
        assert set(row.tolist()) == set(range(1024))


def test_pre_unmask_last_k_pad():
    """With pre_unmask enabled: [0:D) == 0 (mask), [D:L) == PAD id."""
    args = _make_args(
        diffusion_steps=16,
        sequence_length=1024,
        pre_unmask=True,
        pre_unmask_fill_with="PAD",
    )
    init, sampling_path = _build_initial_mask_state(args, batch_size=2, tokens=TOKENS)
    assert init.shape == (2, 1024)
    assert torch.all(init[:, :16] == 0)
    assert torch.all(init[:, 16:] == PAD_ID)
    assert sampling_path.shape == (2, 16)
    for row in sampling_path:
        assert set(row.tolist()) == set(range(16))


def test_pre_unmask_budget_exceeds_sequence_length():
    """D > sequence_length must raise."""
    args = _make_args(
        diffusion_steps=2048,
        sequence_length=1024,
        pre_unmask=True,
        pre_unmask_fill_with="PAD",
    )
    with pytest.raises(ValueError, match="must be <= sequence_length"):
        _build_initial_mask_state(args, batch_size=1, tokens=TOKENS)


def test_resolve_fill_token_id_aliases():
    for alias in ("PAD", "pad", "<PAD>"):
        assert _resolve_fill_token_id(alias, TOKENS) == PAD_ID


def test_resolve_fill_token_id_unknown():
    with pytest.raises(ValueError, match="not supported"):
        _resolve_fill_token_id("mask", TOKENS)


def test_load_pre_unmask_config_valid(tmp_path):
    cfg = {"strategy": "last_k", "fill_with": "PAD", "diffusion_budget": 16}
    path = tmp_path / "pre_unmask.json"
    path.write_text(json.dumps(cfg))
    loaded = load_pre_unmask_config(str(path))
    assert loaded == cfg


def test_load_pre_unmask_config_missing_key(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"strategy": "last_k", "fill_with": "PAD"}))
    with pytest.raises(ValueError, match="missing keys"):
        load_pre_unmask_config(str(path))


def test_load_pre_unmask_config_unknown_strategy(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "strategy": "first_k", "fill_with": "PAD", "diffusion_budget": 16,
    }))
    with pytest.raises(ValueError, match="not supported"):
        load_pre_unmask_config(str(path))


def test_load_pre_unmask_config_unknown_key(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "strategy": "last_k", "fill_with": "PAD", "diffusion_budget": 16,
        "extra": 1,
    }))
    with pytest.raises(ValueError, match="unknown keys"):
        load_pre_unmask_config(str(path))


def test_load_pre_unmask_config_bad_budget(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "strategy": "last_k", "fill_with": "PAD", "diffusion_budget": 0,
    }))
    with pytest.raises(ValueError, match="positive int"):
        load_pre_unmask_config(str(path))


def test_load_pre_unmask_config_requires_path():
    with pytest.raises(ValueError, match="--pre_unmask_config"):
        load_pre_unmask_config(None)
