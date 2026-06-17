"""Unit tests for the pre-unmask feature in Stage 3 ProteoScribe sampling."""

import argparse
import json

import pytest
import torch

import biom3.Stage3.sampling_analysis as Stage3_sample_tools
from biom3.Stage3.run_ProteoScribe_sample import (
    _build_initial_mask_state,
    _pre_revealed_offset,
    _resolve_fill_token_id,
    load_pre_unmask_config,
)


TOKENS = [
    '-', '<START>', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '<END>', '<PAD>',
    'X', 'U', 'Z', 'B', 'O',
]
PAD_ID = TOKENS.index('<PAD>')
FAVORED_ID = TOKENS.index('A')  # 2 — a non-MASK, non-PAD token the stub emits


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
    """With pre_unmask enabled: [0:D) == 0 (mask), [D:L) == PAD id, and the
    sampling-path values are offset by (L - D) so the model time index counts
    the PAD tail as already revealed."""
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
    offset = 1024 - 16
    for row in sampling_path:
        assert set(row.tolist()) == set(range(offset, offset + 16))


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


# ---------------------------------------------------------------------------
#  Time-index offset in the sampler (pre_unmask / in-painting regression coverage)
# ---------------------------------------------------------------------------

class _StubModel(torch.nn.Module):
    """Records the time index it is conditioned on at each step and returns
    constant logits favouring a single non-special token.

    Lets the structural bookkeeping (time index, which positions get written)
    be tested without weights — token quality is irrelevant here.
    """

    def __init__(self, num_classes, favored=FAVORED_ID):
        super().__init__()
        self.num_classes = num_classes
        self.favored = favored
        self.seen_t = []

    def forward(self, x, t, y_c):
        self.seen_t.append(t.detach().clone())
        batch, seq_len = x.shape
        logits = torch.zeros(batch, self.num_classes, seq_len)
        logits[:, self.favored, :] = 10.0
        return logits


def _sampler_args(diffusion_steps, sequence_length, pre_unmask=True):
    return argparse.Namespace(
        device="cpu",
        diffusion_steps=diffusion_steps,
        sequence_length=sequence_length,
        num_classes=len(TOKENS),
        token_strategy="argmax",
        pre_unmask=pre_unmask,
        pre_unmask_fill_with="PAD",
        pad_token_id=PAD_ID,
        _world_size=1,
        _silent_tqdm=True,
    )


def _run_random_sampler(args, batch_size, extract_time=None):
    """Mirror _process_batch for the random-path sampler with a stub model."""
    init, sampling_path = _build_initial_mask_state(args, batch_size, TOKENS)
    if extract_time is None:
        offset = _pre_revealed_offset(args, args.diffusion_steps)
        extract_time = torch.full((batch_size,), offset, dtype=torch.long)
    model = _StubModel(num_classes=len(TOKENS))
    cond = torch.randn(batch_size, 4)
    mask_list, time_list, _ = Stage3_sample_tools.batch_generate_denoised_sampled(
        args=args,
        model=model,
        extract_digit_samples=init,
        extract_time=extract_time,
        extract_digit_label=cond,
        sampling_path=sampling_path,
    )
    return model, init, mask_list, time_list


def test_time_index_spans_full_revealed_count():
    """Pre_unmask sampler: the model is told t == (L - D) on the first step and
    t == L - 1 on the last step — the true revealed count, not [0, D)."""
    D, L, batch = 16, 1024, 2
    args = _sampler_args(D, L)
    model, _, _, time_list = _run_random_sampler(args, batch)

    offset = L - D
    assert torch.all(model.seen_t[0] == offset)
    assert torch.all(model.seen_t[-1] == L - 1)
    # The stored time index matches what the model saw.
    assert time_list[0].flatten().tolist() == [offset] * batch
    assert time_list[-1].flatten().tolist() == [L - 1] * batch
    # D steps total.
    assert len(model.seen_t) == D


def test_only_budget_positions_written_tail_preserved():
    """Pre_unmask sampler writes exactly the first D positions; the PAD tail is
    untouched and every budget position is filled once."""
    D, L, batch = 16, 1024, 2
    args = _sampler_args(D, L)
    _, _, mask_list, _ = _run_random_sampler(args, batch)

    final = mask_list[-1]  # [batch, 1, L]
    for b in range(batch):
        row = final[b, 0]
        assert (row[:D] == FAVORED_ID).all(), "all budget positions must be filled"
        assert (row[D:] == PAD_ID).all(), "PAD tail must be preserved"


def test_non_pre_unmask_time_index_unchanged():
    """With pre_unmask disabled the offset is 0 and t spans [0, L)."""
    L, batch = 32, 2
    args = _sampler_args(L, L, pre_unmask=False)
    model, _, _, _ = _run_random_sampler(args, batch)

    assert torch.all(model.seen_t[0] == 0)
    assert torch.all(model.seen_t[-1] == L - 1)
    assert len(model.seen_t) == L


def test_zeros_time_with_offset_path_is_degenerate():
    """Regression guard: the old behaviour (extract_time == zeros while the
    sampling path is offset for pre_unmask with D < L) collapses every step
    onto column 0, leaving the rest of the budget masked. The fix (matching
    offset time) must fill all D positions instead."""
    D, L, batch = 16, 1024, 2
    args = _sampler_args(D, L)

    # Buggy: zeros time index never matches the offset path values.
    _, _, buggy_mask_list, _ = _run_random_sampler(
        args, batch, extract_time=torch.zeros(batch, dtype=torch.long),
    )
    buggy_final = buggy_mask_list[-1]
    for b in range(batch):
        row = buggy_final[b, 0]
        # Only column 0 ever gets written; positions [1, D) stay masked (0).
        assert (row[1:D] == 0).all()

    # Fixed: matching offset time fills the whole budget.
    _, _, fixed_mask_list, _ = _run_random_sampler(args, batch)
    for b in range(batch):
        assert (fixed_mask_list[-1][b, 0][:D] == FAVORED_ID).all()


def test_confidence_path_time_index_offset():
    """The confidence sampler also receives the offset time index and writes
    only the budget positions."""
    D, L, batch = 16, 1024, 2
    args = _sampler_args(D, L)
    init, _ = _build_initial_mask_state(args, batch, TOKENS)
    offset = _pre_revealed_offset(args, args.diffusion_steps)
    extract_time = torch.full((batch,), offset, dtype=torch.long)
    model = _StubModel(num_classes=len(TOKENS))
    cond = torch.randn(batch, 4)

    mask_list, _, _ = Stage3_sample_tools.batch_generate_denoised_sampled_confidence(
        args=args,
        model=model,
        extract_digit_samples=init,
        extract_time=extract_time,
        extract_digit_label=cond,
    )

    assert torch.all(model.seen_t[0] == offset)
    assert torch.all(model.seen_t[-1] == L - 1)
    final = mask_list[-1]
    for b in range(batch):
        row = final[b, 0]
        assert (row[:D] == FAVORED_ID).all()
        assert (row[D:] == PAD_ID).all()
