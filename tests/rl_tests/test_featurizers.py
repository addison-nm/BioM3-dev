"""Unit tests for biom3.rl.featurizers."""

import os

import numpy as np
import pytest

from biom3.rl.featurizers import (
    CANONICAL_AA,
    ESM2MeanPoolFeaturizer,
    OneHotFeaturizer,
    build_featurizer,
)


class TestOneHotFeaturizer:
    def test_shape(self):
        f = OneHotFeaturizer(max_length=10)
        out = f(["ACDE", "MMMMM"])
        assert out.shape == (2, 10 * 20)
        assert out.dtype == np.float32

    def test_one_hot_values_at_correct_positions(self):
        f = OneHotFeaturizer(max_length=4)
        out = f(["ACDE"])
        # Reshape to (1, 4, 20) for inspection
        view = out.reshape(1, 4, 20)
        for j, c in enumerate("ACDE"):
            i = CANONICAL_AA.index(c)
            assert view[0, j, i] == 1.0
            # Every other position should be 0
            assert view[0, j].sum() == 1.0

    def test_padding_is_zero(self):
        f = OneHotFeaturizer(max_length=10)
        out = f(["AC"])
        view = out.reshape(1, 10, 20)
        # Positions 2..9 should be all zeros (padded)
        assert view[0, 2:].sum() == 0

    def test_truncation(self):
        f = OneHotFeaturizer(max_length=3)
        out = f(["ACDEFG"])
        view = out.reshape(1, 3, 20)
        # Only first 3 positions get encoded
        for j, c in enumerate("ACD"):
            assert view[0, j, CANONICAL_AA.index(c)] == 1.0

    def test_drops_non_canonical_chars(self):
        f = OneHotFeaturizer(max_length=5)
        # "<PAD>" and "-" aren't canonical AA — but A, P, D individually are.
        # The cleaner is char-by-char (matches the contract documented in
        # rewards.py's _clean_aa). After cleaning, "AC-DE" -> "ACDE".
        out = f(["AC-DE"])
        view = out.reshape(1, 5, 20)
        assert view[0, 0, CANONICAL_AA.index("A")] == 1.0
        assert view[0, 1, CANONICAL_AA.index("C")] == 1.0
        assert view[0, 2, CANONICAL_AA.index("D")] == 1.0
        assert view[0, 3, CANONICAL_AA.index("E")] == 1.0
        # 4th position empty (only 4 valid chars)
        assert view[0, 4].sum() == 0

    def test_empty_sequence(self):
        f = OneHotFeaturizer(max_length=5)
        out = f([""])
        assert out.shape == (1, 5 * 20)
        assert out.sum() == 0

    def test_invalid_max_length_rejected(self):
        with pytest.raises(ValueError):
            OneHotFeaturizer(max_length=0)
        with pytest.raises(ValueError):
            OneHotFeaturizer(max_length=-3)

    def test_dim_attribute(self):
        f = OneHotFeaturizer(max_length=7)
        assert f.dim == 7 * 20

    def test_dispatcher(self):
        f = build_featurizer("onehot", max_length=8)
        assert isinstance(f, OneHotFeaturizer)
        assert f.max_length == 8


def test_build_featurizer_unknown():
    with pytest.raises(ValueError):
        build_featurizer("nonsense")


# ─────────────────────────────────────────────────────────────────────────────
# ESM-2 featurizer — weight-gated, so skips when fair-esm or weights aren't
# present. We only test shape + finiteness, not predictive quality.
# ─────────────────────────────────────────────────────────────────────────────


_ESM_WEIGHTS = "./weights/LLMs/esm2_t33_650M_UR50D.pt"


@pytest.mark.slow
@pytest.mark.use_gpu
@pytest.mark.skipif(
    not os.path.exists(_ESM_WEIGHTS),
    reason=f"ESM-2 weights not present at {_ESM_WEIGHTS}",
)
def test_esm2_featurizer_shape_and_finite():
    pytest.importorskip("esm")
    f = ESM2MeanPoolFeaturizer(model_path=_ESM_WEIGHTS, max_length=64, batch_size=2, device="cpu")
    seqs = ["ACDEFGHIKLMNPQRSTVWY", "AAAAAAAAAA", ""]
    out = f(seqs)
    assert out.shape == (3, f.dim)
    assert np.isfinite(out).all()
    # Empty sequence should be all-zeros (we explicitly skip those)
    assert out[2].sum() == 0
