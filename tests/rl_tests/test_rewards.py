"""Unit tests for biom3.rl.rewards extensions: TsvLookupReward,
CompositeReward, and dispatcher coverage.
"""

import math

import numpy as np
import pytest
import torch

from biom3.rl.rewards import (
    AAFractionReward,
    CompositeReward,
    StubReward,
    SurrogateReward,
    TsvLookupReward,
    build_reward,
)


def _write_tsv(path, rows, delimiter="\t", header=("sequence", "value")):
    lines = [delimiter.join(header)]
    for row in rows:
        lines.append(delimiter.join(str(c) for c in row))
    path.write_text("\n".join(lines) + "\n")


class TestTsvLookupReward:
    def test_hit_returns_value(self, tmp_path):
        p = tmp_path / "data.tsv"
        _write_tsv(p, [("ACDE", 1.5), ("KLMN", -0.7)])
        r = TsvLookupReward(str(p))
        assert r(["ACDE", "KLMN"]) == [1.5, -0.7]

    def test_miss_penalty_default_zero(self, tmp_path):
        p = tmp_path / "data.tsv"
        _write_tsv(p, [("ACDE", 1.5)])
        r = TsvLookupReward(str(p))
        out = r(["ACDE", "GHIK"])
        assert out == [1.5, 0.0]

    def test_miss_penalty_custom(self, tmp_path):
        p = tmp_path / "data.tsv"
        _write_tsv(p, [("ACDE", 1.5)])
        r = TsvLookupReward(str(p), miss_value=-99.0)
        assert r(["GHIK"]) == [-99.0]

    def test_miss_skip_returns_nan(self, tmp_path):
        p = tmp_path / "data.tsv"
        _write_tsv(p, [("ACDE", 1.5)])
        r = TsvLookupReward(str(p), miss_strategy="skip")
        out = r(["ACDE", "GHIK"])
        assert out[0] == 1.5
        assert math.isnan(out[1])

    def test_clean_strips_residual_non_aa_chars(self, tmp_path):
        p = tmp_path / "data.tsv"
        _write_tsv(p, [("ACDE", 1.5)])
        r = TsvLookupReward(str(p))
        # decode_tokens (in grpo.py) already strips full special-token
        # strings like <PAD>, <END>; clean=True is only the secondary
        # defense against single-char residuals: gap chars, whitespace,
        # lowercase, etc.
        assert r(["ACDE-"]) == [1.5]
        assert r(["ACDE "]) == [1.5]
        assert r(["acde"]) == [0.0]   # lowercase isn't an AA; this becomes a miss

    def test_csv_with_custom_delimiter_and_columns(self, tmp_path):
        p = tmp_path / "data.csv"
        _write_tsv(p, [("ACDE", 9.0)], delimiter=",", header=("aa", "ddG"))
        r = TsvLookupReward(
            str(p), key_column="aa", value_column="ddG", delimiter=","
        )
        assert r(["ACDE"]) == [9.0]

    def test_invalid_miss_strategy_rejected(self, tmp_path):
        p = tmp_path / "data.tsv"
        _write_tsv(p, [("A", 1.0)])
        with pytest.raises(ValueError):
            TsvLookupReward(str(p), miss_strategy="bogus")

    def test_missing_key_column_raises(self, tmp_path):
        p = tmp_path / "data.tsv"
        _write_tsv(p, [("ACDE", 1.0)])
        r = TsvLookupReward(str(p), key_column="nope")
        with pytest.raises(KeyError):
            r(["ACDE"])

    def test_dispatcher_builds_tsv_lookup(self, tmp_path):
        p = tmp_path / "data.tsv"
        _write_tsv(p, [("ACDE", 1.0)])
        r = build_reward("tsv_lookup", device=torch.device("cpu"), path=str(p))
        assert isinstance(r, TsvLookupReward)


class _FixedReward:
    """Test helper: returns a preconfigured list, length-matched."""

    def __init__(self, values):
        self.values = list(values)

    def __call__(self, completions, **kwargs):
        return [self.values[i % len(self.values)] for i in range(len(completions))]


class TestCompositeReward:
    def test_weighted_sum(self):
        r = CompositeReward(
            {
                "a": (_FixedReward([2.0, 4.0]), 0.5),
                "b": (_FixedReward([10.0, 20.0]), 1.0),
            },
            reduction="weighted_sum",
        )
        out = r(["X", "Y"])
        assert out == [0.5 * 2.0 + 1.0 * 10.0, 0.5 * 4.0 + 1.0 * 20.0]

    def test_product(self):
        r = CompositeReward(
            {
                "a": (_FixedReward([2.0]), 1.0),
                "b": (_FixedReward([3.0]), 2.0),
            },
            reduction="product",
        )
        # 2.0 ** 1.0 * 3.0 ** 2.0 = 18.0
        assert r(["X"]) == [18.0]

    def test_last_components_breakdown(self):
        r = CompositeReward(
            {"a": (_FixedReward([1.0, 2.0]), 1.0), "b": (_FixedReward([10.0, 20.0]), 1.0)},
        )
        r(["X", "Y"])
        comps = r.last_components()
        assert comps == {"a": [1.0, 2.0], "b": [10.0, 20.0]}

    def test_empty_components_rejected(self):
        with pytest.raises(ValueError):
            CompositeReward({})

    def test_unknown_reduction_rejected(self):
        with pytest.raises(ValueError):
            CompositeReward({"a": (StubReward(), 1.0)}, reduction="median")

    def test_composes_real_rewards(self):
        # Sanity: a real Reward (StubReward) plugs in unchanged.
        r = CompositeReward(
            {"stub": (StubReward(target_length=5), 1.0)}, reduction="weighted_sum"
        )
        out = r(["AAAAA"])
        assert isinstance(out, list) and len(out) == 1
        assert 0.0 <= out[0] <= 100.0


class TestAAFractionReward:
    def test_peak_at_target(self):
        r = AAFractionReward(target_aa="A", target_fraction=0.4, scale=100.0)
        # 4 A's in 10 chars = 0.4 → peak
        assert r(["AAAACCCCCC"]) == [100.0]

    def test_falloff_linear_then_clamps(self):
        r = AAFractionReward(target_aa="A", target_fraction=0.4, scale=100.0)
        # All A: frac=1.0, |1.0 - 0.4| = 0.6, score = 100 * (1 - 1.2) = clamped to 0
        assert r(["AAAAAAAAAA"]) == pytest.approx([0.0])
        # No A: frac=0, |0 - 0.4| = 0.4, score = 100 * (1 - 0.8) = 20
        assert r(["CCCCCCCCCC"]) == pytest.approx([20.0])

    def test_empty_returns_zero(self):
        r = AAFractionReward()
        assert r([""]) == [0.0]
        assert r(["123!@#"]) == [0.0]   # no canonical AAs after cleaning

    def test_dispatcher_builds_aa_fraction(self):
        r = build_reward(
            "aa_fraction", device=torch.device("cpu"),
            target_aa="L", target_fraction=0.3,
        )
        assert isinstance(r, AAFractionReward)
        assert r.target_aa == "L"
        assert r.target_fraction == 0.3

    def test_invalid_target_aa_rejected(self):
        with pytest.raises(ValueError):
            AAFractionReward(target_aa="AA")
        with pytest.raises(ValueError):
            AAFractionReward(target_aa="x")  # lowercase isn't in VALID_AA

    def test_invalid_target_fraction_rejected(self):
        with pytest.raises(ValueError):
            AAFractionReward(target_fraction=1.5)
        with pytest.raises(ValueError):
            AAFractionReward(target_fraction=-0.1)


class _StubPredictor:
    """Test helper that mimics the sklearn .predict() interface."""

    def __init__(self, by_index=None, constant=None):
        self.by_index = by_index
        self.constant = constant

    def predict(self, features):
        n = features.shape[0]
        if self.by_index is not None:
            return np.arange(n, dtype=np.float64) * self.by_index
        return np.full(n, self.constant, dtype=np.float64)


class _IdentityFeaturizer:
    """Returns shape (N, 1) — content doesn't matter for SurrogateReward tests."""

    def __call__(self, sequences):
        return np.zeros((len(sequences), 1), dtype=np.float32)


class TestSurrogateReward:
    def test_returns_list_of_floats(self):
        r = SurrogateReward(predictor=_StubPredictor(constant=3.5), featurizer=_IdentityFeaturizer())
        out = r(["AAAA", "CCCC"])
        assert out == [3.5, 3.5]

    def test_preserves_order(self):
        r = SurrogateReward(predictor=_StubPredictor(by_index=2.0), featurizer=_IdentityFeaturizer())
        out = r(["A", "B", "C"])
        assert out == [0.0, 2.0, 4.0]

    def test_predictor_must_have_predict(self):
        class _Bogus:
            pass
        with pytest.raises(TypeError):
            SurrogateReward(predictor=_Bogus(), featurizer=_IdentityFeaturizer())
