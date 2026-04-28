"""Unit tests for biom3.rl.rewards extensions: TsvLookupReward,
CompositeReward, and dispatcher coverage.
"""

import math

import pytest
import torch

from biom3.rl.rewards import (
    CompositeReward,
    StubReward,
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
