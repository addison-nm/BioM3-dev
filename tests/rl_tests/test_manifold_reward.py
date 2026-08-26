"""CPU tests for ManifoldReward and the PenCL z_p featurizer wiring.

No weights, no GPU: the manifold is fitted on synthetic vectors and the
featurizer is a stub, so these exercise polarity, transforms, the
unbound-featurizer guard, the dim/space check, and the bind protocol.
The parts that need real weights (does the embedder reproduce the fit's
band) are covered by _misc/edist_scripts/check_zp_embedder.py.
"""

import numpy as np
import pytest

from biom3.geometry import fit_manifold
from biom3.rl.rewards import CompositeReward, ManifoldReward, build_reward
from biom3.rl.rewards.manifold import TRANSFORMS, bind_pencl_rewards
from biom3.rl.rewards.stub import StubReward

DIM = 16


@pytest.fixture
def manifold():
    rng = np.random.default_rng(0)
    return fit_manifold(rng.normal(size=(200, DIM)), label="test")


class _StubFeaturizer:
    """Maps sequence length to a radius, so d_E rises with length."""

    dim = DIM

    def __init__(self, scale=1.0):
        self.scale = scale
        self.calls = 0

    def __call__(self, sequences):
        self.calls += 1
        out = np.zeros((len(sequences), DIM))
        for i, s in enumerate(sequences):
            if s:
                out[i, 0] = self.scale * len(s)
        return out


def test_lower_distance_gets_higher_reward(manifold):
    r = ManifoldReward(manifold, _StubFeaturizer(), transform="neg")
    rewards = r(["A", "AAAA", "AAAAAAAA"])
    assert rewards[0] > rewards[1] > rewards[2], "reward must invert d_E"


def test_scores_are_raw_distances(manifold):
    r = ManifoldReward(manifold, _StubFeaturizer(), transform="neg")
    d = r.scores(["A", "AAAA"])
    assert (d > 0).all() and d[0] < d[1]


@pytest.mark.parametrize("transform", sorted(TRANSFORMS))
def test_every_transform_is_monotone_decreasing_in_distance(manifold, transform):
    r = ManifoldReward(manifold, _StubFeaturizer(), transform=transform, scale=1.0)
    rewards = r(["A" * n for n in (1, 3, 6, 10)])
    assert all(a >= b for a, b in zip(rewards, rewards[1:])), (
        f"{transform} is not monotone in d_E: {rewards}"
    )


def test_exp_transform_is_bounded(manifold):
    r = ManifoldReward(manifold, _StubFeaturizer(), transform="exp")
    rewards = r(["A" * n for n in (1, 5, 50)])
    assert all(0.0 < x <= 1.0 for x in rewards)


def test_clipped_transform_bounds_the_tail(manifold):
    r = ManifoldReward(manifold, _StubFeaturizer(), transform="clipped", scale=1.0)
    rewards = r(["A" * n for n in (50, 500)])
    assert rewards[0] == rewards[1] == pytest.approx(-manifold.band["p95"])


def test_band_transform_is_zero_at_p95(manifold):
    r = ManifoldReward(manifold, _StubFeaturizer(), transform="band")
    d = np.array([manifold.band["p95"]])
    assert TRANSFORMS["band"](d, manifold.band, 1.0)[0] == pytest.approx(0.0)


def test_unembeddable_sequence_gets_worst_distance(manifold):
    r = ManifoldReward(manifold, _StubFeaturizer(), transform="neg")
    rewards = r(["AAAA", ""])
    assert rewards[1] < rewards[0], "empty sequence must not out-score a real one"


def test_last_components_exposes_distance_and_band_position(manifold):
    r = ManifoldReward(manifold, _StubFeaturizer(), transform="band")
    r(["A", "AAAAAAAAAA"])
    comps = r.last_components()
    assert set(comps) == {"d_E", "band_position"}
    assert len(comps["d_E"]) == 2
    assert all(v in (-1.0, 0.0, 1.0) for v in comps["band_position"])


def test_rejects_featurizer_from_a_different_space(manifold):
    class Wrong(_StubFeaturizer):
        dim = DIM + 1

    with pytest.raises(ValueError, match="not the same space"):
        ManifoldReward(manifold, Wrong())


def test_rejects_unknown_transform(manifold):
    with pytest.raises(ValueError, match="transform must be one of"):
        ManifoldReward(manifold, _StubFeaturizer(), transform="nope")


def test_unbound_featurizer_raises_clearly(manifold):
    r = ManifoldReward(manifold)
    with pytest.raises(RuntimeError, match="no featurizer"):
        r(["AAAA"])


def test_bind_does_not_override_an_explicit_featurizer(manifold):
    feat = _StubFeaturizer()
    r = ManifoldReward(manifold, feat)
    bind_pencl_rewards(r, object())      # would explode if it tried to build one
    assert r.featurizer is feat


def test_bind_reaches_into_composite_components(manifold):
    seen = []

    class Recorder(ManifoldReward):
        def bind_pencl(self, pencl):
            seen.append(pencl)

    comp = CompositeReward({
        "manifold": (Recorder(manifold, _StubFeaturizer()), 1.0),
        "stub": (StubReward(), 0.5),
    })
    sentinel = object()
    assert bind_pencl_rewards(comp, sentinel) == 1
    assert seen == [sentinel]


def test_validate_against_band_passes_on_the_fit_itself():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=(200, DIM))
    m = fit_manifold(reference, label="test")

    class Replay:
        dim = DIM

        def __call__(self, sequences):
            return reference[: len(sequences)]

    r = ManifoldReward(m, Replay())
    result = r.validate_against_band([""] * 200, stored_vectors=reference)
    assert result["passed"], result
    assert result["max_relative_vector_error"] == pytest.approx(0.0)


def test_validate_against_band_flags_a_subset_as_incomparable():
    """A band is a population statistic; a subset deviating means nothing.

    Regression for a real false alarm: 24 of the 911 SHO1 naturals shifted
    the median 8.79 -> 10.00 while the per-vector error was 2e-06.
    """
    rng = np.random.default_rng(3)
    reference = rng.normal(size=(200, DIM))
    m = fit_manifold(reference, label="test")

    class Replay:
        dim = DIM

        def __call__(self, sequences):
            return reference[: len(sequences)]

    r = ManifoldReward(m, Replay())
    full = r.validate_against_band([""] * 200)
    assert full["band_comparable"] and full["n_given"] == full["n_reference"]

    subset = r.validate_against_band([""] * 24)
    assert not subset["band_comparable"]
    assert subset["n_given"] == 24 and subset["n_reference"] == 200


def test_validate_against_band_fails_on_a_perturbed_embedder():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=(200, DIM))
    m = fit_manifold(reference, label="test")

    class Shifted:
        dim = DIM

        def __call__(self, sequences):
            return reference[: len(sequences)] + 5.0

    r = ManifoldReward(m, Shifted())
    assert not r.validate_against_band([""] * 200)["passed"]


def test_build_reward_constructs_manifold_from_path(tmp_path, manifold):
    from biom3.geometry.io import save_manifold

    path = tmp_path / "m.npz"
    save_manifold(str(path), manifold)
    r = build_reward("manifold", device="cpu", manifold=str(path), transform="neg")
    assert isinstance(r, ManifoldReward)
    assert r.manifold.n_dim == DIM
    assert r.featurizer is None, "should construct unbound; the trainer binds it"
