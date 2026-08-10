"""Tests for the method-agnostic base: fit/score contract and registration.

Exercised through a stub method that carries none of the Gaussian extras (no
band, no norm guard, no minimum size), which is the point -- the base must be
usable by a method that only knows how to fit and score.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from biom3.geometry import io
from biom3.geometry.base import ManifoldModel, as_matrix, available_methods, get_method


@dataclass(frozen=True, kw_only=True, repr=False, eq=False)
class _CentroidManifold(ManifoldModel):
    """Bare-minimum method: Euclidean distance to the reference centroid."""

    centroid: np.ndarray

    METHOD = "_test_centroid"

    @classmethod
    def _fit_state(cls, reference):
        return {"centroid": reference.mean(axis=0)}

    def _score(self, queries):
        return np.linalg.norm(queries - self.centroid, axis=1)

    def _arrays(self):
        return {"centroid": self.centroid}

    @classmethod
    def _from_arrays(cls, arrays):
        return {"centroid": np.asarray(arrays["centroid"], dtype=np.float64)}


def _cloud(n, d, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


def test_minimal_method_fits_and_scores():
    reference = _cloud(20, 5, seed=0)
    model = _CentroidManifold.fit(reference)

    assert model.n_dim == 5
    np.testing.assert_allclose(model.centroid, reference.mean(axis=0))

    scores = model.score(reference[:3])
    np.testing.assert_allclose(
        scores, np.linalg.norm(reference[:3] - reference.mean(axis=0), axis=1)
    )


def test_minimal_method_needs_no_band_or_norm_guard():
    model = _CentroidManifold.fit(_cloud(20, 5, seed=1))

    assert not hasattr(model, "band")
    assert not hasattr(model, "ref_mean_norm")
    # Scoring wildly rescaled queries is silent: no norm guard was opted into.
    model.score(_cloud(4, 5, seed=2) * 1000)


def test_method_is_registered_and_round_trips_through_io(tmp_path):
    assert "_test_centroid" in available_methods()
    assert get_method("_test_centroid") is _CentroidManifold

    model = _CentroidManifold.fit(_cloud(20, 5, seed=3))
    path = str(tmp_path / "centroid.npz")
    io.save_manifold(path, model)
    loaded = io.load_manifold(path)

    assert type(loaded) is _CentroidManifold
    np.testing.assert_array_equal(loaded.centroid, model.centroid)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown manifold method"):
        get_method("no_such_method")


def test_duplicate_method_registration_is_rejected():
    with pytest.raises(ValueError, match="already registered"):
        @dataclass(frozen=True, kw_only=True, repr=False, eq=False)
        class _Clash(ManifoldModel):
            METHOD = "_test_centroid"

            @classmethod
            def _fit_state(cls, reference):
                return {}

            def _score(self, queries):
                return np.zeros(len(queries))

            def _arrays(self):
                return {}

            @classmethod
            def _from_arrays(cls, arrays):
                return {}


def test_base_validates_dimension_and_input_shape():
    model = _CentroidManifold.fit(_cloud(20, 5, seed=4))

    with pytest.raises(ValueError, match="dimension"):
        model.score(_cloud(3, 4, seed=5))
    with pytest.raises(ValueError, match="2-D"):
        model.score(np.zeros(5))
    with pytest.raises(ValueError, match="NaN"):
        model.score(np.full((3, 5), np.nan))
    with pytest.raises(ValueError, match="empty"):
        _CentroidManifold.fit(np.zeros((0, 5)))


def test_abstract_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ManifoldModel(n_dim=4)


@pytest.mark.parametrize("bad", [np.zeros(4), np.zeros((0, 4)), np.full((2, 2), np.inf)])
def test_as_matrix_rejects_malformed_input(bad):
    with pytest.raises(ValueError):
        as_matrix(bad, "reference")
