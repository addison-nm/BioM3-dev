"""Tests for manifold persistence and embedding-matrix loading."""

import os
from dataclasses import dataclass

import numpy as np
import pytest

from biom3.geometry import io
from biom3.geometry.base import ManifoldModel
from biom3.geometry.manifold import fit_manifold, manifold_score


def _cloud(n, d, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)) @ (rng.normal(size=(d, d)) / np.sqrt(d))


def test_manifold_round_trip(tmp_path):
    manifold = fit_manifold(_cloud(50, 16, seed=0), label="ref v1")
    path = os.path.join(tmp_path, "manifold.npz")
    io.save_manifold(path, manifold)
    loaded = io.load_manifold(path)

    assert type(loaded) is type(manifold)
    assert loaded.n_dim == manifold.n_dim
    np.testing.assert_array_equal(loaded.centroid, manifold.centroid)
    np.testing.assert_array_equal(loaded.precision, manifold.precision)
    np.testing.assert_array_equal(loaded.covariance, manifold.covariance)
    assert loaded.n_reference == manifold.n_reference
    assert loaded.band == manifold.band
    assert loaded.ref_mean_norm == manifold.ref_mean_norm
    assert loaded.shrinkage == manifold.shrinkage
    assert loaded.label == "ref v1"


def test_loaded_manifold_scores_identically(tmp_path):
    reference = _cloud(50, 16, seed=1)
    queries = _cloud(10, 16, seed=2)
    manifold = fit_manifold(reference)
    path = os.path.join(tmp_path, "manifold.npz")
    io.save_manifold(path, manifold)

    np.testing.assert_array_equal(
        manifold_score(queries, io.load_manifold(path), check_norm=False),
        manifold_score(queries, manifold, check_norm=False),
    )


def _rewrite_field(path, **replacements):
    with np.load(path) as data:
        fields = {k: data[k] for k in data.files}
    fields.update(replacements)
    np.savez_compressed(path, **fields)


def test_load_rejects_unknown_schema_version(tmp_path):
    path = os.path.join(tmp_path, "manifold.npz")
    io.save_manifold(path, fit_manifold(_cloud(20, 8, seed=3)))
    _rewrite_field(path, schema_version=np.int64(io.SCHEMA_VERSION + 1))

    with pytest.raises(ValueError, match="schema version"):
        io.load_manifold(path)


def test_load_rejects_unknown_method(tmp_path):
    path = os.path.join(tmp_path, "manifold.npz")
    io.save_manifold(path, fit_manifold(_cloud(20, 8, seed=4)))
    _rewrite_field(path, method=np.array("some_future_method"))

    with pytest.raises(ValueError, match="unknown manifold method"):
        io.load_manifold(path)


@dataclass(frozen=True, kw_only=True, repr=False, eq=False)
class _ShadowingManifold(ManifoldModel):
    """A method that persists a field name the format reserves for itself."""

    METHOD = "_test_shadowing"

    @classmethod
    def _fit_state(cls, reference):
        return {}

    def _score(self, queries):
        return np.zeros(len(queries))

    def _arrays(self):
        return {"n_dim": np.int64(3)}

    @classmethod
    def _from_arrays(cls, arrays):
        return {}


def test_save_rejects_a_method_that_shadows_a_reserved_name(tmp_path):
    with pytest.raises(ValueError, match="reserved array names"):
        io.save_manifold(
            os.path.join(tmp_path, "manifold.npz"), _ShadowingManifold(n_dim=3)
        )


def test_load_vectors_from_npy(tmp_path):
    array = _cloud(12, 6, seed=4)
    path = os.path.join(tmp_path, "vectors.npy")
    np.save(path, array.astype(np.float32))

    loaded = io.load_vectors(path)
    assert loaded.dtype == np.float64
    np.testing.assert_allclose(loaded, array, rtol=1e-6, atol=1e-6)


def test_load_vectors_from_npz(tmp_path):
    array = _cloud(12, 6, seed=5)
    path = os.path.join(tmp_path, "vectors.npz")
    np.savez(path, z_p=array, z_t=array * 2)

    np.testing.assert_array_equal(io.load_vectors(path, key="z_p"), array)

    with pytest.raises(ValueError, match="specify which one"):
        io.load_vectors(path)
    with pytest.raises(KeyError, match="z_c"):
        io.load_vectors(path, key="z_c")


def test_load_vectors_from_single_array_npz(tmp_path):
    array = _cloud(12, 6, seed=6)
    path = os.path.join(tmp_path, "vectors.npz")
    np.savez(path, z_p=array)

    np.testing.assert_array_equal(io.load_vectors(path), array)
