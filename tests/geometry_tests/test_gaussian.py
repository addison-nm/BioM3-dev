"""Tests for the Gaussian shrinkage manifold and the functional facade.

Everything runs on synthetic Gaussian clouds -- no weights, no fixtures, no
PenCL. Two of these tests pin our numbers to the two reference implementations
this method was ported from.
"""

import warnings

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

from biom3.geometry.gaussian import (
    MIN_REFERENCE_SIZE,
    GaussianManifold,
    NormMismatchWarning,
    band_position,
)
from biom3.geometry.manifold import DEFAULT_METHOD, fit_manifold, manifold_score


def _cloud(n, d, seed, scale=1.0, mean=0.0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=scale, size=(n, d))


def _correlated_cloud(n, d, seed):
    """A cloud with a non-trivial covariance (anisotropic, correlated)."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(d, d)) / np.sqrt(d)
    return rng.normal(size=(n, d)) @ factor + rng.normal(size=(1, d))


def _random_orthogonal(d, seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    return q * np.sign(np.diag(r))


def test_facade_fits_the_gaussian_method_by_default():
    manifold = fit_manifold(_correlated_cloud(40, 16, seed=0), label="synthetic")

    assert DEFAULT_METHOD == GaussianManifold.METHOD == "gaussian_shrinkage"
    assert isinstance(manifold, GaussianManifold)
    assert manifold.label == "synthetic"


def test_fit_reports_shape_size_and_band():
    reference = _correlated_cloud(60, 16, seed=1)
    manifold = fit_manifold(reference)

    assert manifold.n_dim == 16
    assert manifold.n_reference == 60
    assert manifold.centroid.shape == (16,)
    assert manifold.precision.shape == (16, 16)
    assert manifold.covariance.shape == (16, 16)
    assert 0.0 <= manifold.shrinkage <= 1.0
    assert manifold.band["p5"] <= manifold.band["median"] <= manifold.band["p95"]
    assert manifold.ref_mean_norm == pytest.approx(
        np.linalg.norm(reference, axis=1).mean()
    )


def test_band_is_the_reference_set_scored_against_itself():
    reference = _correlated_cloud(80, 12, seed=2)
    manifold = fit_manifold(reference)

    self_distances = manifold_score(reference, manifold)
    p5, median, p95 = np.percentile(self_distances, [5, 50, 95])

    assert manifold.band["p5"] == pytest.approx(p5)
    assert manifold.band["median"] == pytest.approx(median)
    assert manifold.band["p95"] == pytest.approx(p95)


def test_matches_the_production_precision_route():
    """Parity with `sqrt((x - mu) @ LedoitWolf().fit(R).precision_ @ (x - mu))`."""
    reference = _correlated_cloud(100, 32, seed=3)
    queries = _correlated_cloud(20, 32, seed=4)

    ours = manifold_score(queries, fit_manifold(reference), check_norm=False)

    precision = LedoitWolf().fit(reference).precision_
    delta = queries - reference.mean(axis=0)
    expected = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, precision, delta), 0))

    np.testing.assert_allclose(ours, expected, rtol=1e-12, atol=1e-12)


def test_matches_the_pinv_of_shrunk_covariance_route():
    """Parity with the library route: pinv of the pre-centered shrunk covariance."""
    reference = _correlated_cloud(100, 32, seed=5)
    queries = _correlated_cloud(20, 32, seed=6)

    ours = manifold_score(queries, fit_manifold(reference), check_norm=False)

    centroid = reference.mean(axis=0)
    covariance = LedoitWolf().fit(reference - centroid).covariance_
    precision = np.linalg.pinv(covariance)
    delta = queries - centroid
    expected = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, precision, delta), 0))

    np.testing.assert_allclose(ours, expected, rtol=1e-7, atol=1e-7)


def test_singular_reference_smaller_than_dimension():
    """M < D is the normal case for a natural-homolog reference; shrinkage carries it."""
    reference = _correlated_cloud(177, 512, seed=7)
    queries = _correlated_cloud(10, 512, seed=8)

    manifold = fit_manifold(reference)
    distances = manifold_score(queries, manifold, check_norm=False)

    empirical = np.cov(reference, rowvar=False, bias=True)
    assert np.linalg.matrix_rank(empirical) < reference.shape[1]
    assert manifold.shrinkage > 0
    np.testing.assert_allclose(
        manifold.precision @ manifold.covariance, np.eye(512), atol=1e-8
    )
    assert distances.shape == (10,)
    assert np.all(np.isfinite(distances))
    assert np.all(distances > 0)


def test_centroid_has_zero_distance_and_never_produces_nan():
    manifold = fit_manifold(_correlated_cloud(40, 24, seed=9))

    distances = manifold_score(
        manifold.centroid[None, :], manifold, check_norm=False
    )

    assert distances.shape == (1,)
    assert not np.isnan(distances[0])
    assert distances[0] == pytest.approx(0.0, abs=1e-9)


def test_off_manifold_queries_score_further_than_the_reference_band():
    reference = _correlated_cloud(120, 20, seed=10)
    manifold = fit_manifold(reference)

    inside = manifold_score(_correlated_cloud(50, 20, seed=11), manifold,
                            check_norm=False)
    outside = manifold_score(
        _cloud(50, 20, seed=12, scale=5.0) + manifold.centroid,
        manifold, check_norm=False,
    )

    assert np.median(outside) > np.median(inside)
    assert np.median(outside) > manifold.band["p95"]


def test_distance_is_invariant_under_rotation_and_uniform_scaling():
    """The metric is defined by the reference cloud, not by the coordinate frame."""
    reference = _correlated_cloud(90, 16, seed=13)
    queries = _correlated_cloud(15, 16, seed=14)
    rotation = _random_orthogonal(16, seed=15)

    plain = manifold_score(queries, fit_manifold(reference), check_norm=False)
    transformed = manifold_score(
        3.0 * queries @ rotation,
        fit_manifold(3.0 * reference @ rotation),
        check_norm=False,
    )

    np.testing.assert_allclose(plain, transformed, rtol=1e-8, atol=1e-8)


def test_reference_below_the_minimum_size_is_rejected():
    too_small = _correlated_cloud(MIN_REFERENCE_SIZE - 1, 16, seed=16)
    with pytest.raises(ValueError, match="at least 8"):
        fit_manifold(too_small)

    just_enough = _correlated_cloud(MIN_REFERENCE_SIZE, 16, seed=17)
    assert fit_manifold(just_enough).n_reference == MIN_REFERENCE_SIZE


def test_norm_mismatch_warns_and_can_be_disabled():
    reference = _correlated_cloud(60, 16, seed=18) + 4.0
    manifold = fit_manifold(reference)
    normalized = reference / np.linalg.norm(reference, axis=1, keepdims=True)

    with pytest.warns(NormMismatchWarning, match=r"outside the tolerated range"):
        manifold_score(normalized, manifold)

    with warnings.catch_warnings():
        warnings.simplefilter("error", NormMismatchWarning)
        manifold_score(reference, manifold)
        manifold_score(normalized, manifold, check_norm=False)


def test_norm_warning_reports_both_norms_and_the_threshold():
    reference = _correlated_cloud(60, 16, seed=20) + 4.0
    manifold = fit_manifold(reference)

    with pytest.warns(NormMismatchWarning) as record:
        manifold_score(reference * 4.0, manifold, norm_ratio_tol=2.0)

    message = str(record[0].message)
    assert f"{manifold.ref_mean_norm:.4g}" in message
    assert f"{4.0 * manifold.ref_mean_norm:.4g}" in message
    assert "norm_ratio_tol=2" in message


def test_norm_ratio_tol_sets_where_the_warning_starts():
    reference = _correlated_cloud(60, 16, seed=21) + 4.0
    manifold = fit_manifold(reference)

    with warnings.catch_warnings():
        warnings.simplefilter("error", NormMismatchWarning)
        manifold_score(reference * 1.4, manifold)

    with pytest.warns(NormMismatchWarning):
        manifold_score(reference * 1.4, manifold, norm_ratio_tol=1.2)


def test_band_position_labels_each_distance():
    manifold = fit_manifold(_correlated_cloud(100, 12, seed=19))
    distances = np.array([
        manifold.band["p5"] - 1.0,
        manifold.band["median"],
        manifold.band["p95"] + 1.0,
    ])

    assert list(band_position(distances, manifold)) == ["below", "within", "above"]
