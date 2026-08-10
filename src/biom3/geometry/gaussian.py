"""Flat Gaussian manifold: centroid + Ledoit-Wolf shrinkage precision.

The reference cloud is described by its centroid ``mu`` and the shrinkage
inverse covariance ``P``, and a query's score is its Mahalanobis distance

    d(x) = sqrt((x - mu) @ P @ (x - mu))

Unlike the base contract, this score really is a metric: it is a norm in the
reference cloud's own geometry, measured in units of reference standard
deviations.

On top of fit and score this method carries three things of its own:

- **The reference band** -- the p5 / median / p95 of the reference set's own
  distances. A global fit can measure this in-sample.
- **The reference mean norm** -- checked against queries at score time.
- **A minimum reference size** of 8, below which a covariance is not fit.

Two properties of the fit are easy to undo by accident:

- ``M < D`` is expected and fine. The covariance is singular; Ledoit-Wolf
  shrinkage is what makes its inverse well-defined. Reducing the dimension first
  to avoid singularity changes the metric.
- The quadratic form is clamped at zero before the square root, so round-off on
  a query sitting at the centroid cannot produce a NaN.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf

from biom3.geometry.base import ManifoldModel, as_matrix

MIN_REFERENCE_SIZE = 8

NORM_RATIO_TOL = 1.5

BAND_KEYS = ("p5", "median", "p95")


class NormMismatchWarning(UserWarning):
    """Query vectors differ in mean L2 norm from the reference set."""


def reference_band(scores):
    """The p5 / median / p95 of a set of scores."""
    p5, median, p95 = np.percentile(np.asarray(scores, dtype=np.float64), [5, 50, 95])
    return {"p5": float(p5), "median": float(median), "p95": float(p95)}


def band_position(scores, manifold):
    """Where each score falls relative to a manifold's reference band.

    Returns an array of strings, one per score: ``"below"`` (tighter to the
    manifold than the reference 5th percentile), ``"within"``, or ``"above"``
    (outside the reference 95th percentile).
    """
    scores = np.asarray(scores, dtype=np.float64)
    position = np.full(scores.shape, "within", dtype="<U6")
    position[scores < manifold.band["p5"]] = "below"
    position[scores > manifold.band["p95"]] = "above"
    return position


@dataclass(frozen=True, kw_only=True, repr=False, eq=False)
class GaussianManifold(ManifoldModel):
    """A reference cloud as a single Gaussian ellipsoid.

    Attributes:
        centroid: ``(D,)`` mean of the reference vectors.
        precision: ``(D, D)`` Ledoit-Wolf shrinkage inverse covariance.
        covariance: ``(D, D)`` the corresponding shrunk covariance.
        shrinkage: the Ledoit-Wolf shrinkage coefficient chosen for the fit.
        n_reference: number of reference vectors the fit used.
        band: the reference set's own distances, as ``{"p5", "median", "p95"}``.
        ref_mean_norm: mean L2 norm of the reference vectors.
        label: free-form provenance string (embedding checkpoint, reference set
            name, ...).
    """

    centroid: np.ndarray
    precision: np.ndarray
    covariance: np.ndarray
    shrinkage: float
    n_reference: int
    band: dict
    ref_mean_norm: float
    label: str = ""

    METHOD = "gaussian_shrinkage"

    @classmethod
    def _fit_state(cls, reference, label=""):
        n_ref = reference.shape[0]
        if n_ref < MIN_REFERENCE_SIZE:
            raise ValueError(
                f"a covariance cannot be fit from {n_ref} reference vectors; "
                f"at least {MIN_REFERENCE_SIZE} are required"
            )
        estimator = LedoitWolf().fit(reference)
        centroid = reference.mean(axis=0)
        precision = np.asarray(estimator.precision_, dtype=np.float64)
        delta = reference - centroid
        self_distances = np.sqrt(
            np.maximum(np.einsum("ij,jk,ik->i", delta, precision, delta), 0.0)
        )
        return {
            "centroid": centroid,
            "precision": precision,
            "covariance": np.asarray(estimator.covariance_, dtype=np.float64),
            "shrinkage": float(estimator.shrinkage_),
            "n_reference": int(n_ref),
            "band": reference_band(self_distances),
            "ref_mean_norm": float(np.linalg.norm(reference, axis=1).mean()),
            "label": str(label),
        }

    def score(self, queries, check_norm=True, norm_ratio_tol=NORM_RATIO_TOL):
        """Mahalanobis distance of each query from this manifold.

        Args:
            queries: ``(N, D)`` query vectors, with the same ``D`` as the fit.
            check_norm: compare the queries' mean L2 norm against the reference
                mean norm and warn when they differ by more than
                ``norm_ratio_tol``. Scores shift with the scale of the queries,
                so a large gap is worth seeing before the numbers are read; one
                set L2-normalized and the other not is the case that motivated
                the check, but the check only reports the norms it measured.
            norm_ratio_tol: warn when the ratio of mean norms falls outside
                ``[1 / norm_ratio_tol, norm_ratio_tol]``.
        """
        if check_norm:
            self._warn_on_norm_mismatch(as_matrix(queries, "queries"), norm_ratio_tol)
        return super().score(queries)

    def _warn_on_norm_mismatch(self, matrix, norm_ratio_tol):
        query_norm = float(np.linalg.norm(matrix, axis=1).mean())
        if self.ref_mean_norm <= 0 or query_norm <= 0:
            return
        ratio = query_norm / self.ref_mean_norm
        if ratio > norm_ratio_tol or ratio < 1.0 / norm_ratio_tol:
            warnings.warn(
                f"query mean L2 norm {query_norm:.4g} is {ratio:.3g}x the reference "
                f"mean norm {self.ref_mean_norm:.4g}, outside the tolerated range "
                f"{1.0 / norm_ratio_tol:.3g}x-{norm_ratio_tol:.3g}x (norm_ratio_tol="
                f"{norm_ratio_tol:.3g})",
                NormMismatchWarning,
                stacklevel=4,
            )

    def _score(self, queries):
        delta = queries - self.centroid
        quadratic = np.einsum("ij,jk,ik->i", delta, self.precision, delta)
        return np.sqrt(np.maximum(quadratic, 0.0))

    def _arrays(self):
        return {
            "centroid": self.centroid,
            "precision": self.precision,
            "covariance": self.covariance,
            "shrinkage": np.float64(self.shrinkage),
            "n_reference": np.int64(self.n_reference),
            "band": np.array([self.band[k] for k in BAND_KEYS], dtype=np.float64),
            "band_keys": np.array(BAND_KEYS),
            "ref_mean_norm": np.float64(self.ref_mean_norm),
            "label": np.array(self.label),
        }

    @classmethod
    def _from_arrays(cls, arrays):
        band_keys = [str(k) for k in arrays["band_keys"]]
        band_values = [float(v) for v in arrays["band"]]
        return {
            "centroid": np.asarray(arrays["centroid"], dtype=np.float64),
            "precision": np.asarray(arrays["precision"], dtype=np.float64),
            "covariance": np.asarray(arrays["covariance"], dtype=np.float64),
            "shrinkage": float(arrays["shrinkage"]),
            "n_reference": int(arrays["n_reference"]),
            "band": dict(zip(band_keys, band_values)),
            "ref_mean_norm": float(arrays["ref_mean_norm"]),
            "label": str(arrays["label"]),
        }

    def __repr__(self):
        label = f", label={self.label!r}" if self.label else ""
        return (
            f"GaussianManifold(n_dim={self.n_dim}, n_reference={self.n_reference}, "
            f"shrinkage={self.shrinkage:.4g}, "
            f"band={{p5: {self.band['p5']:.3g}, median: {self.band['median']:.3g}, "
            f"p95: {self.band['p95']:.3g}}}, "
            f"ref_mean_norm={self.ref_mean_norm:.4g}{label})"
        )
