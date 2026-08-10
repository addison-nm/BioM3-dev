"""Latent-space geometry
"""

from biom3.geometry.base import ManifoldModel, available_methods, get_method
from biom3.geometry.gaussian import (
    MIN_REFERENCE_SIZE,
    GaussianManifold,
    NormMismatchWarning,
    band_position,
    reference_band,
)
from biom3.geometry.manifold import DEFAULT_METHOD, fit_manifold, manifold_score
from biom3.geometry.io import load_manifold, load_vectors, save_manifold

__all__ = [
    "DEFAULT_METHOD",
    "MIN_REFERENCE_SIZE",
    "GaussianManifold",
    "ManifoldModel",
    "NormMismatchWarning",
    "available_methods",
    "band_position",
    "fit_manifold",
    "get_method",
    "load_manifold",
    "load_vectors",
    "manifold_score",
    "reference_band",
    "save_manifold",
]
