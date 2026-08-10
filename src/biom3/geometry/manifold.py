"""Functional entry points for manifold fitting and scoring.

Two calls, deliberately separate:

1. :func:`fit_manifold` -- fit a reference cloud once, persist the result
   (:mod:`biom3.geometry.io`).
2. :func:`manifold_score` -- score query vectors against a fitted manifold,
   many times, without ever re-touching the reference.

For the default method that split is also a cost split: the fit is a ``D x D``
shrinkage inverse, the score is a quadratic form. Other methods invert that
balance, but the interface is the same.

Arrays in, arrays out. Producing the embeddings is not this package's job; the
caller supplies a reference matrix ``R`` of shape ``(M, D)`` and query matrices
``X`` of shape ``(N, D)``, and ``D`` is whatever the data says it is.

    from biom3.geometry import fit_manifold, manifold_score

    manifold = fit_manifold(reference, label="PenCL z_p / Sho1 naturals")
    scores = manifold_score(queries, manifold)

Both calls are thin: everything method-specific -- extra fit options, extra
score options -- passes straight through to the model. See
:class:`~biom3.geometry.base.ManifoldModel` for what a method must implement,
and :class:`~biom3.geometry.gaussian.GaussianManifold` for the default one.
"""

from __future__ import annotations

from biom3.geometry.base import get_method
from biom3.geometry.gaussian import GaussianManifold

DEFAULT_METHOD = GaussianManifold.METHOD

__all__ = ["DEFAULT_METHOD", "fit_manifold", "manifold_score"]


def fit_manifold(reference, method=DEFAULT_METHOD, **params):
    """Fit a manifold to an ``(M, D)`` reference cloud.

    Args:
        reference: reference vectors.
        method: registered method name (see
            :func:`~biom3.geometry.base.available_methods`).
        **params: method-specific fit options -- for the default method,
            ``label``.

    Returns:
        A fitted :class:`~biom3.geometry.base.ManifoldModel`.
    """
    return get_method(method).fit(reference, **params)


def manifold_score(queries, manifold, **kwargs):
    """Score query vectors against a fitted manifold. Lower = more on-manifold.

    A thin wrapper over :meth:`~biom3.geometry.base.ManifoldModel.score`;
    ``kwargs`` are method-specific score options -- for the default method,
    ``check_norm`` and ``norm_ratio_tol``.
    """
    return manifold.score(queries, **kwargs)
