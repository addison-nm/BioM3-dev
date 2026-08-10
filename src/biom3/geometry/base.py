"""Method-agnostic base for reference-cloud models.

A *manifold model* is a fitted description of a reference cloud of embedding
vectors, together with a way to score how far any query vector sits from it.
Concrete methods differ in what they fit (a Gaussian ellipsoid, a mixture, a
neighbor graph, a learned decoder) and in what else they carry, but share two
operations:

    fit(R: (M, D))          -> model
    model.score(X: (N, D))  -> (N,) float, lower = more on-manifold

That is the whole contract. In particular the base promises a *score*, not a
metric: for :class:`~biom3.geometry.gaussian.GaussianManifold` the score is a
true Mahalanobis distance in units of reference-cloud standard deviations, but a
neighbor distance or a reconstruction error satisfies none of that, and a
log-density is not even sign-constrained. Scores are comparable across query
sets scored against *the same* model; they are not comparable across methods.

Anything beyond fit and score -- reference bands, provenance guards, minimum
reference sizes, hyperparameters -- belongs to the method that needs it, because
the sensible answer differs per method. A single global fit can score its own
reference set in-sample; a memorizing method cannot (under kNN every reference
point is its own nearest neighbor, so self-scores collapse to zero without
leave-one-out).

A new method subclasses :class:`ManifoldModel`, declares ``METHOD``, adds its
fitted state as dataclass fields, and implements four hooks: ``_fit_state``,
``_score``, ``_arrays``, ``_from_arrays``. Declaring ``METHOD`` registers the
class, which is what lets :mod:`biom3.geometry.io` reload a persisted model as
the right type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

_REGISTRY: dict[str, type] = {}


def available_methods():
    """Names of every registered manifold method."""
    return sorted(_REGISTRY)


def get_method(name):
    """Look up a registered manifold class by its ``METHOD`` string."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown manifold method {name!r}; available: "
            f"{', '.join(available_methods()) or '(none registered)'}"
        ) from None


def as_matrix(array, name):
    """Validate and cast an ``(n, D)`` float64 array."""
    matrix = np.asarray(array, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D (n, D) array; got shape {matrix.shape}"
        )
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} is empty; got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return matrix


@dataclass(frozen=True, kw_only=True, repr=False, eq=False)
class ManifoldModel(ABC):
    """A fitted reference cloud that can score query vectors.

    Subclasses add their fitted state, and whatever else they need, as further
    fields.

    Attributes:
        n_dim: dimension the model was fit in. Query matrices must match it.
    """

    n_dim: int

    METHOD: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        method = cls.__dict__.get("METHOD", "")
        if not method:
            return
        registered = _REGISTRY.get(method)
        if registered is not None and registered is not cls:
            raise ValueError(
                f"manifold method {method!r} is already registered to "
                f"{registered.__name__}"
            )
        _REGISTRY[method] = cls

    @classmethod
    def fit(cls, reference, **params):
        """Fit the model to an ``(M, D)`` reference cloud.

        Args:
            reference: reference vectors.
            **params: method-specific fit options, forwarded to ``_fit_state``.

        Raises:
            ValueError: if ``reference`` is not a finite 2-D array, or the
                method rejects it.
        """
        matrix = as_matrix(reference, "reference")
        return cls(n_dim=int(matrix.shape[1]), **cls._fit_state(matrix, **params))

    def score(self, queries):
        """Score each query vector against this manifold. Lower = more on-manifold.

        Raises:
            ValueError: if ``queries`` is not a finite 2-D array or its
                dimension does not match the model's.
        """
        matrix = as_matrix(queries, "queries")
        if matrix.shape[1] != self.n_dim:
            raise ValueError(
                f"queries have dimension {matrix.shape[1]} but the manifold was "
                f"fit in dimension {self.n_dim}"
            )
        return self._score(matrix)

    @classmethod
    @abstractmethod
    def _fit_state(cls, reference, **params):
        """Fit the method's state; return it as a dict of field values.

        ``reference`` is already validated and cast to ``(M, D)`` float64.
        """

    @abstractmethod
    def _score(self, queries):
        """Score a validated ``(N, D)`` float64 query matrix."""

    @abstractmethod
    def _arrays(self):
        """The method's state, as a dict of arrays/scalars to persist."""

    @classmethod
    @abstractmethod
    def _from_arrays(cls, arrays):
        """Rebuild the method's field values from a persisted mapping."""

    def __repr__(self):
        return f"{type(self).__name__}(method={self.METHOD!r}, n_dim={self.n_dim})"
