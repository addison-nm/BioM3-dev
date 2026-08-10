"""Persistence for fitted manifolds, and loading of embedding matrices.

A fitted manifold is written as a single compressed ``.npz``: the two fields
every model has (``method``, ``n_dim``) plus whatever the method itself declares
through ``_arrays``. The stored ``method`` string is what makes the file
self-identifying -- :func:`load_manifold` looks it up in the registry and
rebuilds the right class, so a file written by one method can never be
misread as another.

No pickle: everything round-trips through plain numpy arrays, and loads are
performed with ``allow_pickle=False``. A method whose state does not fit that
(learned weights, say) should override save/load rather than bend this format.
"""

from __future__ import annotations

import numpy as np

from biom3.geometry.base import get_method

from biom3.geometry import gaussian  # noqa: F401  registers gaussian_shrinkage

SCHEMA_VERSION = 2

_RESERVED_KEYS = frozenset({"schema_version", "method", "n_dim"})


def save_manifold(path, manifold):
    """Write a fitted manifold to ``path`` as a compressed ``.npz``."""
    arrays = manifold._arrays()
    collisions = _RESERVED_KEYS.intersection(arrays)
    if collisions:
        raise ValueError(
            f"method {manifold.METHOD!r} declares reserved array names: "
            f"{', '.join(sorted(collisions))}"
        )
    np.savez_compressed(
        path,
        schema_version=np.int64(SCHEMA_VERSION),
        method=np.array(manifold.METHOD),
        n_dim=np.int64(manifold.n_dim),
        **arrays,
    )


def load_manifold(path):
    """Read a manifold written by :func:`save_manifold`.

    Raises:
        ValueError: if the file was written by an incompatible schema version,
            or names a method this build does not know.
    """
    with np.load(path, allow_pickle=False) as data:
        version = int(data["schema_version"])
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: manifold schema version {version} is not supported "
                f"by this build (expected {SCHEMA_VERSION})"
            )
        cls = get_method(str(data["method"]))
        arrays = {k: data[k] for k in data.files if k not in _RESERVED_KEYS}
        return cls(n_dim=int(data["n_dim"]), **cls._from_arrays(arrays))


def load_vectors(path, key=None):
    """Load an ``(n, D)`` embedding matrix from ``.npy`` or ``.npz``.

    Args:
        path: path to a ``.npy`` file, or a ``.npz`` archive.
        key: array name to read from an ``.npz``. Required when the archive
            holds more than one array.
    """
    if str(path).endswith(".npz"):
        with np.load(path, allow_pickle=False) as data:
            names = list(data.files)
            if key is None:
                if len(names) != 1:
                    raise ValueError(
                        f"{path} holds {len(names)} arrays ({', '.join(names)}); "
                        "specify which one to load"
                    )
                key = names[0]
            elif key not in names:
                raise KeyError(
                    f"{path} has no array {key!r}; available: {', '.join(names)}"
                )
            array = np.asarray(data[key], dtype=np.float64)
    else:
        array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    return array
