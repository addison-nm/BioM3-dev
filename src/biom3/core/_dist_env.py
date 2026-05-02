"""Launcher-env scanning for distributed rank/world-size detection.

Leaf module: pure stdlib, zero biom3 dependencies. Both
``biom3.backend.device`` (for ``setup_logger``'s rank check) and
``biom3.core.distributed`` (for its public rank/world API) import from
here. Keeping this layer dependency-free is what breaks the
backend.device <-> core.distributed import cycle.

The public API is re-exported from ``biom3.core.distributed`` — callers
should import from there, not from this module.
"""

import os


_LOCAL_RANK_ENV_VARS = (
    "LOCAL_RANK",
    "PALS_LOCAL_RANKID",
    "MPI_LOCALRANKID",
    "OMPI_COMM_WORLD_LOCAL_RANK",
)
_GLOBAL_RANK_ENV_VARS = (
    "RANK",
    "PALS_RANKID",
    "PMI_RANK",
    "OMPI_COMM_WORLD_RANK",
)
_WORLD_SIZE_ENV_VARS = (
    "WORLD_SIZE",
    "PMI_SIZE",
    "OMPI_COMM_WORLD_SIZE",
)


def _read_int_env(names, default=None):
    for var in names:
        val = os.environ.get(var)
        if val is not None and val != "":
            return int(val)
    return default


def get_local_rank() -> int:
    return _read_int_env(_LOCAL_RANK_ENV_VARS, default=0)


def get_global_rank() -> int:
    return _read_int_env(_GLOBAL_RANK_ENV_VARS, default=0)


def get_world_size() -> int:
    ws = _read_int_env(_WORLD_SIZE_ENV_VARS)
    if ws is not None:
        return ws
    pals_local = os.environ.get("PALS_LOCAL_SIZE")
    nodefile = os.environ.get("PBS_NODEFILE")
    if pals_local and nodefile and os.path.exists(nodefile):
        with open(nodefile) as fh:
            nodes = sum(1 for line in fh if line.strip())
        return int(pals_local) * nodes
    return 1


def is_launched() -> bool:
    """True iff at least one launcher env var is set, indicating mpiexec/torchrun."""
    for var in _GLOBAL_RANK_ENV_VARS + _LOCAL_RANK_ENV_VARS + _WORLD_SIZE_ENV_VARS:
        if os.environ.get(var) not in (None, ""):
            return True
    return False
