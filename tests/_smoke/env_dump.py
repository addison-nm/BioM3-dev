"""Smoke 1/4: dump every distributed-related env var, per rank.

No biom3 imports, no torch. Just os.environ. By default each rank emits
one single-line record (greppable). Pass ``--multiline`` (``-m``) to
emit one ``key=value`` per line instead - easier to scan when reading
a single rank's output, less convenient for cross-rank diffing.

Aggregation across ranks is left to the launcher's combined log file
(e.g. PBS .o file or redirected log).

Usage (Aurora multinode example)::

    NGPU_PER_NODE=12 NGPU_TOTAL=24 \\
        ./scripts/launchers/aurora_multinode.sh \\
            python tests/_smoke/env_dump.py [--multiline]
"""

import argparse
import os
import socket
import sys


_VARS = (
    # Global rank
    "RANK",
    "PALS_RANKID",
    "PMI_RANK",
    "OMPI_COMM_WORLD_RANK",
    # Local rank
    "LOCAL_RANK",
    "PALS_LOCAL_RANKID",
    "MPI_LOCALRANKID",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    # World / local size
    "WORLD_SIZE",
    "PALS_LOCAL_SIZE",
    "PMI_SIZE",
    "OMPI_COMM_WORLD_SIZE",
    # PALS topology (per-rank exported)
    "PALS_NODEID",
    "PALS_NUM_NODES",
    "PALS_PPN",
    "PALS_DEPTH",
    # Rendezvous
    "MASTER_ADDR",
    "MASTER_PORT",
    # Device pinning
    "ZE_AFFINITY_MASK",
    "CUDA_VISIBLE_DEVICES",
    # PBS / job context
    "PBS_NODEFILE",
    "PBS_JOBID",
    "PBS_O_WORKDIR",
)

_GLOBAL_RANK_CANDIDATES = ("RANK", "PALS_RANKID", "PMI_RANK", "OMPI_COMM_WORLD_RANK")


def _kv(name):
    val = os.environ.get(name)
    if val is None:
        return f"{name}=<unset>"
    if val == "":
        return f"{name}=<empty>"
    return f"{name}={val}"


def _rank_label():
    """Best-effort ``rank=N`` tag for the per-line prefix in multiline mode.

    Falls back to ``rank=?`` when no candidate is set.
    """
    for var in _GLOBAL_RANK_CANDIDATES:
        v = os.environ.get(var)
        if v not in (None, ""):
            return f"rank={v}"
    return "rank=?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-l", "--multiline", action="store_true",
        help="emit one key=value per line (default: single line per rank)",
    )
    args = ap.parse_args()

    host = socket.gethostname()
    pid = os.getpid()

    if args.multiline:
        prefix = f"[env_dump {_rank_label()} host={host} pid={pid}]"
        for v in _VARS:
            sys.stdout.write(f"{prefix} {_kv(v)}\n")
        sys.stdout.flush()
        return

    parts = [f"host={host}", f"pid={pid}"]
    parts.extend(_kv(v) for v in _VARS)
    sys.stdout.write("[env_dump] " + " ".join(parts) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
