"""Smoke 3/4: does biom3.core._dist_env agree with the raw env vars?

Each rank emits two parallel single-line records:

  [rank_audit raw]    - independent re-scan of the env-var tuples
  [rank_audit biom3]  - what the audited code (core._dist_env +
                        backend.device + backend.xpu/cuda/cpu) reports

The two should agree. If they don't, the audit's tuple ordering or
fallback logic is wrong for this launcher.

We do NOT call init_process_group here - that's pg_probe.py. This
script only tests env interpretation and per-rank device resolution.

Usage::

    NGPU_PER_NODE=12 NGPU_TOTAL=24 \\
        ./scripts/launchers/aurora_multinode.sh \\
            python tests/_smoke/rank_audit.py
"""

import os
import socket
import sys


# --- Independent raw scan (does not import biom3 yet) ---

_GLOBAL_RANK_VARS = ("RANK", "PALS_RANKID", "PMI_RANK", "OMPI_COMM_WORLD_RANK")
_LOCAL_RANK_VARS = (
    "LOCAL_RANK",
    "PALS_LOCAL_RANKID",
    "MPI_LOCALRANKID",
    "OMPI_COMM_WORLD_LOCAL_RANK",
)
_WORLD_SIZE_VARS = ("WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE")


def _first_set(names):
    for n in names:
        v = os.environ.get(n)
        if v is not None and v != "":
            return n, v
    return None, None


def _raw_record():
    g_name, g_val = _first_set(_GLOBAL_RANK_VARS)
    l_name, l_val = _first_set(_LOCAL_RANK_VARS)
    w_name, w_val = _first_set(_WORLD_SIZE_VARS)

    if w_val is None:
        pals_local = os.environ.get("PALS_LOCAL_SIZE")
        nodefile = os.environ.get("PBS_NODEFILE")
        if pals_local and nodefile and os.path.exists(nodefile):
            with open(nodefile) as fh:
                nodes = sum(1 for line in fh if line.strip())
            w_val = str(int(pals_local) * nodes)
            w_name = f"PALS_LOCAL_SIZE*nodes(={pals_local}*{nodes})"

    return {
        "global_rank_src": g_name or "<none>",
        "global_rank_val": g_val if g_val is not None else "0",
        "local_rank_src": l_name or "<none>",
        "local_rank_val": l_val if l_val is not None else "0",
        "world_size_src": w_name or "<none>",
        "world_size_val": w_val if w_val is not None else "1",
    }


def _emit(prefix, parts):
    sys.stdout.write(f"[rank_audit {prefix}] " + " ".join(parts) + "\n")
    sys.stdout.flush()


def main():
    host = socket.gethostname()
    pid = os.getpid()

    # Raw record first - does NOT depend on biom3
    raw = _raw_record()
    _emit("raw", [
        f"host={host}",
        f"pid={pid}",
        f"global={raw['global_rank_val']}<-{raw['global_rank_src']}",
        f"local={raw['local_rank_val']}<-{raw['local_rank_src']}",
        f"world={raw['world_size_val']}<-{raw['world_size_src']}",
    ])

    # Now import biom3 and emit the parallel record
    try:
        from biom3.core._dist_env import (
            get_global_rank, get_local_rank, get_world_size,
        )
        from biom3.backend.device import (
            BACKEND_NAME, DIST_BACKEND, resolve_device_for_local_rank,
        )
    except Exception as e:
        _emit("biom3", [
            f"host={host}", f"pid={pid}",
            f"import_error={type(e).__name__}:{e}",
        ])
        return

    g = get_global_rank()
    l = get_local_rank()
    w = get_world_size()
    dev = resolve_device_for_local_rank(l)

    _emit("biom3", [
        f"host={host}",
        f"pid={pid}",
        f"global={g}",
        f"local={l}",
        f"world={w}",
        f"backend={BACKEND_NAME}",
        f"dist_backend={DIST_BACKEND}",
        f"resolved_device={dev}",
    ])

    # Self-consistency check on this rank's own values
    raw_g = int(raw["global_rank_val"])
    raw_l = int(raw["local_rank_val"])
    raw_w = int(raw["world_size_val"])
    ok = (raw_g == g) and (raw_l == l) and (raw_w == w)
    _emit("check", [
        f"host={host}",
        f"pid={pid}",
        f"raw=({raw_g},{raw_l},{raw_w})",
        f"biom3=({g},{l},{w})",
        f"agree={ok}",
    ])


if __name__ == "__main__":
    main()
