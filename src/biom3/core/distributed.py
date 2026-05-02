"""Distributed launch helpers shared across BioM3 entrypoints.

The Stage 3 sampling pipeline (and any future inference pipeline) launches
one process per device under ``mpiexec`` on Aurora / Polaris. This module
exposes a small, well-defined surface so each entrypoint only has to call
``init_distributed_if_launched`` near the top of ``main()``.

Single-process behaviour is preserved: when no launcher env vars are
present, ``init_distributed_if_launched`` is a no-op that returns
``(0, 0, 1, requested_device)`` — no ``init_process_group`` call.
"""

import atexit
import os

import torch.distributed as dist

from biom3.backend.device import (
    BACKEND_NAME,
    DIST_BACKEND,
    _CPU,
    resolve_device_for_local_rank,
    set_device_for_local_rank,
)
# Public rank / world-size API is implemented in the leaf module
# `_dist_env` (pure stdlib) so that `backend.device.setup_logger` can also
# use it without creating a backend.device <-> core.distributed import
# cycle. Re-exported below as part of the public surface of this module.
from biom3.core._dist_env import (  # noqa: F401  (re-exported)
    get_global_rank,
    get_local_rank,
    get_world_size,
    is_launched,
)


def _resolve_device_str(device: str, local_rank: int) -> str:
    """Map ``local_rank`` to a concrete per-rank device string.

    Per-backend resolution lives in ``biom3.backend.{cpu,cuda,xpu}``. This
    function only honours an explicit ``cpu`` request (used by tests on
    GPU hosts) and otherwise delegates to the active backend.
    """
    requested = (device or "").lower()
    if requested == _CPU or BACKEND_NAME == _CPU:
        return _CPU
    return resolve_device_for_local_rank(local_rank)


def _set_master_addr_port():
    """Populate MASTER_ADDR / MASTER_PORT if the launcher hasn't already.

    Applies to any PBS + PALS deployment (Aurora, Polaris). PALS doesn't
    surface either var. MASTER_ADDR comes from the first line of
    $PBS_NODEFILE (the head node of the allocation). MASTER_PORT
    defaults to 29500 — the torch.distributed convention. PBS allocates
    compute nodes exclusively per job, so cross-job port collisions
    can't happen. Override either by exporting it in the launcher / job
    script before invoking. torchrun-style launches that already set
    both env vars are unaffected (setdefault is a no-op there).
    """
    if not os.environ.get("MASTER_ADDR"):
        nodefile = os.environ.get("PBS_NODEFILE")
        if nodefile and os.path.exists(nodefile):
            with open(nodefile) as fh:
                first = next((line.strip() for line in fh if line.strip()), None)
            if first:
                os.environ["MASTER_ADDR"] = first
        else:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


def _destroy_if_initialized():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def init_distributed_if_launched(device: str) -> tuple[int, int, int, str]:
    """Initialise ``torch.distributed`` if launched under mpiexec / torchrun.

    Returns ``(rank, local_rank, world_size, resolved_device)``. When no
    launcher env vars are present, this is a no-op and returns
    ``(0, 0, 1, device)`` so the single-process entrypoint path is
    preserved exactly.
    """
    if not is_launched():
        return 0, 0, 1, device

    rank = get_global_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()

    resolved_device = _resolve_device_str(device, local_rank)

    if resolved_device != _CPU:
        set_device_for_local_rank(local_rank)

    if dist.is_available() and not dist.is_initialized():
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        _set_master_addr_port()

        dist.init_process_group(
            backend=DIST_BACKEND,
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        atexit.register(_destroy_if_initialized)

    return rank, local_rank, world_size, resolved_device


def is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return get_global_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def gather_object_to_main(obj, dst: int = 0):
    """Gather ``obj`` from every rank to ``dst``. Returns a list on dst, ``None`` elsewhere.

    No-op (returns ``[obj]``) when not running under a distributed launcher.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [None] * world_size if rank == dst else None
    dist.gather_object(obj, gathered, dst=dst)
    return gathered


def broadcast_int(value: int, src: int = 0) -> int:
    """Broadcast a single int from ``src`` to all ranks. No-op when single-process."""
    if not (dist.is_available() and dist.is_initialized()):
        return value
    payload = [int(value)]
    dist.broadcast_object_list(payload, src=src)
    return int(payload[0])
