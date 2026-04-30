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
import hashlib
import os

import torch
import torch.distributed as dist

from biom3.backend.device import BACKEND_NAME, _CPU, _CUDA, _XPU


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


def _resolve_device_str(device: str, local_rank: int) -> str:
    """Map ``local_rank`` to a concrete per-rank device string.

    On Aurora the launcher's ``--cpu-bind`` and ``ZE_AFFINITY_MASK``
    typically pin one tile per process, so ``torch.xpu.device_count()`` is
    1 inside each rank and the only valid device id is ``xpu:0``. When
    the affinity mask is *not* set (e.g. interactive single-process
    multi-tile testing) we fall back to ``xpu:{local_rank}``.
    """
    requested = (device or "").lower()
    if requested == _CPU or BACKEND_NAME == _CPU:
        return _CPU
    if requested == _CUDA or BACKEND_NAME == _CUDA:
        return f"cuda:{local_rank}"
    if requested == _XPU or BACKEND_NAME == _XPU:
        if os.environ.get("ZE_AFFINITY_MASK"):
            return "xpu:0"
        try:
            count = torch.xpu.device_count()
        except (AttributeError, RuntimeError):
            count = 1
        if count <= 1:
            return "xpu:0"
        return f"xpu:{local_rank}"
    return device


def _select_backend() -> str:
    if BACKEND_NAME == _XPU:
        return "xccl"
    if BACKEND_NAME == _CUDA:
        return "nccl"
    return "gloo"


def _set_master_addr_port():
    """Populate MASTER_ADDR / MASTER_PORT if the launcher hasn't already.

    Pulls MASTER_ADDR from the first line of $PBS_NODEFILE and uses a
    deterministic MASTER_PORT derived from $PBS_JOBID (so two concurrent
    jobs on the same login node don't collide on port 29500).
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
    if not os.environ.get("MASTER_PORT"):
        jobid = os.environ.get("PBS_JOBID", "")
        offset = int(hashlib.blake2b(jobid.encode(), digest_size=2).hexdigest(), 16) % 1000
        os.environ["MASTER_PORT"] = str(29500 + offset)


def _device_launched() -> bool:
    """True iff at least one launcher env var is set, indicating mpiexec/torchrun."""
    candidates = _GLOBAL_RANK_ENV_VARS + _LOCAL_RANK_ENV_VARS + _WORLD_SIZE_ENV_VARS
    return any(os.environ.get(v) not in (None, "") for v in candidates)


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
    if not _device_launched():
        return 0, 0, 1, device

    rank = get_global_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()

    resolved_device = _resolve_device_str(device, local_rank)

    if BACKEND_NAME == _CUDA and resolved_device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
    elif BACKEND_NAME == _XPU and resolved_device.startswith("xpu"):
        try:
            torch.xpu.set_device(int(resolved_device.split(":")[-1]))
        except (AttributeError, ValueError, RuntimeError):
            pass

    if dist.is_available() and not dist.is_initialized():
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        _set_master_addr_port()

        backend = _select_backend()
        dist.init_process_group(
            backend=backend,
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
