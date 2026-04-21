"""Device detection, selection, and helper utilities

"""

import os
import logging
import platform
import socket
import sys
import torch

_CPU = "cpu"
_CUDA = "cuda"
_XPU = "xpu"

def get_backend_name() -> str:
    if torch.cuda.is_available():
        return _CUDA
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return _XPU
    return _CPU

def get_device():
    backend = get_backend_name()
    if backend == _CUDA:
        return torch.device("cuda")
    if backend == _XPU:
        return torch.device("xpu")
    return torch.device(_CPU)


def reset_peak_memory_stats():
    """Reset peak memory counters for the active device backend.

    No-op on CPU.  CUDA and XPU expose the same-named function.
    """
    backend = get_backend_name()
    if backend == _CUDA:
        torch.cuda.reset_peak_memory_stats()
    elif backend == _XPU:
        torch.xpu.reset_peak_memory_stats()


def get_peak_memory_stats():
    """Return ``(peak_allocated_bytes, peak_reserved_bytes)`` for the local device.

    Returns ``(None, None)`` on CPU or if the backend does not expose the
    relevant APIs.  CUDA and XPU expose identical function names.
    """
    backend = get_backend_name()
    if backend == _CUDA:
        return (torch.cuda.max_memory_allocated(),
                torch.cuda.max_memory_reserved())
    if backend == _XPU:
        try:
            return (torch.xpu.max_memory_allocated(),
                    torch.xpu.max_memory_reserved())
        except AttributeError:
            return (None, None)
    return (None, None)


def device_sync():
    """Block the host until all queued work on the active device finishes.

    Intended for benchmark timing — wrap before ``perf_counter`` reads so
    asynchronous kernel launches are accounted for.  No-op on CPU.
    """
    backend = get_backend_name()
    if backend == _CUDA:
        torch.cuda.synchronize()
    elif backend == _XPU:
        torch.xpu.synchronize()


def get_device_name(index: int = 0) -> str:
    """Return a human-readable name for the active device.

    CUDA → ``torch.cuda.get_device_name``; XPU → ``torch.xpu.get_device_name``;
    CPU → ``platform.processor()`` (falls back to ``platform.machine()``).
    """
    backend = get_backend_name()
    if backend == _CUDA:
        return torch.cuda.get_device_name(index)
    if backend == _XPU:
        try:
            return torch.xpu.get_device_name(index)
        except AttributeError:
            return "xpu"
    return platform.processor() or platform.machine() or "cpu"


def get_device_info(index: int = 0) -> dict:
    """Return a dict describing the active compute device and host environment.

    Suitable for serialising alongside benchmark results so runs on different
    machines (Spark / Polaris / Aurora / CPU) remain distinguishable.
    """
    backend = get_backend_name()
    info = {
        "backend": backend,
        "device_name": get_device_name(index),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if backend == _CUDA:
        info["device_count"] = torch.cuda.device_count()
        try:
            cap = torch.cuda.get_device_capability(index)
            info["cuda_capability"] = f"{cap[0]}.{cap[1]}"
        except Exception:
            pass
        info["cuda_version"] = torch.version.cuda
    elif backend == _XPU:
        try:
            info["device_count"] = torch.xpu.device_count()
        except AttributeError:
            info["device_count"] = None
    else:
        info["device_count"] = 0
    return info

def get_rank() -> int:
    """Get the global rank from environment variables set by the launcher.

    Checks variables from multiple launchers:
    - torchrun / deepspeed: RANK, LOCAL_RANK
    - PALS (Aurora mpiexec): PALS_RANKID, PALS_LOCAL_RANKID
    - PMI (Cray MPICH): PMI_RANK
    - Open MPI: OMPI_COMM_WORLD_RANK

    Returns 0 when running without a launcher (single-process).
    """
    for var in ("RANK", "PALS_RANKID", "PMI_RANK", "OMPI_COMM_WORLD_RANK",
                "LOCAL_RANK", "PALS_LOCAL_RANKID"):
        val = os.environ.get(var)
        if val is not None:
            return int(val)
    return 0


def setup_logger(name: str = "biom3", level: int = logging.INFO) -> logging.Logger:
    """Return a rank-aware logger that only emits on rank 0.

    Non-zero ranks are silenced (set to CRITICAL) so that duplicate
    messages are never printed in multi-node / multi-GPU settings.

    Call once per module::

        from biom3.backend.device import setup_logger
        logger = setup_logger(__name__)
        logger.info("only printed on rank 0")
    """
    logger = logging.getLogger(name)
    logger.propagate = False  # prevent duplicate output via root logger
    rank = get_rank()
    if rank == 0:
        logger.setLevel(level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            ))
            logger.addHandler(handler)
    else:
        logger.setLevel(logging.CRITICAL)
    return logger


# Set the backend name for import convenience
BACKEND_NAME = get_backend_name()

# Import from device-specific module
if BACKEND_NAME == _CUDA:
    from .cuda import *
elif BACKEND_NAME == _XPU:
    from .xpu import *
elif BACKEND_NAME == _CPU:
    from .cpu import *
else:
    raise RuntimeError(f"Unexpected backend name: {BACKEND_NAME}")
