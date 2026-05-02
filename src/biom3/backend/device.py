"""Device detection, selection, and helper utilities

"""

import logging
import os
import platform
import socket
import sys

import psutil
import torch

from biom3.core._dist_env import get_global_rank

_CPU = "cpu"
_CUDA = "cuda"
_XPU = "xpu"


def get_backend_name() -> str:
    if torch.cuda.is_available():
        return _CUDA
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return _XPU
    return _CPU


# Computed once at import time. The active backend never changes within a
# process, so every helper below references this constant directly instead
# of recomputing it on each call.
BACKEND_NAME = get_backend_name()


def get_device():
    return torch.device(BACKEND_NAME)


def reset_peak_memory_stats():
    """Reset peak memory counters for the active device backend.

    No-op on CPU.  CUDA and XPU expose the same-named function.
    """
    if BACKEND_NAME == _CUDA:
        torch.cuda.reset_peak_memory_stats()
    elif BACKEND_NAME == _XPU:
        torch.xpu.reset_peak_memory_stats()


def get_peak_memory_stats():
    """Return ``(peak_allocated_bytes, peak_reserved_bytes)`` for the local device.

    Returns ``(None, None)`` on CPU or if the backend does not expose the
    relevant APIs.  CUDA and XPU expose identical function names.
    """
    if BACKEND_NAME == _CUDA:
        return (torch.cuda.max_memory_allocated(),
                torch.cuda.max_memory_reserved())
    if BACKEND_NAME == _XPU:
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
    if BACKEND_NAME == _CUDA:
        torch.cuda.synchronize()
    elif BACKEND_NAME == _XPU:
        torch.xpu.synchronize()


def get_device_name(index: int = 0) -> str:
    """Return a human-readable name for the active device.

    CUDA → ``torch.cuda.get_device_name``; XPU → ``torch.xpu.get_device_name``;
    CPU → ``platform.processor()`` (falls back to ``platform.machine()``).
    """
    if BACKEND_NAME == _CUDA:
        return torch.cuda.get_device_name(index)
    if BACKEND_NAME == _XPU:
        try:
            return torch.xpu.get_device_name(index)
        except AttributeError:
            return _XPU
    return platform.processor() or platform.machine() or _CPU


def get_device_info(index: int = 0) -> dict:
    """Return a dict describing the active compute device and host environment.

    Suitable for serialising alongside benchmark results so runs on different
    machines (Spark / Polaris / Aurora / CPU) remain distinguishable.
    """
    info = {
        "backend": BACKEND_NAME,
        "device_name": get_device_name(index),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if BACKEND_NAME == _CUDA:
        info["device_count"] = torch.cuda.device_count()
        try:
            cap = torch.cuda.get_device_capability(index)
            info["cuda_capability"] = f"{cap[0]}.{cap[1]}"
        except Exception:
            pass
        info["cuda_version"] = torch.version.cuda
    elif BACKEND_NAME == _XPU:
        try:
            info["device_count"] = torch.xpu.device_count()
        except AttributeError:
            info["device_count"] = None
    else:
        info["device_count"] = 0
    return info

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
    rank = get_global_rank()
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


_logger = setup_logger(__name__)


def print_memory_usage() -> float:
    """Log and return this Python process's resident-set size in MB.

    Backend-agnostic: measures the host process, not the device. Useful
    on every machine including CPU. Logs via the rank-aware logger so
    only rank 0 prints.
    """
    mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    _logger.info("CPU memory used by this script: %.2f MB", mb)
    return mb


# Pull in the active backend's symbols (DIST_BACKEND, resolve_/set_device_for_local_rank,
# print_gpu_initialization, print_gpu_utilization, etc.). BACKEND_NAME was
# computed near the top of this module.
if BACKEND_NAME == _CUDA:
    from .cuda import *
elif BACKEND_NAME == _XPU:
    from .xpu import *
elif BACKEND_NAME == _CPU:
    from .cpu import *
else:
    raise RuntimeError(f"Unexpected backend name: {BACKEND_NAME}")
