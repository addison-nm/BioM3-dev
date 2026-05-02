"""XPU specific functions

"""

import os

import dpctl
import torch

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

DIST_BACKEND = "xccl"


def resolve_device_for_local_rank(local_rank: int) -> str:
    """Map local_rank to a per-rank XPU device string.

    On Aurora the launcher's ``--cpu-bind`` and ``ZE_AFFINITY_MASK``
    typically pin one tile per process, so ``torch.xpu.device_count()`` is
    1 inside each rank and the only valid device id is ``xpu:0``. When
    the affinity mask is *not* set (e.g. interactive single-process
    multi-tile testing) we fall back to ``xpu:{local_rank}``.
    """
    if os.environ.get("ZE_AFFINITY_MASK"):
        return "xpu:0"
    try:
        count = torch.xpu.device_count()
    except (AttributeError, RuntimeError):
        count = 1
    if count <= 1:
        return "xpu:0"
    return f"xpu:{local_rank}"


def set_device_for_local_rank(local_rank: int) -> None:
    resolved = resolve_device_for_local_rank(local_rank)
    try:
        torch.xpu.set_device(int(resolved.split(":")[-1]))
    except (AttributeError, ValueError, RuntimeError):
        pass

def print_gpu_initialization():
    try:
        q = dpctl.SyclQueue()
        device = q.sycl_device
        memory_in_megabytes = device.global_mem_size // (1024 ** 2)
        logger.info(f"Running on device: {device.name}")
        logger.info(f"Vendor: {device.vendor}")
        logger.info(f"Global memory: {memory_in_megabytes} MB")
        return memory_in_megabytes
    except Exception as e:
        logger.warning(f"Failed to initialize GPU tracking: {e}")
        return 0.0

def print_gpu_utilization():
    logger.info("Intel GPU utilization monitoring is limited at Python level.")
    return 0.0
