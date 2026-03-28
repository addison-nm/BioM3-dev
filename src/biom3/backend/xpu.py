"""XPU specific functions

"""

import dpctl
import psutil
import os

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

def print_gpu_initialization():
    try:
        q = dpctl.SyclQueue()
        device = q.sycl_device

        logger.info(f"Running on device: {device.name}")
        logger.info(f"Vendor: {device.vendor}")
        logger.info(f"Global memory: {device.global_mem_size // (1024 ** 2)} MB")

        return device.global_mem_size // (1024 ** 2)

    except Exception as e:
        logger.warning(f"Failed to initialize GPU tracking: {e}")
        return None

def print_gpu_utilization():
    logger.info("Intel GPU utilization monitoring is limited at Python level.")

def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_in_bytes = process.memory_info().rss
    memory_in_megabytes = memory_in_bytes / (1024 ** 2)
    logger.info(f"CPU memory used by this script: {memory_in_megabytes:.2f} MB")
    return memory_in_megabytes
