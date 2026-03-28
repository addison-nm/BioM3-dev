"""CPU functions (fallbacks)

"""

import numpy as np

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

def print_gpu_initialization():
    logger.info("No GPU initialization performed. No GPU available.")
    return np.nan

def print_gpu_utilization():
    logger.info("Cannot determine GPU utilization. No GPU available.")
    return

def print_memory_usage():
    logger.info("Not printing memory usage. No GPU available.")
    return np.nan
