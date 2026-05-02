"""CPU functions (fallbacks)

"""

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

DIST_BACKEND = "gloo"


def resolve_device_for_local_rank(local_rank: int) -> str:
    return "cpu"


def set_device_for_local_rank(local_rank: int) -> None:
    pass

def print_gpu_initialization():
    logger.info("No GPU initialization performed. No GPU available.")
    return float("nan")

def print_gpu_utilization():
    logger.info("Cannot determine GPU utilization. No GPU available.")
    return
