"""Stage3 helper functions

Note: moved device-specific functions (GPU, memory usage, etc.) to backend

"""

from biom3.backend.device import setup_logger, print_gpu_utilization

logger = setup_logger(__name__)

def print_summary(result):
    logger.info("Time: %.2f", result.metrics['train_runtime'])
    logger.info("Samples/second: %.2f", result.metrics['train_samples_per_second'])
    print_gpu_utilization()
