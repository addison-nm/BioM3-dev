"""CUDA specific functions

To track memory allocation, let's take advantage of the nvidia-ml-py3 package
and GPU memory allocation from python.
ref: https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one

"""

import pynvml
import torch

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

DIST_BACKEND = "nccl"


def resolve_device_for_local_rank(local_rank: int) -> str:
    return f"cuda:{local_rank}"


def set_device_for_local_rank(local_rank: int) -> None:
    torch.cuda.set_device(local_rank)

def print_gpu_initialization():
    if _nvml_available():
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used // 1024**2
    return float("nan")

def print_gpu_utilization():
    pass

########################
##  Helper functions  ##
########################

def _nvml_available() -> bool:
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return True
    except pynvml.NVMLError:
        return False
