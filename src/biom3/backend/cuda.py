"""CUDA specific functions

To track memory allocation, let's take advantage of the nvidia-ml-py3 package 
and GPU memory allocation from python.
ref: https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one

"""

import numpy as np
import pynvml

def print_gpu_initialization():
    if _nvml_available():
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print(f"GPU memory occupied: {info.used//1024**2} MB.")
        return info.used // 1024**2
    return np.nan

def print_gpu_utilization():
    pass

def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_in_bytes = process.memory_info().rss
    memory_in_megabytes = memory_in_bytes / (1024 ** 2)
    print(f"CPU memory used by this script: {memory_in_megabytes:.2f} MB")
    return memory_in_megabytes

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
