"""
To track memory allocation, let's take advantage of the nvidia-ml-py3 package and GPU memory allocation from python.

ref: https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one
"""

import numpy as np
import pynvml

def nvml_available() -> bool:
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return True
    except pynvml.NVMLError:
        return False

def print_gpu_initialization():
    if nvml_available():
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print(f"GPU memory occupied: {info.used//1024**2} MB.")
        return info.used // 1024**2
    return np.nan


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()










