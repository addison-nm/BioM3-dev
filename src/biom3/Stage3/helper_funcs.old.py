from pynvml import *


"""
To track memory allocation, let's take advantage of the nvidia-ml-py3 package and GPU memory allocation from python.

ref: https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one
"""


def print_gpu_initialization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    return info.used // 1024**2


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()










