import dpctl
import psutil
import os

def print_gpu_initialization():
    try:
        q = dpctl.SyclQueue()
        device = q.sycl_device

        print(f"Running on device: {device.name}")
        print(f"Vendor: {device.vendor}")
        print(f"Global memory: {device.global_mem_size // (1024 ** 2)} MB")

        return device.global_mem_size // (1024 ** 2)

    except Exception as e:
        print(f"Failed to initialize GPU tracking: {e}")
        return None

def print_gpu_utilization():
    print("Intel GPU utilization monitoring is limited at Python level.")

def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_in_bytes = process.memory_info().rss
    memory_in_megabytes = memory_in_bytes / (1024 ** 2)
    print(f"CPU memory used by this script: {memory_in_megabytes:.2f} MB")
    return memory_in_megabytes

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()





















