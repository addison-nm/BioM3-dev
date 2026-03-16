"""CPU functions (fallbacks)

"""

import numpy as np

def print_gpu_initialization():
    print("No GPU initialization performed. No GPU available.")
    return np.nan

def print_gpu_utilization():
    print("Cannot determine GPU utilization. No GPU available.")
    return

def print_memory_usage():
    print("Not printing memory usage. No GPU available.")
    return np.nan
