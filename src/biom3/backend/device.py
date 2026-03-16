"""Device detection, selection, and helper utilities

"""

import torch

_CPU = "cpu"
_CUDA = "cuda"
_XPU = "xpu"

def get_backend_name() -> str:
    if torch.cuda.is_available():
        return _CUDA
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return _XPU
    return _CPU

def get_device():
    backend = get_backend_name()
    if backend == _CUDA:
        return torch.device("cuda")
    if backend == _XPU:
        return torch.device("xpu")
    return torch.device(_CPU)

# Set the backend name for import convenience
BACKEND_NAME = get_backend_name()

# Import from device-specific module
if BACKEND_NAME == _CUDA:
    from .cuda import *
elif BACKEND_NAME == _XPU:
    from .xpu import *
elif BACKEND_NAME == _CPU:
    from .cpu import *
else:
    raise RuntimeError(f"Unexpected backend name: {BACKEND_NAME}")
