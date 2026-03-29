"""Device detection, selection, and helper utilities

"""

import os
import logging
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

def get_rank() -> int:
    """Get the global rank from environment variables set by the launcher.

    Checks variables from multiple launchers:
    - torchrun / deepspeed: RANK, LOCAL_RANK
    - PALS (Aurora mpiexec): PALS_RANKID, PALS_LOCAL_RANKID
    - PMI (Cray MPICH): PMI_RANK
    - Open MPI: OMPI_COMM_WORLD_RANK

    Returns 0 when running without a launcher (single-process).
    """
    for var in ("RANK", "PALS_RANKID", "PMI_RANK", "OMPI_COMM_WORLD_RANK",
                "LOCAL_RANK", "PALS_LOCAL_RANKID"):
        val = os.environ.get(var)
        if val is not None:
            return int(val)
    return 0


def setup_logger(name: str = "biom3", level: int = logging.INFO) -> logging.Logger:
    """Return a rank-aware logger that only emits on rank 0.

    Non-zero ranks are silenced (set to CRITICAL) so that duplicate
    messages are never printed in multi-node / multi-GPU settings.

    Call once per module::

        from biom3.backend.device import setup_logger
        logger = setup_logger(__name__)
        logger.info("only printed on rank 0")
    """
    logger = logging.getLogger(name)
    logger.propagate = False  # prevent duplicate output via root logger
    rank = get_rank()
    if rank == 0:
        logger.setLevel(level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            ))
            logger.addHandler(handler)
    else:
        logger.setLevel(logging.CRITICAL)
    return logger


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
