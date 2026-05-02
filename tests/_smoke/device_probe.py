"""Smoke 2/4: what each rank actually sees on the hardware.

Imports torch (no biom3). Each rank emits one single-line record with:

  - host, pid
  - cpu_affinity (sorted list of core ids the kernel will let this PID run on)
  - ZE_AFFINITY_MASK env var (raw)
  - torch.xpu.is_available()
  - torch.xpu.device_count()
  - torch.xpu.current_device()  (None if no XPU)
  - device_name(0)              (only if device_count > 0)
  - torch.cuda.is_available()   (sanity for non-Aurora runs)
  - torch.cuda.device_count()

Tests the assumption baked into backend.xpu.resolve_device_for_local_rank
that ZE_AFFINITY_MASK is set per-rank by the launcher AND that, given
that, torch.xpu.device_count() is 1 inside each rank.

Usage::

    NGPU_PER_NODE=12 NGPU_TOTAL=24 \\
        ./scripts/launchers/aurora_multinode.sh \\
            python tests/_smoke/device_probe.py
"""

import os
import socket
import sys

import torch


def _xpu_summary():
    has_xpu = hasattr(torch, "xpu")
    if not has_xpu:
        return "xpu_avail=missing", "xpu_count=na", "xpu_current=na", "xpu_dev0=na"
    try:
        avail = torch.xpu.is_available()
    except Exception as e:  # pragma: no cover
        return f"xpu_avail=err({type(e).__name__})", "xpu_count=na", "xpu_current=na", "xpu_dev0=na"
    if not avail:
        return "xpu_avail=False", "xpu_count=0", "xpu_current=na", "xpu_dev0=na"
    try:
        count = torch.xpu.device_count()
    except Exception as e:  # pragma: no cover
        count = f"err({type(e).__name__})"
    try:
        current = torch.xpu.current_device()
    except Exception as e:
        current = f"err({type(e).__name__})"
    if isinstance(count, int) and count > 0:
        try:
            name = torch.xpu.get_device_name(0)
        except Exception as e:
            name = f"err({type(e).__name__})"
    else:
        name = "na"
    return (
        "xpu_avail=True",
        f"xpu_count={count}",
        f"xpu_current={current}",
        f"xpu_dev0={name!r}",
    )


def _cpu_affinity():
    try:
        cores = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as e:
        return f"cpu_affinity=err({type(e).__name__})"
    return f"cpu_affinity={cores}"


def main():
    host = socket.gethostname()
    pid = os.getpid()

    ze_mask = os.environ.get("ZE_AFFINITY_MASK", "<unset>")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")

    parts = [
        f"host={host}",
        f"pid={pid}",
        _cpu_affinity(),
        f"ZE_AFFINITY_MASK={ze_mask}",
        f"CUDA_VISIBLE_DEVICES={cuda_visible}",
    ]
    parts.extend(_xpu_summary())
    parts.append(f"cuda_avail={torch.cuda.is_available()}")
    parts.append(f"cuda_count={torch.cuda.device_count()}")

    sys.stdout.write("[device_probe] " + " ".join(parts) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
