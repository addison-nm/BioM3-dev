"""Smoke 4/4: end-to-end init_process_group + collectives.

Calls the production ``init_distributed_if_launched`` (which on Aurora
selects ``xccl``, on CUDA hosts ``nccl``, otherwise ``gloo``), then runs
three collectives:

  - barrier
  - all_reduce on a [rank] tensor (sum should equal world*(world-1)/2)
  - broadcast_object_list of "hello-from-{rank}" with src=0
    (every rank should observe "hello-from-0")

Each rank prints PASS/FAIL per step. The script intentionally has no
tolerance for hangs - if init_process_group blocks, that's the answer.
Wrap the launcher in ``timeout 60`` if you want a hard ceiling.

Usage (Aurora multinode example)::

    NGPU_PER_NODE=12 NGPU_TOTAL=24 \\
        ./scripts/launchers/aurora_multinode.sh \\
            python tests/_smoke/pg_probe.py

Optional flag:
    --device {xpu,cuda,cpu}   force the requested device (default: '' =
                              let the active backend decide)
"""

import argparse
import os
import socket
import sys


def _emit(parts):
    sys.stdout.write("[pg_probe] " + " ".join(str(p) for p in parts) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="", help="requested device str")
    args = ap.parse_args()

    host = socket.gethostname()
    pid = os.getpid()
    base = [f"host={host}", f"pid={pid}"]

    try:
        import torch
        import torch.distributed as dist
        from biom3.core.distributed import init_distributed_if_launched
        from biom3.backend.device import BACKEND_NAME, DIST_BACKEND
    except Exception as e:
        _emit(base + [f"import_error={type(e).__name__}:{e}"])
        return 2

    _emit(base + [
        f"backend={BACKEND_NAME}",
        f"dist_backend={DIST_BACKEND}",
        f"requested_device={args.device or '<none>'}",
        "stage=before_init",
    ])

    try:
        rank, local_rank, world_size, dev = init_distributed_if_launched(args.device)
    except Exception as e:
        _emit(base + [f"init_error={type(e).__name__}:{e}"])
        return 3

    _emit(base + [
        f"rank={rank}",
        f"local_rank={local_rank}",
        f"world={world_size}",
        f"resolved_device={dev}",
        f"is_initialized={dist.is_available() and dist.is_initialized()}",
        "stage=after_init",
    ])

    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        _emit(base + [f"rank={rank}", "result=skipped_singleproc"])
        return 0

    # --- barrier ---
    try:
        dist.barrier()
        _emit(base + [f"rank={rank}", "step=barrier", "result=PASS"])
    except Exception as e:
        _emit(base + [f"rank={rank}", "step=barrier",
                      f"result=FAIL:{type(e).__name__}:{e}"])

    # --- all_reduce on cpu tensor (gloo/xccl/nccl all support cpu/gpu;
    #     pick a placement matching the resolved device when possible) ---
    placement = "cpu"
    try:
        if dev != "cpu" and dev.startswith(("xpu", "cuda")):
            placement = dev
    except AttributeError:
        pass
    try:
        t = torch.tensor([rank], dtype=torch.long)
        if placement != "cpu":
            t = t.to(placement)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        observed = int(t.detach().to("cpu").item())
        expected = world_size * (world_size - 1) // 2
        ok = observed == expected
        _emit(base + [
            f"rank={rank}", "step=all_reduce",
            f"placement={placement}",
            f"observed={observed}", f"expected={expected}",
            f"result={'PASS' if ok else 'FAIL'}",
        ])
    except Exception as e:
        _emit(base + [
            f"rank={rank}", "step=all_reduce",
            f"placement={placement}",
            f"result=FAIL:{type(e).__name__}:{e}",
        ])

    # --- broadcast_object_list ---
    try:
        payload = [f"hello-from-{rank}"] if rank == 0 else [None]
        dist.broadcast_object_list(payload, src=0)
        ok = payload[0] == "hello-from-0"
        _emit(base + [
            f"rank={rank}", "step=broadcast_object_list",
            f"observed={payload[0]!r}",
            f"result={'PASS' if ok else 'FAIL'}",
        ])
    except Exception as e:
        _emit(base + [
            f"rank={rank}", "step=broadcast_object_list",
            f"result=FAIL:{type(e).__name__}:{e}",
        ])

    # Hold all ranks until rank 0 finishes printing
    try:
        dist.barrier()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
