# Distributed Training on ALCF Aurora

Notes on running multi-node PyTorch Lightning + DeepSpeed training on
Intel Data Center GPU Max 1550 (Aurora).

## Hardware topology

Each Aurora node has **6 Intel Data Center GPU Max 1550** GPUs. Each GPU
contains **2 tiles** (stacks). By default, each tile is exposed as a
separate torch device, giving **12 devices per node**.

```
Node
├── GPU 0:  tile 0 (device 0),  tile 1 (device 1)
├── GPU 1:  tile 2 (device 3),  tile 3 (device 3)
├── GPU 2:  tile 4 (device 4),  tile 5 (device 5)
├── GPU 3:  tile 6 (device 6),  tile 7 (device 7)
├── GPU 4:  tile 8 (device 8),  tile 9 (device 9)
└── GPU 5:  tile 10 (device 10), tile 11 (device 11)
```

Training configs should use `gpu_devices: 12` per node.
`ZE_FLAT_DEVICE_HIERARCHY` default already exposes tiles as devices.

## oneCCL (Intel Collective Communications Library)

Aurora uses oneCCL as the distributed backend for `torch.distributed`,
exposed as `"ccl"` or `"xccl"`.

### Known limitations

| Issue | Impact | Workaround |
|-------|--------|------------|
| `ReduceOp.AVG` not supported | RuntimeError on any avg reduce | Use SUM + manual division (patched in custom Lightning `DDPStrategy.reduce`) |
| Integer dtype all-reduce unreliable | `all_reduce(SUM)` on int32/int64 may return wrong values | Cast to float32 before collective, or bypass the all-reduce |

### The `reduce_boolean_decision` bug (April 2026)

Lightning's `ModelCheckpoint` uses `reduce_boolean_decision` to get
unanimous consent from all ranks before saving. This does:

```python
decision = torch.tensor(int(True), device=xpu)   # int64
all_reduce(decision, op=SUM)                       # should be world_size
check: decision == world_size                      # True → save
```

Two things broke this on Aurora:

1. **Custom Lightning DDP patch** (March 2026): Our `DDPStrategy.reduce()`
   workaround for the AVG issue unconditionally divided every reduce by
   `world_size`, turning SUM into MEAN. Fixed April 2026 — the division
   now only applies when the caller requests `"mean"` or `"avg"`.

2. **CCL integer all-reduce** (potential): Even with the DDP fix, oneCCL
   may still mishandle int64 all-reduce. The `SyncSafeModelCheckpoint` in
   `biom3.Stage3.callbacks` bypasses the consensus check entirely (safe
   because `sync_dist=True` ensures all ranks see identical metric values).

### Environment variables

Set automatically by the job scripts or `environment.sh`:

```bash
CCL_OP_SYNC=1                              # synchronous collectives
CCL_PROCESS_LAUNCHER=pmix                  # matches PBS/PALS
ONEAPI_DEVICE_SELECTOR=level_zero:gpu      # expose all GPU tiles
```

## Config for multi-node finetuning

Use a machine override config in `configs/stage3_training/machines/`:

```json
{
  "device": "xpu",
  "precision": "bf16",
  "gpu_devices": 12,
  "num_nodes": 2,
  "log_every_n_steps": 1
}
```

Set `"use_sync_safe_checkpoint": true` if the standard `ModelCheckpoint`
still fails after the DDP reduce fix (indicates a CCL integer all-reduce
issue on the current Aurora software stack).

## ALCF documentation

- [Python setup](https://docs.alcf.anl.gov/aurora/data-science/python/)
- [Programming environment](https://docs.alcf.anl.gov/aurora/aurora-pe/)
- [Running jobs](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/)
- [PyTorch on Aurora](https://docs.alcf.anl.gov/aurora/data-science/frameworks/pytorch/)
