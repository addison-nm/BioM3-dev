# ALCF support ticket draft — Aurora xccl DDP hang in PyTorch backward

**To:** support@alcf.anl.gov
**Subject:** Aurora xccl DDP hang in PyTorch backward (frameworks/2025.3.1, 12-tile single node)

---

## Summary

PyTorch DDP training on a single Aurora node (12 PVC tiles, frameworks/2025.3.1, xccl backend) deterministically deadlocks during the autograd backward pass of the last training batch of an epoch. 1–4 random ranks per trial get stuck inside `_engine_run_backward` (C++ thread spinning at `wchan=0`, MainThread parked on `futex_wait_queue`); the remaining ranks complete backward and pile up at the next downstream synchronizing collective. Reproduction rate is approximately 100% with stock DDP+xccl on this workload.

We have exhausted in-tree workarounds (12+ documented experimental rounds, summarized below) and believe this is consistent with the open issue [intel/torch-xpu-ops#2701](https://github.com/intel/torch-xpu-ops/issues/2701) — the `test_monitored_barrier_allreduce_hang` test currently failing on XCCL on the same PyTorch 2.10 line, with fix targeted for PyTorch 2.12.

We are looking for: (a) confirmation this matches a known issue, (b) any out-of-band patched CCL build or environment workaround we should try, and (c) ETA for the PyTorch 2.12 fix landing in an Aurora frameworks module.

---

## Environment

- **System**: Aurora compute node (Intel Sapphire Rapids ×2 sockets × 52 cores + 6 PVC GPUs × 2 tiles = 12 XPU tiles per node)
- **Module**: `frameworks/2025.3.1`
- **Stack**:
  - torch 2.10.0a0
  - IPEX 2.10.10
  - xccl backend (the new native torch backend; `oneccl_bindings_for_pytorch` no longer used after 2025.3.1)
  - PyTorch Lightning fork at `https://github.com/addison-nm/lightning` (master), installed editable
- **Launcher**: `mpiexec --envall -n 12 --ppn 12 --cpu-bind list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100`
- **Project workspace**: `/flare/NLDesignProtein/ahowe/BioM3-dev-space/BioM3-dev`

---

## Workload

BioM3 Stage 3 — fine-tuning an 86M-parameter conditional diffusion transformer on the SH3pFamonly dataset (~13800 samples, 72 train batches at batch_size=16 per rank, 18 val batches per rank). PyTorch Lightning + plain DDP (`DDPStrategy(process_group_backend='xccl', static_graph=True, gradient_as_bucket_view=True)`), `acc_grad_batches=1`. Two-epoch trial.

---

## Symptom

After ~71 successful train iterations, on the **last train batch of epoch 1 (idx=71)**, autograd's backward never completes on a random subset of 1–4 ranks. Print-instrumented timeline shows:

- **Stragglers** (1–4 random ranks per trial): last printed label = `AFTER_COND_ELBO stage=train idx=71` — meaning they returned from forward + ELBO computation but `loss.backward()` never returns. py-spy MainThread:
  ```
  _engine_run_backward (torch/autograd/graph.py:865)
  backward (torch/autograd/__init__.py:364)
  backward (torch/_tensor.py:630)
  ```
  Per-thread `/proc/<pid>/task/<tid>/wchan` shows one C++ worker thread `R` (running) at `wchan=0` — busy-spin in native code, no Python frames. `/proc/<pid>/status` shows `voluntary_ctxt_switches` not increasing over 30s — true deadlock, not slow execution.

- **Waiters** (the remaining 8–11 ranks): completed backward, completed `on_train_epoch_end`, advanced to `Trainer.save_checkpoint` barrier in [Lightning's trainer.py:1374](https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/trainer/trainer.py#L1374) — call stack `all_reduce` / `barrier` inside `torch.distributed`. py-spy MainThread:
  ```
  barrier (torch/distributed/distributed_c10d.py:...)
  ```

The `Trainer.save_checkpoint` barrier is **not the deadlock cause** — it's the first synchronizing collective downstream of the drift. With validation disabled (`limit_val_batches=0`), the same hang surfaces at `on_advance_end` of batch 70 instead. The actual race is upstream, in DDP's gradient-bucket allreduce inside the backward pass.

---

## Reproducer

Single command on the BioM3 working tree:

```bash
cd /flare/NLDesignProtein/ahowe/BioM3-dev-space/BioM3-dev
bash debug/loop_one_trial.sh round_alcf_ticket
```

This wraps a 2-epoch finetune trial with a 240s timeout and on hang invokes `scripts/diagnose_hang.sh` to capture per-rank `/proc/<pid>/{status,wchan,syscall,task/*/wchan}` and `py-spy dump --pid <pid>` for all 12 ranks. Output lands in `debug/rounds/round_alcf_ticket/{logs,hang_diags}/`.

The diagnostic capture script (`scripts/diagnose_hang.sh`) is sudo-less, gdb-less, and works as a regular user. It is the most reproducible non-intrusive snapshot we can take.

---

## What we've ruled out

| Hypothesis | Rounds | Result |
|---|---|---|
| Default config | R1 | 5/5 hang |
| `DeviceStatsMonitor` callback | R2 | 5/5 hang (removed; no effect) |
| Validation triggers it | R5 | 5/5 hang with `limit_val_batches=0` |
| DeepSpeed Stage 2 strategy | R4 | 2/3 hang (lower rate; same site) |
| Async kernel completion (`xpu.synchronize` after backward) | R9 | hang (straggler stuck *inside* backward, hook never runs on it) |
| `static_graph=True` artifact | (parametric) | hang with both `True` and default; DeepSpeed has no static_graph analog and also hangs |
| Alternate backend `gloo` | R7 | incompatible: `RuntimeError: No backend type associated with device type xpu` |
| Alternate backend `mpi` | R8 | not built into torch in this frameworks module |
| Lightning DDP wrap-order per ALCF docs (CPU-wrap, then move) | R10 | incompatible: `RuntimeError: No backend type associated with device type cpu` at `_verify_param_shape_across_processes` |
| `TORCH_LLM_ALLREDUCE=0` + `CCL_ALLREDUCE=direct` | R11 | converts silent hang into a CCL fast-fail (`oneCCL: allreduce_entry.hpp:38 start: EXCEPTION: ALLREDUCE entry failed. atl_status: FAILURE`) at the same site |
| Switch to a collaborator's known-working PenCL Aurora env (`CCL_PROCESS_LAUNCHER=None`, unset `ONEAPI_DEVICE_SELECTOR`, drop `CCL_ATL_TRANSPORT/CCL_KVS_MODE/CCL_WORKER_AFFINITY/CCL_ATL_SYNC_COLL/FI_MR_CACHE_MONITOR`) | R12 | 1/1 hang with byte-identical signature |

CCL environment we currently run with includes `CCL_PROCESS_LAUNCHER=pmix`, `CCL_ATL_TRANSPORT=mpi`, `CCL_KVS_MODE=mpi`, `FI_MR_CACHE_MONITOR=userfaultfd`, `CCL_ATL_SYNC_COLL=1`, `CCL_OP_SYNC=1` (set by frameworks module), `CCL_WORKER_AFFINITY=8,16,…,100`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `TMPDIR=/tmp`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`. None of these (in any combination tried) prevent the hang.

---

## Likely root cause

The pattern points to a missing CCL completion-event signal during a gradient-bucket allreduce issued from inside autograd's backward C++ path on xccl. Cross-references:

1. [intel/torch-xpu-ops#2701](https://github.com/intel/torch-xpu-ops/issues/2701) — `test_monitored_barrier_allreduce_hang` reproducibly fails on XCCL on the `madhumitha0102/distributed_2.10` branch (same PyTorch 2.10 line). Open, milestone **PT2.12**, assigned to `@syedshahbaaz`. The mechanism (monitored barrier across allreduce path) is the same path our gradient-bucket allreduces traverse.

2. [argonne-lcf/AuroraBugTracking#36](https://github.com/argonne-lcf/AuroraBugTracking/issues/36) — "Occasional interruptible hangs in applications" (QUDA/MILC/GRID/nekRS). Reporter notes attaching gdb makes them resume. Our observation that adding `print(..., flush=True)` calls on the rank's main path probabilistically helps (~33% trial success vs 100% hang baseline) is the same fingerprint — an external-event-driven nudge unblocks a stuck waiter, consistent with a lost wake-up signal.

3. ALCF System Updates (2025-09-08 entry) and ALCF oneCCL guide both document `CCL_OP_SYNC=1` as the workaround for "potential hang at large scale" in `frameworks/2025.3.x`. We have this set; it does not prevent our hang.

4. Intel IPEX known-issues page documents a "race condition between oneDNN kernels and the oneCCL Bindings for PyTorch allreduce primitive" with workaround `TORCH_LLM_ALLREDUCE=0`. We tested this — does not prevent our hang.

---

## Diagnostic artifacts attached

We can attach (or place under `/flare/NLDesignProtein/ahowe/BioM3-dev-space/BioM3-dev/debug/rounds/`):

- One full `hang_diag_*/summary.txt` per representative trial (per-rank `/proc/<pid>/{status,wchan,syscall,task/*/wchan}` plus `py-spy dump` for all 12 ranks)
- Print-instrumented trial logs showing the deadlock site to per-line precision (last `AFTER_COND_ELBO stage=train idx=71` per straggler, last `DDP_BARRIER_BEFORE name=Trainer.save_checkpoint` per waiter)
- `args.json` showing the exact resolved Trainer/strategy configuration
- `scripts/diagnose_hang.sh` (read-only proc capture; ~150 lines, sudo-less)

The full BioM3 codebase including the Lightning fork is at the workspace path above. Read access can be granted to ALCF support engineers on request.

---

## Ask

1. **Confirm or refute** that this matches [intel/torch-xpu-ops#2701](https://github.com/intel/torch-xpu-ops/issues/2701).
2. Any **patched CCL build** or **alternate frameworks module** (e.g., `frameworks/2025.1.X`, since [AuroraBugTracking#36](https://github.com/argonne-lcf/AuroraBugTracking/issues/36) reporter notes "seems working with 2025.1.X") that we can load to bypass.
3. Any **environment-variable workaround** we have not yet tried.
4. **ETA for PT2.12 / fixed XCCL** in an Aurora frameworks module.

In the meantime, we are unblocking by training Stage 3 on Polaris (CUDA) and treating Aurora as Stage-1-only until this is resolved.

---

## Contact

Addison Howe (addison@thenaturalmachine.ai). Working tree at `/flare/NLDesignProtein/ahowe/BioM3-dev-space/BioM3-dev`. Available to provide additional diagnostic captures or grant ALCF support read access to the workspace as needed.
