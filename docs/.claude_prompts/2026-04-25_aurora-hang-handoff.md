# Handoff: BioM3 Stage 3 distributed-training hang on Aurora

**Audience:** an agent picking this up on the Aurora compute node (`aurora-uan*` /
`x4*`) inside `/flare/NLDesignProtein/ahowe/BioM3-dev-space/BioM3-dev`.

**TL;DR:** Stage 3 finetune intermittently hangs at the train→val boundary on
Aurora's 12-tile XPU + xccl + Lightning + DDP stack. We have characterized the
failure thoroughly; the remaining work is either (a) confirm/refute that
`DeviceStatsMonitor` is involved, then (b) instrument Lightning with strategic
prints to pin the exact code line where each rank is parked, then (c) file an
ALCF ticket if neither yields a fix.

---

## 1. Background

**BioM3** is a 3-stage protein-sequence-generation pipeline (Stage 1 PenCL
encoder, Stage 2 Facilitator, Stage 3 ProteoScribe diffusion). Stage 3 is
the GPU-heavy part — an 86M-parameter diffusion transformer trained with
PyTorch Lightning + DDP. The repo is a Python package (`biom3`) installed
editable; entry point `biom3_pretrain_stage3`.

**Environment in use:**

- ALCF Aurora compute nodes — Intel Sapphire Rapids CPU (2 sockets ×
  52 cores) + 6 PVC GPUs × 2 tiles = 12 XPU tiles per node.
- `frameworks/2025.3.1` module: torch 2.10.0a0, IPEX 2.10.10, xccl backend
  (replaced the old `oneccl_bindings_for_pytorch` / `ccl` backend).
- Lightning installed from a fork: `addison-nm/lightning`, currently pinned
  via `pip install git+...@<sha>`. We've patched it (skip pre-allreduce
  barrier in `_sync_ddp` on xccl) and want to make it editable for fast
  iteration — see §6.
- biom3-env is at `venvs/biom3-env/` — a venv with `--system-site-packages`.
- Job launcher abstraction: `scripts/stage{1,2,3}_train_{single,multi}node.sh`
  dispatches to `scripts/launchers/${BIOM3_MACHINE}_{single,multi}node.sh`
  (machine resolved by `environment.sh`). Aurora's launcher uses
  `mpiexec --envall -n 12 --ppn 12 --cpu-bind list:1-8:9-16:...:93-100`.

---

## 2. The problem

Run `bash debug/jobscript.sh` (a 12-rank, single-node Stage 3 finetune trial).
Most trials hang. The hang has a consistent signature:

- Log reaches `Validation DataLoader 0: 18/18` (val *did* complete).
- Then nothing. No further progress for >5 minutes.
- 1–3 ranks ("stragglers") have their Python MainThread parked on a
  futex (`/proc/<pid>/wchan = futex_wait_queue`), with one C++ worker
  thread `R` (running) at `wchan=0` — a busy-spin in native code. py-spy
  shows the MainThread idle inside `torch.autograd._engine_run_backward`.
  The active C++ thread has no Python frames.
- The other 9–11 ranks are all on `MainThread` with stack
  `all_reduce → _sync_ddp → reduce → compute → _store_dataloader_outputs
  → on_advance_end (training_epoch_loop:295)` — i.e., they completed
  training + val, are at val-epoch-end metric aggregation, and are spinning
  in oneCCL's all_reduce waiting for the stragglers to join.
- Across 30-second snapshot intervals the MainThread frames don't move
  and `voluntary_ctxt_switches` doesn't increase: the stragglers truly are
  stuck, not slowly progressing.

Rate is nondeterministic: 0/5 trials hang sometimes, 5/5 other times.
Across many trials the average stragglers/run was 1–3.

**This is a real deadlock**, not slow execution. Something inside the
straggler's autograd C++ engine — most likely a CCL backward-allreduce
hook fired from inside the backward pass, or an XPU kernel completion
event that never signals — never returns. The other ranks moved on to
the next collective; classic mismatched-collective hang on xccl.

---

## 3. What we've tried (and what's currently in the codebase)

In rough chronological order. Things that landed and stayed are noted **(live)**.
Items below the horizontal rule didn't help on their own.

**Strategy / configuration changes that landed and helped:**

- **Switch to `mpiexec` launcher** **(live)**. Originally jobs went through
  Lightning's `subprocess_script.py` launcher; this caused oneCCL to fall
  back to OFI/ATL transport (CCL warning: "did not find MPI-launcher specific
  variables, switch to ATL/OFI"). Switching to mpiexec via the per-machine
  launcher abstraction puts CCL on the ALCF-tested pmix+mpi+mpi path.
- **ALCF-canonical CCL env vars** **(live, in `environment.sh` Aurora branch)**:
  `CCL_PROCESS_LAUNCHER=pmix`, `CCL_ATL_TRANSPORT=mpi`, `CCL_KVS_MODE=mpi`,
  `FI_MR_CACHE_MONITOR=userfaultfd`, `CCL_ATL_SYNC_COLL=1`,
  `CCL_WORKER_AFFINITY=8,16,24,32,40,48,60,68,76,84,92,100` (one progress
  thread per rank, on the last core of each rank's 8-core pin domain),
  `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `TMPDIR=/tmp`.
- **8-cores-per-rank CPU bind** **(live, in launchers)**:
  `--cpu-bind=verbose,list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100`.
  Pairs ranks with adjacent GPUs on the same socket (ranks 0,1 → GPU 0;
  2,3 → GPU 1; etc.). Confirmed correct via per-process `taskset -cp` and
  per-thread `/proc/<pid>/task/<tid>/wchan`.
- **xccl backend explicit** **(live)**: `process_group_backend='xccl'` in
  the strategy. `frameworks/2025.3.1` removed `oneccl_bindings_for_pytorch`
  and `'ccl'` is no longer a valid backend on torch 2.10.
- **Lightning fork patch** `008da352f` **(live)**: skip the redundant
  `torch.distributed.barrier()` before `all_reduce` in
  `_sync_ddp` (`src/lightning/fabric/utilities/distributed.py:222`) on
  xccl only. The barrier was an extra collective per metric; on xccl
  it could release asymmetrically, contributing to drift.
- **biom3 fixes** **(live)**:
  - `PL_wrapper.common_step` no longer calls `self.log(...sync_dist=True)`
    per step on val — only at epoch end (cuts val collectives ~20×).
  - `cond_elbo_objective` uses `self.device` (rank-local) instead of the
    CLI string `--device xpu` (which resolved to xpu:0 on every rank,
    crashing under DDP and silently being papered over by DeepSpeed).
- **DDPStrategy(static_graph=True, gradient_as_bucket_view=True)**
  **(live)**: tells DDP the autograd graph is stable, so it precomputes
  bucket-readiness order at iteration 0 instead of recomputing each
  backward. **Reduced** hang frequency but did **not eliminate** it.

---

**Things tried that did NOT help on their own:**

- **DeepSpeedStrategy** (ZeRO stage 2, with `overlap_comm=False`) — produced
  the same hang shape, same code path. py-spy confirmed stragglers in
  `deepspeed/runtime/zero/stage_1_and_2.py::allreduce_bucket`. DeepSpeed's
  bucket scheduler doesn't honor `static_graph` and has its own race on
  xccl. Reverted to DDP.
- `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=16384` — raising IPC handle
  cache 16× had no effect on hang rate.
- `gpu_tile_compact.sh` (ALCF GPU pinning helper) — incompatible with
  `ZE_FLAT_DEVICE_HIERARCHY=FLAT` (which the frameworks module sets);
  ALCF docs say to bind in app instead, which Lightning does via
  `torch.xpu.set_device(local_rank)`. Confirmed correct via per-rank UUID
  diagnostic.
- Adding `on_train_batch_end → torch.distributed.barrier()` per step:
  added more collectives, did not help. Reverted.

---

**Diagnostic tools that work and are in the repo:**

- **`scripts/diagnose_hang.sh`** [Path: `scripts/diagnose_hang.sh`]
  Sudo-less, gdb-less. Captures per-process `/proc/<pid>/{status,wchan,syscall}`
  twice (T=0 and T+30s) for every biom3 rank, plus per-thread state, plus
  `py-spy dump --pid <pid>` for each rank. Run from a second shell on
  the same compute node while a job is hung. Outputs to a `hang_diag_<ts>/`
  directory; the `summary.txt` is the most informative file. Detailed reading
  guide in [docs/debug_aurora.md](docs/debug_aurora.md).
- **`docs/debug_aurora.md`** — full debug cookbook: how to capture stacks,
  verify CPU bind, identify common stack patterns, distinguish hangs vs
  stalls.

---

## 4. Hypotheses still on the table

In approximate priority for testing:

1. **`DeviceStatsMonitor` callback perturbing autograd timing.** It runs
   per train batch, querying Level Zero for XPU device stats. Those L0
   calls enqueue against the same command queue as autograd's backward
   kernels. On a busy queue, a poorly-timed stats query could delay a
   gradient kernel completion event; the autograd worker thread blocks
   on the event; that's the C++ "active" thread we see spinning. **This
   is the cheapest test we haven't run.** [§5.3]

2. **A specific Lightning code path holds a futex during a collective.**
   The Python MainThread on stragglers is on `futex_wait_queue`. Some
   userspace lock isn't being released. Most likely candidates: torchmetrics'
   internal sync state, Lightning's DDP forwarding wrapper, or a
   logger thread holding a lock that the autograd hook also wants.
   Strategic `print(..., flush=True)` instrumentation in Lightning will
   pin the line. [§5.4]

3. **A oneCCL bug at the xccl C++ level.** If neither (1) nor (2) yields,
   we're past what biom3/Lightning can fix. The C++ active thread on
   stragglers is in oneCCL or autograd's XPU kernel execution; native
   stacks would be definitive but require gdb (we don't have sudo). At
   that point, file an ALCF ticket with our diagnostic bundle. [§5.5]

---

## 5. Plan of action

### 5.1 Make Lightning editable on Aurora (one-time)

Currently Lightning is pinned via `pip install git+https://github.com/addison-nm/lightning.git@<sha>`,
which means every change to Lightning requires `pip install --force-reinstall`.
Make it editable so any edit to `lightning/src/lightning/...` takes effect
on the next run with no reinstall:

```bash
cd /flare/NLDesignProtein/ahowe/BioM3-dev-space
[ -d lightning ] || git clone https://github.com/addison-nm/lightning.git
cd lightning
git checkout master
git pull
pip install --force-reinstall --no-deps -e .

# verify — should print a path INSIDE this checkout, not site-packages:
python -c "import lightning; print(lightning.__file__)"
```

### 5.2 Branch for experimentation

Don't muck up `addison-dev`. Create a scratch branch:

```bash
cd /flare/NLDesignProtein/ahowe/BioM3-dev-space/BioM3-dev
git fetch origin
git checkout -b addison-aurora-debug origin/addison-dev
# work freely; commit small as you go:
git add -A && git commit -m 'WIP: <what you tried>'
git push origin addison-aurora-debug
```

Same pattern in the Lightning checkout if you edit there.

### 5.3 Test #1: comment out `DeviceStatsMonitor`

In `src/biom3/Stage3/run_PL_training.py`, find where `DeviceStatsMonitor`
is added to the callbacks list (`grep -n DeviceStatsMonitor src/biom3/Stage3/run_PL_training.py`)
and comment it out:

```python
# DIAGNOSTIC: temporarily disabled to test whether per-step XPU stat
# queries are perturbing autograd backward kernel timing on Aurora xccl.
# callbacks.append(DeviceStatsMonitor())
```

Then run 5 trials with the loop runner:

```bash
for i in {1..5}; do bash debug/loop_one_trial.sh; done

# tally results:
cat debug/logs/result_*.txt | sort | uniq -c
# look for: "5 EXIT_0" (all ran clean)  vs  "3 HANG / 2 EXIT_0" (still flaky)
```

If hang count goes to 0/5 — committed, that's the answer; explain in commit
message and move `DeviceStatsMonitor` out of the per-step path or remove it.

If hang count is still ≥ 1/5 — re-enable `DeviceStatsMonitor`, move to §5.4.

### 5.4 Test #2: strategic `print` instrumentation in Lightning + biom3

Goal: when a trial hangs, the **last line each rank printed** tells us
exactly which call site is the deadlock point. py-spy gave us a snapshot
but the timeline `print` statements give is much more useful for pinning a
specific line.

**Print pattern** (use everywhere):

```python
import os, time
_R = int(os.environ.get('PALS_LOCAL_RANKID', os.environ.get('RANK', '?')))
print(f"[r{_R} {time.time():.3f}] LABEL extra=info", flush=True)
```

Add identical-format prints (rank + timestamp + LABEL) at these 6 points.
Pick a unique LABEL for each so you can grep:

| File | Function | LABEL location |
|---|---|---|
| Lightning fork: `src/lightning/fabric/utilities/distributed.py` | `_sync_ddp` | first line of body, plus immediately before `torch.distributed.all_reduce(...)`, plus immediately after |
| Lightning fork: `src/lightning/pytorch/strategies/ddp.py` | `DDPStrategy.reduce` | first line of body |
| Lightning fork: `src/lightning/pytorch/loops/training_epoch_loop.py` | `advance` and `on_advance_end` | first line of each |
| Lightning fork: `src/lightning/pytorch/loops/evaluation_loop.py` | `_store_dataloader_outputs` | first line |
| biom3: `src/biom3/Stage3/PL_wrapper.py` | `common_step` | top, plus immediately before and after the `cond_elbo_objective(...)` call |
| biom3: `src/biom3/Stage3/PL_wrapper.py` | `cond_elbo_objective` | top, plus immediately before `self(x=...)` (the model forward), plus immediately after |

Then run a trial. When it hangs, find the last LABEL each rank printed:

```bash
# extract last 5 LABELs per rank for the most recent log:
LATEST=$(ls -t debug/logs/trial_*.log | head -1)
for r in 0 1 2 3 4 5 6 7 8 9 10 11; do
  echo "=== rank ${r} last 5 prints ==="
  grep "^\[r${r} " "${LATEST}" | tail -5
done
```

That output, combined with a diagnose_hang.sh capture from the same run,
is enough to pinpoint the exact line and write a fix.

### 5.5 Escalate to ALCF if §5.3 and §5.4 don't yield a fix

Submit a ticket to [support@alcf.anl.gov](mailto:support@alcf.anl.gov) with:

- A description of the workload (Stage 3 PyTorch Lightning DDP, 12 XPU tiles,
  86M-param diffusion transformer).
- The exact `frameworks` module version (`module list | grep frameworks`).
- A clean log from a hung trial.
- One full `hang_diag_*/` artifact (the per-thread state and py-spy dumps
  give them a complete reproducer without needing the workload).
- A summary of what we've ruled out (this document).
- The hypothesis: oneCCL or autograd-XPU executor is failing to signal an
  event from inside a backward-pass async collective, leaving the autograd
  worker thread spinning indefinitely.

---

## 6. Reference: the iteration loop

You have two scripts in `debug/` that automate trials:

**`debug/jobscript.sh`** — one job, configured for a 2-epoch sanity finetune
of Stage 3 on the SH3pFamonly dataset. Already wired to dispatch through
`scripts/stage3_train_singlenode.sh` → `scripts/launchers/aurora_singlenode.sh`
(mpiexec + ALCF CPU bind).

**`debug/loop_one_trial.sh`** — runs `jobscript.sh` once with a 600-second
timeout, captures live output (via `script -qfec` for pseudo-TTY; needed
because tqdm doesn't render through plain redirect), and on hang
automatically runs `scripts/diagnose_hang.sh` to capture diagnostic
artifacts. Writes `debug/logs/trial_<ts>.log` (full output) and
`debug/logs/result_<ts>.txt` (`HANG` or `EXIT_<n>`). Diagnostic artifacts
land in `debug/hang_diags/hang_diag_<ts>/`.

Usage:

```bash
# one trial:
bash debug/loop_one_trial.sh

# 5 trials in a row:
for i in {1..5}; do bash debug/loop_one_trial.sh; done

# follow live (in another shell):
tail -F debug/logs/trial_*.log

# tally:
cat debug/logs/result_*.txt | sort | uniq -c
```

If a trial's `result` file says `HANG`, the corresponding
`debug/hang_diags/hang_diag_<ts>/summary.txt` has the per-rank state at
hang time. Reading guide in `docs/debug_aurora.md`.

---

## 7. Files to know

```
BioM3-dev/
├── docs/
│   ├── debug_aurora.md                              # the debug cookbook
│   ├── aurora_distributed_training.md
│   └── .claude_prompts/2026-04-25_aurora-hang-handoff.md   # this file
├── debug/
│   ├── jobscript.sh                                 # the trial config
│   ├── loop_one_trial.sh                            # the trial runner
│   ├── logs/                                        # trial logs + result_*.txt
│   └── hang_diags/                                  # diagnostic dirs (gitignored?)
├── environment.sh                                   # ⚠ DO NOT EDIT — exports CCL env, BIOM3_MACHINE
├── scripts/
│   ├── stage{1,2,3}_train_{single,multi}node.sh    # train wrappers
│   ├── launchers/aurora_{single,multi}node.sh      # mpiexec + cpu-bind
│   └── diagnose_hang.sh                             # /proc + py-spy capture
├── src/biom3/Stage3/
│   ├── PL_wrapper.py                                # PL_ProtARDM, common_step
│   └── run_PL_training.py                           # DDPStrategy / DeepSpeedStrategy wiring
└── venvs/biom3-env/                                 # the active venv (--system-site-packages)
```

Lightning fork (separately versioned, also editable per §5.1):

```
~/BioM3-dev-space/lightning/      (or wherever it's cloned on Aurora)
└── src/lightning/
    ├── fabric/utilities/distributed.py              # patched _sync_ddp at line ~222
    └── pytorch/strategies/{ddp.py, deepspeed.py}    # DDPStrategy, DeepSpeedStrategy
```

---

## 8. Constraints / things to know

- **`environment.sh` is off-limits for edits.** The user explicitly
  reserved that file. If you need an env var, set it in
  `debug/jobscript.sh` (local to the trial) or in your shell.
- **No sudo, no gdb.** All diagnostics must work as a regular user.
  `/proc/<pid>/{status,wchan,syscall,task/*/wchan}` and `py-spy` are the
  only tools available. The `scripts/diagnose_hang.sh` script already wraps
  what works.
- **Don't push to `main` or `addison-dev`** without explicit user approval.
  Push to `addison-aurora-debug` or another scratch branch you create.
  The user will merge.
- **Don't modify `scripts/launchers/aurora_*node.sh`** unless you've
  validated the change reproduces ALCF-canonical CPU bind via `taskset -cp`
  on a live job. The bind layout is load-bearing for separating framework
  threads from CCL progress threads.
- **A single clean trial is not evidence of a fix.** The failure is
  nondeterministic. Run at least 5 trials before concluding anything.

---

## 9. Suggested first action

Start with §5.1 (make Lightning editable, ~30 seconds), then §5.2 (branch),
then §5.3 (disable `DeviceStatsMonitor`, run 5 trials). If that doesn't
clear the hang, proceed to §5.4 (Lightning instrumentation). Report back
with the result tally and a `hang_diag_*/summary.txt` from the most recent
hung trial if applicable.
