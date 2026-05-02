# 2026-05-02 — Audit of `core.distributed` / `backend`, Aurora launcher consistency, smoke-probe harness

Continuation of the multi-node thread from
[2026-04-30_stage3_multinode_sampling.md](2026-04-30_stage3_multinode_sampling.md).
That session introduced `src/biom3/core/distributed.py` as the launcher /
rendezvous helper for the new Stage 3 sampling pipeline. This session
audits the resulting code for drift, fixes empirically-discovered
launcher inconsistencies, consolidates per-backend device specifics,
and adds a hand-run smoke-probe harness for verifying Aurora's
distributed plumbing without needing a full Stage 3 run.

Branch: `addison-hack-grpo`. Three code commits + this note.

## Headline changes

### 1. Aurora launcher consistency

`aurora_singlenode.sh` had `--gpu-bind=list:0.0:0.1:...:5.1` (added in
`f5edb82`); `aurora_multinode.sh` had no `--gpu-bind` at all, and used
the slightly different `--cpu-bind verbose,list:...` form (space + the
`verbose,` prefix) instead of `--cpu-bind=list:...`. Three bugs of
omission. Both launchers now share:

- `CPU_BIND_SCHEME="--cpu-bind=list:1-8:9-16:...:93-100"` (equals form,
  consistent across files)
- `GPU_BIND_SCHEME="--gpu-bind=list:0:1:2:3:4:5:6:7:8:9:10:11"` (one
  entry per tile, FLAT-mode addressing — see §4 below)

### 2. `backend.device.get_rank()` was buggy; rank detection moved to a leaf module

The old `get_rank()` scanned `("RANK", "PALS_RANKID", "PMI_RANK",
"OMPI_COMM_WORLD_RANK", "LOCAL_RANK", "PALS_LOCAL_RANKID")` in *one
flat list*. If `LOCAL_RANK` was set but no global-rank var was set,
it returned the local rank as if it were global. Masked on Aurora
because PALS always sets `PALS_RANKID` first, but a real footgun for
torchrun-style launches with `LOCAL_RANK` set on a single rank.

Fix: introduced `src/biom3/core/_dist_env.py` as a pure-stdlib leaf
module owning the env-var scan (`get_global_rank`, `get_local_rank`,
`get_world_size`, `is_launched`) with cleanly separated tuples. Both
`backend.device.setup_logger` and `core.distributed` import from it,
breaking the structural import cycle (`backend.device` finishes its
load with `from .{cpu,cuda,xpu} import *`, which calls
`setup_logger(__name__)` mid-load — a deferred import inside
`setup_logger` doesn't fix this because the circular ImportError
fires from the import-star chain, not the call site). Public API
unchanged: `core.distributed` re-exports the three rank functions.

`backend.device.get_rank()` deleted. Callers in
`Stage{1,2,3}/run_PL_training.py` and `core/run_utils.py` updated to
import `get_global_rank` from `core.distributed`.

### 3. Per-backend device resolution moved into `backend/{cpu,cuda,xpu}.py`

`core.distributed._resolve_device_str` previously hard-coded
`"cuda:{local_rank}"` for CUDA and a `ZE_AFFINITY_MASK`-aware sentinel
for XPU. Each backend module now exports `DIST_BACKEND` (the
`init_process_group` backend name: `nccl` / `xccl` / `gloo`),
`resolve_device_for_local_rank(local_rank)`, and
`set_device_for_local_rank(local_rank)`. `core.distributed` delegates
via the active-backend symbols pulled in through
`backend.device`'s existing `from .{backend} import *`. Net effect on
`core/distributed.py`: ~95 lines removed, no behaviour change in the
production case.

The XPU branch retained the `ZE_AFFINITY_MASK` / `device_count <= 1`
sentinel. The CUDA branch is currently naive — returns
`f"cuda:{local_rank}"` unconditionally. Latent gotcha: if Polaris
launchers ever start setting `CUDA_VISIBLE_DEVICES` per rank (none do
today), this will crash for `local_rank >= 1` because each rank only
sees one GPU. Tracked as future work.

### 4. Aurora `ZE_FLAT_DEVICE_HIERARCHY` mode discovery

Initial assumption was that the launcher's `0.0:0.1:...` GPU bind
syntax should work on `frameworks/2025.3.1`. Empirically, it does not:

```
launch failed on x4309c4s0b0n0: Requested GPU 0.0 in FLAT mode,
try setting ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
```

`palsd` parses `--gpu-bind` against its own pre-fork view of the
device hierarchy, which on this frameworks module is FLAT. A
*post-job* `export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE` does **not**
reach palsd — only the rank's env after fork. So setting the var in
the user shell or even in `environment.sh` after job allocation
doesn't help.

Two correctness paths: (a) FLAT-mode bind syntax (`0:1:2:...:11`),
(b) export `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE` *before* the job
launches (i.e. in `.bash_profile` or in a wrapper that runs before
`qsub`). Chose (a). Both launchers now use FLAT.

Also confirmed empirically via the new `device_probe.py`:

- FLAT mode → `torch.xpu.device_count() == 12` per unbound process
- COMPOSITE mode → `torch.xpu.device_count() == 6`

Both have `ZE_AFFINITY_MASK=<unset>` from the user shell (PALS sets
this per rank inside `mpiexec --gpu-bind`, not via login env).

### 5. `_set_master_addr_port` simplified

Was `MASTER_PORT = 29500 + blake2b($PBS_JOBID) mod 1000` with the
rationale "so two concurrent jobs on the same login node don't
collide on port 29500" — a scenario that doesn't actually occur,
because PBS allocates compute nodes exclusively per job and
`MASTER_ADDR` derives from `$PBS_NODEFILE`'s first line (a compute
node, not a login node). Two concurrent jobs sit on disjoint nodes
with disjoint `MASTER_ADDR`s and cannot collide on the same port.

Replaced with `os.environ.setdefault("MASTER_PORT", "29500")`. This
is torch.distributed's documented default and what ALCF examples use.
Also removed `import hashlib`. `MASTER_ADDR` derivation from
`$PBS_NODEFILE` is unchanged — that part *is* genuinely needed
because PALS doesn't surface `MASTER_ADDR`. Function docstring
updated to clarify it's PBS-applicable (Aurora *and* Polaris), not
Aurora-specific.

### 6. `backend/` module simplifications

Independent of the audit but discovered while reviewing:

- `BACKEND_NAME = get_backend_name()` hoisted to module top in
  [device.py](../src/biom3/backend/device.py); six helpers below it
  switched from per-call `get_backend_name()` to direct constant
  reference. `get_device()` collapsed to `torch.device(BACKEND_NAME)`.
- `print_memory_usage` was duplicated byte-identically across
  `cuda.py` and `xpu.py` (and stubbed on `cpu.py`). Backend-agnostic
  by nature (measures host RSS via `psutil`). Lifted to `device.py`
  as a single shared definition.
- `np.nan` → `float("nan")` in `cpu.py` and `cuda.py`; dropped
  `import numpy as np`. Dropped now-unused `import os` and
  `import psutil` from `cuda.py`; dropped unused `import psutil`
  from `xpu.py`.
- Replaced 11 lines of commented-out dead code in `backend/__init__.py`
  with a docstring pointing at `backend.device`.
- Net: ~18 lines smaller across the five backend files; no public
  API change.

### 7. Smoke-probe harness — `tests/_smoke/`

New (not pytest):

- [env_dump.py](../tests/_smoke/env_dump.py) — pure stdlib, no biom3
  imports. Per-rank one-line dump of every distributed-related env
  var. `-l/--multiline` emits one var per line, prefixed with a
  best-effort `rank=N` tag.
- [device_probe.py](../tests/_smoke/device_probe.py) — torch only,
  no biom3. Per-rank report of `os.sched_getaffinity`,
  `ZE_AFFINITY_MASK`, `torch.{xpu,cuda}.is_available/device_count/
  current_device`, device name. Tests the audit's `count <= 1` /
  pinned-tile assumption.
- [rank_audit.py](../tests/_smoke/rank_audit.py) — independent raw
  env-var scan vs. `biom3.core._dist_env`'s answers. Per-rank
  `agree=True/False` consistency check.
- [pg_probe.py](../tests/_smoke/pg_probe.py) — calls the production
  `init_distributed_if_launched`, then runs `barrier`,
  `all_reduce` (sum check on `[rank]` tensor),
  `broadcast_object_list` from rank 0. End-to-end xccl/nccl
  rendezvous + collective validation. Wrap in `timeout 120` to
  avoid trapping the shell on hang.
- [run_aurora_singlenode.sh](../tests/_smoke/run_aurora_singlenode.sh)
  — 18-line bash harness that calls `mpiexec` directly (not via
  the production launcher) so config can be tweaked in-place. Three
  knobs: `CPU_BIND`, `GPU_BIND`, `ZE_FLAT_DEVICE_HIERARCHY`.

## Empirical validation

Multinode run on a 2-node Aurora interactive allocation, sweeping
`NGPU_PER_NODE ∈ {1..12}` (24 invocations: 12 not-sourced + 12 sourced
`environment.sh`). Probe outputs in `output{1..12}.txt` and
`output_sourced{1..12}.txt` (not committed; deleted after analysis).

What was confirmed:

- PALS sets `PALS_RANKID`, `PALS_LOCAL_RANKID`, `PALS_LOCAL_SIZE`,
  `PALS_NODEID`, `PALS_DEPTH`. Notably does **not** surface
  `PMI_RANK`, `PMI_SIZE`, `PALS_NUM_NODES`, `PALS_PPN`, `RANK`,
  `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`,
  `OMPI_COMM_WORLD_*`. The `_dist_env` tuple ordering (`PALS_RANKID`
  second after `RANK`) is correct.
- World-size derivation `PALS_LOCAL_SIZE × len($PBS_NODEFILE)`
  matches actual world on every NGPU value.
- `rank_audit.py`'s independent raw scan and `_dist_env` answers
  `agree=True` on every rank.
- `device_probe.py`: `xpu_count=1` per rank in launcher-pinned
  context, `ZE_AFFINITY_MASK` populated correctly per local rank,
  `cpu_affinity` matches the local rank's `--cpu-bind` slot.
- `pg_probe.py`: `barrier`, `all_reduce` (sums correct for world
  ∈ {2, 4, ..., 24}), `broadcast_object_list` all PASS once past
  the port issue described below.

What was *not* an audit issue:

- `output1..7.txt` (NGPU=1..7, not sourced) all failed with
  `EADDRINUSE` on `port: 29663` (the old hashed port). Output 8+
  succeeded. The transition was wall-clock-driven (TIME_WAIT or a
  lingering process from the earlier hung NGPU=24 invocation),
  *not* sourcing-driven. The sourced batch all passed because it
  ran 2-4 minutes later, after the squatter cleaned up. This was
  the prompt to remove the hash and standardize on port 29500 (§5).
- `environment.sh` Aurora branch sets several CCL/ONEAPI/FI env
  vars that env_dump doesn't track (CCL_PROCESS_LAUNCHER,
  CCL_ATL_TRANSPORT, ONEAPI_DEVICE_SELECTOR, etc.). These genuinely
  matter for production xccl behaviour, but did not cause or
  resolve the port collision. Worth keeping `source environment.sh`
  in production launches anyway.

## Polaris portability — what to expect

The audit changes are designed to work on Polaris's NCCL+CUDA stack
without modification, but were untested in this session. Things that
should work as-is: `_dist_env` tuple ordering (Polaris also uses
PALS), `_set_master_addr_port` (PBS-applicable, not Aurora-specific),
`backend.cuda.{resolve,set}_device_for_local_rank`,
`init_process_group(backend="nccl")` rendezvous via PBS-derived
`MASTER_ADDR`. Things to verify with the smoke harness on a Polaris
allocation: NCCL bootstrap interface selection (may need
`NCCL_SOCKET_IFNAME=hsn0`), per-rank `cuda_count` under the current
no-`--gpu-bind` Polaris launcher (should be 4, not 1).

Latent gotcha: if Polaris launchers ever add `--gpu-bind`,
`backend.cuda.resolve_device_for_local_rank` needs a
`CUDA_VISIBLE_DEVICES` sentinel mirroring XPU's pattern.

## Files touched

```
M  scripts/launchers/aurora_singlenode.sh
M  scripts/launchers/aurora_multinode.sh
M  src/biom3/backend/__init__.py
M  src/biom3/backend/device.py
M  src/biom3/backend/cpu.py
M  src/biom3/backend/cuda.py
M  src/biom3/backend/xpu.py
A  src/biom3/core/_dist_env.py
M  src/biom3/core/distributed.py
M  src/biom3/core/run_utils.py
M  src/biom3/Stage1/run_PL_training.py
M  src/biom3/Stage2/run_PL_training.py
M  src/biom3/Stage3/run_PL_training.py
A  tests/_smoke/__init__.py
A  tests/_smoke/env_dump.py
A  tests/_smoke/device_probe.py
A  tests/_smoke/rank_audit.py
A  tests/_smoke/pg_probe.py
A  tests/_smoke/run_aurora_singlenode.sh
```
