# Debugging Aurora distributed training

Snippets for diagnosing hangs, performance variance, and collective issues in
distributed PyTorch + Lightning + xccl jobs on ALCF Aurora. Scoped to Aurora
(XPU / oneCCL / frameworks/2025.3.1). All commands assume the `biom3-env`
venv is active and the job was launched via one of the
`scripts/stage*_train_*.sh` wrappers, which dispatch to
`scripts/launchers/aurora_{single,multi}node.sh` (which run `mpiexec` with
ALCF-canonical CPU binding).

## Configuration status (2026-04-24)

**The earlier "known-good" claim for DDP + `static_graph=True` was
premature.** A subsequent trial under that exact configuration hung at
the *end* of validation (Validation DataLoader 0: 18/18 reached, no
forward progress after) — a different boundary than the
last-training-step hang `static_graph=True` was theorized to fix. Both
DDPStrategy+static_graph and DeepSpeedStrategy currently exhibit
intermittent hangs; neither is verified clean over a long run.

What **does** appear to be load-bearing for *getting past most boundaries
some of the time*:

- **Launch:** `scripts/stage3_train_singlenode.sh` →
  `scripts/launchers/aurora_singlenode.sh` (mpiexec --envall, ALCF
  8-cores-per-rank `--cpu-bind` list). Without mpiexec, CCL silently
  downgrades to OFI/ATL transport and hangs on the first real validation.
- **CCL env (`environment.sh`):** `CCL_PROCESS_LAUNCHER=pmix`,
  `CCL_ATL_TRANSPORT=mpi`, `CCL_KVS_MODE=mpi`, `FI_MR_CACHE_MONITOR=userfaultfd`,
  `CCL_ATL_SYNC_COLL=1`, `CCL_WORKER_AFFINITY=8,16,...,100`,
  `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `TMPDIR=/tmp`.
- **Lightning fork:** `_sync_ddp` skips its pre-`all_reduce` barrier on xccl
  (commit `008da352f`). Without this, jobs hang on the val-epoch-end
  metric barrier even if everything else is right.
- **biom3 fixes:**
  `PL_wrapper.common_step` only syncs val metrics at epoch end (not
  per-step); device placement uses rank-local `self.device` (not the CLI
  `--device` string); `cond_elbo_objective` runs on the rank's tile.
- **Strategy:** alternated between
  `DDPStrategy(static_graph=True, gradient_as_bucket_view=True)` and
  `DeepSpeedStrategy(stage=2, overlap_comm=False, ...)`. Both have shown
  hangs in trials. `static_graph=True` plausibly addresses the
  last-training-step deadlock pattern but does not address the
  val-epoch-end deadlock pattern observed afterward.

**Open hang** (one failure mode, has reproduced under every
configuration tried so far):

| Boundary | Symptom | Stragglers stuck in | Affects |
|---|---|---|---|
| Last training step → val-epoch-end metric sync | log reaches `Validation DataLoader 0: 18/18`, no further progress | `_engine_run_backward` (autograd C++ engine) on 1–3 ranks; `all_reduce` from `_sync_ddp` on the rest | every config tested (DDP, DDP+static_graph, DeepSpeed) |

The progress-bar reading at "Validation DataLoader 0: 18/18" is misleading
— val *did* complete, but the all_reduce that follows it (val-epoch-end
metric aggregation) is where 11 ranks block, waiting for 1–3 stragglers
that never finished the previous training step's backward. So the actual
deadlock site is the same as it has been throughout: the train→val
boundary's collective.

Per-configuration straggler-count signal (lower is better but every
config still produced ≥1 straggler in some trial):

| Config | Trials | Stragglers per trial |
|---|---|---|
| DDP (no static_graph) | 7 | 3, 3, 2, 1, 0✓, ?, 2 |
| DDP + static_graph=True | 1+ | 1 |
| DeepSpeed ZeRO 2 | 1+ | 3 |

A single clean run is not evidence of a fix — the failure is
nondeterministic, ranges from 0 to 3 stragglers per run on the same
config, and reproduces eventually. When changing knobs, retest at least
3 times.

## Capturing py-spy stacks during a hung or stalled job

Run from a **second shell on the same compute node** while the job is
hanging. Use the absolute path to py-spy so a PATH hiccup doesn't lose
the iteration.

```bash
# Install py-spy into the active env once if missing
"$(which python)" -m pip install py-spy

# Dump all 12 ranks (and their dataloader workers) to one file per pid
PY_SPY="$(dirname "$(which python)")/py-spy"
mkdir -p ~/tmp/hang_dumps && rm -f ~/tmp/hang_dumps/*.txt
for pid in $(pgrep -u "$USER" -f 'biom3_train_stage3'); do
  ppid=$(ps -o ppid= -p $pid | tr -d ' ')
  echo "pid=$pid ppid=$ppid" > ~/tmp/hang_dumps/pid_${pid}.txt
  "$PY_SPY" dump --pid $pid >> ~/tmp/hang_dumps/pid_${pid}.txt 2>&1
done

# scp back to your dev box for offline analysis
# scp -r aurora:/tmp/hang_dumps ./
```

Falls back to gdb if py-spy can't attach (rare on Aurora compute nodes):

```bash
for pid in $(pgrep -u "$USER" -f 'biom3_train_stage3'); do
  echo "===== PID $pid ====="
  gdb -batch -p $pid -ex 'set pagination off' \
    -ex 'thread apply all bt' -ex detach -ex quit 2>&1
done > /tmp/hang_dumps_gdb.txt
```

## Aurora node topology and our binding rationale

Aurora node = 2 sockets, each with 1 CPU (52 physical cores, 104 logical
via SMT) and 3 PVC GPUs (2 tiles each, so 6 tiles per socket / 12 per
node). Cores 0 and 52 (and their SMT siblings 104, 156) are reserved
for the OS.

Our train scripts use the ALCF-canonical 8-cores-per-rank binding,
mapping 12 ranks to GPUs adjacent to their CPU cores:

| Rank | Cores | Socket | Driving |
|---|---|---|---|
| 0 | 1–8 | 0 | GPU 0, tile 0 |
| 1 | 9–16 | 0 | GPU 0, tile 1 |
| 2 | 17–24 | 0 | GPU 1, tile 0 |
| 3 | 25–32 | 0 | GPU 1, tile 1 |
| 4 | 33–40 | 0 | GPU 2, tile 0 |
| 5 | 41–48 | 0 | GPU 2, tile 1 |
| 6 | 53–60 | 1 | GPU 3, tile 0 |
| 7 | 61–68 | 1 | GPU 3, tile 1 |
| 8 | 69–76 | 1 | GPU 4, tile 0 |
| 9 | 77–84 | 1 | GPU 4, tile 1 |
| 10 | 85–92 | 1 | GPU 5, tile 0 |
| 11 | 93–100 | 1 | GPU 5, tile 1 |

The "jump from 48 to 53" exists because cores 49–52 cover the OS-reserved
core 52 plus a small buffer. Pairs of ranks share the cores nearest each
GPU, so DMA between framework memory and the tile stays within the local
socket.

`CCL_WORKER_AFFINITY=8,16,24,32,40,48,60,68,76,84,92,100` puts one oneCCL
progress thread on the last core of each rank's pin domain — equivalent
to `CCL_WORKER_AFFINITY=auto` but explicit so we can audit it. Each rank
effectively gets 7 framework cores + 1 CCL worker core inside its bind
range.

Lightning's DDPStrategy reads `PALS_LOCAL_RANKID` and calls
`torch.xpu.set_device(local_rank)` so rank N → `xpu:N`. Under the
frameworks-module default `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, that
enumeration is socket-ordered (xpu:0–5 on socket 0, xpu:6–11 on socket 1),
so the rank↔CPU binding above also pins each rank to a same-socket tile.

We do **not** use `gpu_tile_compact.sh` because the frameworks module sets
FLAT mode, under which ALCF explicitly recommends manual binding from the
application instead — Lightning is that manual binding.

## Verifying CPU bind took effect

To confirm the `--cpu-bind=list:...` argument in the train scripts
actually applied:

```bash
# Per-process mask — should show one rank per 8-core range
ps -eo pid,cmd | grep '[b]iom3_pretrain_stage3' \
  | awk '{print $1}' | xargs -n1 taskset -cp
```

What "good" looks like for a 12-rank single-node job:

```
pid <launcher>'s current affinity list: 1-51,53-103,...    # full mask, expected
pid <rank0>'s current affinity list: 1-8
pid <rank1>'s current affinity list: 9-16
...
pid <rank5>'s current affinity list: 41-48                  # socket 0 last
pid <rank6>'s current affinity list: 53-60                  # socket 1 first
...
pid <rank11>'s current affinity list: 93-100
```

If a rank's mask is the full `1-51,53-103,...` range or doesn't match the
configured list, the `--cpu-bind` arg didn't take effect — usually means
the job wasn't launched via `mpiexec`, or `--envall` is missing.

For per-thread placement (catches CCL workers and OMP threads):

```bash
ps -L -eo pid,tid,cmd | grep '[b]iom3_pretrain_stage3' \
  | awk '{print $2}' | xargs -n1 taskset -cp
```

What to look for:
- Each rank has many threads (~100+) on its 8-core range — Python
  interpreter, torch threads, DataLoader workers if `num_workers > 0`.
- **Exactly 12 threads** pinned to single cores in the
  `CCL_WORKER_AFFINITY` set: 8, 16, 24, 32, 40, 48 (socket 0) and 60, 68,
  76, 84, 92, 100 (socket 1). One CCL progress thread per rank, on the
  last core of that rank's range.
- The CCL worker threads are *inside* their rank's pin domain (not on a
  separate "worker-only" core like an older binding). This is intentional
  — the slot is sized so framework threads can use cores 1–7 etc. while
  the CCL worker has core 8 mostly to itself.

## Verifying torch.distributed is on xccl

Aurora frameworks/2025.3.1 requires `backend='xccl'` (the old `'ccl'`
backend was removed). Confirm at runtime:

```bash
python -c "
import inspect
import lightning.fabric.utilities.distributed as d
print('Lightning _sync_ddp source has xccl-skip guard?',
      'get_backend(group) != \"xccl\"' in inspect.getsource(d._sync_ddp))
"
```

The biom3 `PL_wrapper.PL_ProtARDM.on_fit_start` hook also logs the
backend per-rank at fit start; look for `[rank N] torch.distributed
backend = xccl` in stdout. If any rank prints something other than
`xccl`, that rank is on the wrong backend and collectives will not
work correctly.

## Inspecting hang dumps offline

Once dumps are scp'd back, common one-liners:

```bash
cd hang_dumps

# Group dumps by run_id (each py-spy capture writes the run's CLI to file 1)
grep -l 'V2026[0-9]+_[0-9]+' pid_*.txt 2>/dev/null \
  | xargs -I{} sh -c 'grep -oE "V2026[0-9]+_[0-9]+" "{}" | head -1' \
  | sort | uniq -c

# Quick top-of-stack per dump (skip dataloader workers via line-count heuristic)
for f in $(wc -l pid_*.txt | sort -n | awk '$1 > 60 {print $2}' | grep -v total); do
  top2=$(grep -E "^    " "$f" | head -2 | tr '\n' ' | ')
  echo "$(wc -l < $f)L $f: $top2"
done

# Move a run into a subfolder
mkdir -p trial_N
grep -l '<run_id>' pid_*.txt | xargs -I{} mv {} trial_N/
```

Common stack patterns to recognize:

| Stack top | Meaning |
|---|---|
| `barrier (distributed_c10d.py:NNNN)` from `_sync_ddp` | Waiting for all ranks at a Lightning metric-sync barrier |
| `all_reduce (distributed_c10d.py:NNNN)` from `_sync_ddp` | Same, but the patched-Lightning skip-barrier path is active and now we're at the actual all_reduce |
| `_engine_run_backward (autograd/graph.py:NNNN)` | Rank still computing gradients, hasn't reached the post-backward allreduce yet (load-imbalance straggler) |
| `allreduce_bucket (deepspeed/runtime/zero/stage_1_and_2.py:NNNN)` | DeepSpeed ZeRO grad reduction (only present if `strategy=DeepSpeedStrategy`) |
| `select (selectors.py:NNN)` from `_worker_loop` | DataLoader worker waiting for a batch request — harmless |
| `compute_ppl` / `cond_elbo_objective` / `_engine_run_backward` mixed | Healthy snapshot of an actively-running val/training step — NOT a hang |

## Killing a hung job

Ctrl+C usually doesn't work because the main thread is blocked inside a
oneCCL wait below the GIL and the SIGINT handler can't run. Bypass:

```bash
# Kill by command pattern
pkill -9 -f biom3_train_stage3

# Or by launcher PID (then its children)
LAUNCHER=$(pgrep -u "$USER" -f mpiexec | head -1)
kill -9 "$LAUNCHER"
pkill -9 -P "$LAUNCHER"

# If running as a PBS job
qdel <jobid>
```

## CCL warnings — what they mean, what to ignore

Output from a healthy job under our `environment.sh` will include:

```
|CCL_WARN| value of CCL_KVS_MODE changed to be mpi (default:pmi)
|CCL_WARN| value of CCL_ATL_SYNC_COLL changed to be 1 (default:0)
|CCL_WARN| value of CCL_OP_SYNC changed to be 1 (default:0)
|CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be pmix (default:hydra)
```

These are oneCCL **confirming** our overrides were honored — not actual
warnings about anything wrong.

Output that means something went wrong:

| Message | Meaning |
|---|---|
| `did not find MPI-launcher specific variables, switch to ATL/OFI` | Job was NOT launched via mpiexec; oneCCL fell back to libfabric/OFI transport which ALCF doesn't tune for. Re-launch via the `stage*_train_*.sh` wrappers. |
| `PMIx_Init failed: PMIX_ERR_UNREACH` | No PMIx server reachable. Same root cause as above when on a single node — only treat as benign if the rest of the run logs `CCL_PROCESS_LAUNCHER changed to be pmix` afterward. |
| `Receiver cache limit is reached: mem_handle_cache size: 1000, limit: 1000` | CCL's IPC handle cache filled. Raise via `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=8192` if it appears repeatedly. Doesn't on its own cause hangs in our experience. |
| `The AccumulateGrad node's stream does not match the stream of the node that produced the incoming gradient` | DDP autograd hooks landed on a different XPU stream than where grads are produced. On xccl the cross-stream sync is weaker than CUDA's; can manifest as backward-pass stragglers. No clean fix from biom3-side; reportable upstream as an Aurora xccl issue. |

## Stalls vs hangs — distinguishing the two

In our investigations, "hangs" at the train→val boundary often turned out
to be **long stalls**, not deadlocks. Distinguishing test:

1. Capture py-spy dumps now (T=0).
2. Wait 5–10 minutes.
3. Capture again (T=5–10 min).
4. Compare:
   - **Same PIDs in same code locations** = real hang or extremely slow
     straggler.
   - **PIDs/ranks have moved** (e.g., progress bar advanced, different
     ranks now in backward) = it's progressing, just slowly.

If it's a stall, the practical fix is a longer DDP timeout
(`DDPStrategy(timeout=timedelta(minutes=30))`) plus epoch-checkpoint +
retry-wrapper, rather than a code change.
