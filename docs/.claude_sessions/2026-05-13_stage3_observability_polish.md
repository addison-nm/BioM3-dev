# 2026-05-13 — Stage 3 observability polish: time limit, progress callbacks, batched metrics writes

## Top-level summary

Three buckets of work bundled into one commit on top of `58310cc`:

1. **New callbacks for observability and graceful exit:** `TimeLimitCallback`
   (broadcast-synced wall-time deadline → `trainer.should_stop`),
   `EpochProgressCallback` (logs "Completed / ended early" + elapsed time per
   train epoch), `StepProgressCallback` (within-epoch heartbeat at a
   fraction-of-epoch cadence). New CLI args: `--time_limit hh:mm:ss`,
   `--log_progress_fraction` (default `0.5`).

2. **`--log_every_n_steps` cleanup:** `type=float` → `type=int`, the magic
   `default=10001` → explicit `default=None` sentinel handled with a
   "default to once per epoch" branch. Dropped two now-redundant `int(...)`
   casts at the read site and at the combine-mode periodic-checkpoint cadence
   assignment. Fixed the totally wrong help text (it claimed to be about
   sample count for validation; it's actually Lightning's metric-flush
   cadence).

3. **`MetricsHistoryCallback` batched-write refactor:** train step records
   are now buffered in memory through each epoch and written to
   `metrics_history.train.jsonl` in one batched write at
   `on_train_epoch_end` (with a fallback flush in `on_train_end` /
   `on_exception`). Removed `--metrics_history_flush_every_n_steps` and the
   `flush_every_n_steps` / `_maybe_fsync` / `_append_jsonl` / `_unflushed_since_fsync`
   plumbing. Validation records still write + fsync at `on_validation_epoch_end`
   (with a defensive train-buffer flush there too).

Additional polish absorbed in the same commit (from work after `58310cc`
not yet documented in a prior session note):

- `_CheckpointLogMixin._save_checkpoint` now distinguishes `[periodic]`
  vs. `[monitored]` saves in the log line.
- Final-line log message at program exit: `Program exiting. Total elapsed
  time: H:MM:SS`.
- `run_summary.json` gained `completed_epochs` and `completed_steps` fields.
- Time-limit fallback wiring in `main()`'s `finally:` block: if `should_stop`
  was set by TimeLimitCallback, the run-summary's `exit_reason` is upgraded
  from `completed` to `time_limit_exceeded` based on `_MAIN_START_MONOTONIC`.
- Removed redundant post-fit `print_gpu_initialization()` call (it was being
  printed twice — startup + post-fit).
- Consolidated previously lazy in-function callback imports into a single
  top-of-file `from biom3.Stage3.callbacks import (...)` block.

## State before this session began

```bash
git checkout 58310cc  # feat(stage3): tune defaults + filesystem-based machine detection
```

## Why the buffered-write change is safe now

The user's concern with batched-at-epoch-end was always "what if we lose
records on a mid-epoch crash / timeout." Two recent additions remove that
risk for the only failure modes we actually care about:

| Failure mode                              | Behavior |
|-------------------------------------------|----------|
| Wall-time exceeded (`TimeLimitCallback`)  | `trainer.should_stop = True` at batch boundary → Lightning fires `on_train_epoch_end` for the wrap-up → buffer flushed. **Safe.** |
| Python exception in `trainer.fit`         | `on_exception` fires → buffer flushed. **Safe.** |
| Hard SIGKILL / GPU segfault               | Pending buffer (current epoch's step records) lost. Records from *prior* completed epochs are safe (they were flushed at their `on_train_epoch_end`). |

The old per-step-write scheme protected slightly better against hard SIGKILL
— but only by paying per-step JSON serialization + `fh.write()` on the
training hot path. The user explicitly flagged this as the bottleneck
("GPU-inefficient metrics that we write"). The buffered scheme trades a
small worst-case-window (one in-flight epoch) for substantially less
training-time I/O.

`rebuild_metrics_history_pt`'s existing tolerance for a truncated tail line
already covers the "partial last write was interrupted" case.

## Why the `--log_every_n_steps` cleanup

Pre-existing state on this arg:

```python
parser.add_argument('--log_every_n_steps', default=10001, type=float,
                    help='number of samples to validate on...')
```

Three things wrong:

1. **Help text** described a totally different arg (sample count for
   validation). Actual semantic is `Trainer(log_every_n_steps=N)` —
   Lightning's metric-flush cadence — also reused as the default
   periodic-checkpoint cadence under `combine` training mode.

2. **`type=float`** — every consumer immediately `int()`-casts:
   - `run_PL_training.py:1369` — `max(1, int(num_training_batches))` (the
     clamp branch)
   - `run_PL_training.py:1476` — `periodic_every_n_steps = int(log_every_n_steps)`
   - `Trainer(log_every_n_steps=...)` — Lightning's documented type is int

   All 21 in-tree callers (configs + entrypoint_args fixture files) pass
   plain integers. No fractional value is ever used.

3. **`default=10001`** — a value-shaped sentinel meaning "any number larger
   than `num_training_batches` so the auto-clamp falls back to 'log once per
   epoch.'" Working but opaque. Replaced with `default=None` and explicit
   "default to once per epoch" handling at the read site, alongside the
   existing clamp branch for explicit oversized values:

   ```python
   if log_every_n_steps is None:
       log_every_n_steps = max(1, num_training_batches or 1)
       logger.info("Defaulted log_every_n_steps to %d (once per epoch)",
                   log_every_n_steps)
   elif num_training_batches and log_every_n_steps > num_training_batches:
       log_every_n_steps = max(1, num_training_batches)
       logger.info("Clamped log_every_n_steps to %d (number of training batches)",
                   log_every_n_steps)
   ```

   Behavior for any **explicit** caller is identical to before. Only the
   default-path log message changes (`Defaulted` vs `Clamped`).

## TimeLimitCallback design notes

Critical detail: rank-0-only `trainer.should_stop = True` does **not**
propagate on XPU/CCL. The early version of this callback had only rank 0
set `should_stop`, and the first multi-node test deadlocked — node 1's 6
GPUs spun at 100% in collectives while node 2 idled. The same XPU/CCL
integer all-reduce bug that motivated `SyncSafeModelCheckpoint`.

Fix: rank 0 makes the decision; all ranks receive the int32 decision via
`torch.distributed.broadcast(decision_t, src=0)` (point-to-many, works on
XPU/CCL); each rank sets its own `should_stop`. Check fires every 50 train
batches on rank 0 (`check_every_n_steps=50`).

## StepProgressCallback / EpochProgressCallback notes

Both are rank-0-only logging callbacks; neither participates in any collective.

- `EpochProgressCallback` checks `trainer.should_stop` at
  `on_train_epoch_end` and emits either "completed" or "ended early
  (partial)" — without this discrimination, the line was misleading on
  time-limit-triggered partial epochs.
- `StepProgressCallback` uses `trainer.num_training_batches` (set during
  `on_train_epoch_start`) to derive the per-epoch interval. Clamped to
  `max(1, ...)` so very small dataloaders still get one line per batch.
- CLI: `--log_progress_fraction` default `0.5` (fires at 50% and 100%).
  Naming: argued briefly between `step_progress_fraction` and
  `log_progress_fraction`; the user chose the latter. Class name stays
  `StepProgressCallback` for sibling-consistency with
  `EpochProgressCallback`.

## Code changes

[`src/biom3/Stage3/callbacks.py`](src/biom3/Stage3/callbacks.py):

- `MetricsHistoryCallback`: removed `flush_every_n_steps` parameter,
  `_unflushed_since_fsync` field, `_append_jsonl()`, `_maybe_fsync()`.
  Added `_pending_train_jsonl: list[dict]` buffer and
  `_flush_pending_train_jsonl()` helper (batched write + fsync). Hooks
  rewired:
  - `on_train_batch_end` — appends to memory buffer only (no I/O).
  - `on_train_epoch_end` — always runs (was conditional on `every_n_epochs`);
    appends optional epoch-source record, then flushes buffer.
  - `on_validation_epoch_end` — writes val record, defensively flushes
    train buffer too, fsyncs both streams, saves `.pt`.
  - `on_train_end` / `on_exception` — flush pending before close.
- New `TimeLimitCallback` (broadcast-based cross-rank sync).
- New `StepProgressCallback` (within-epoch heartbeat).
- New `EpochProgressCallback` (per-epoch elapsed time + completed/partial).
- `_CheckpointLogMixin._save_checkpoint` — differentiates `[periodic]`
  (monitor=None) vs. `[monitored]` saves in the log line.

[`src/biom3/Stage3/run_PL_training.py`](src/biom3/Stage3/run_PL_training.py):

- Top-of-file eager imports from `biom3.Stage3.callbacks` (consolidated
  the previous in-function lazy imports).
- New module-level state: `_MAIN_START_MONOTONIC`, `_LAST_TRAINER`.
- `--log_every_n_steps`: `type=int`, `default=None`, rewritten help text;
  new None-branch + clamp branch at the read site; dropped two `int(...)`
  casts.
- Removed `--metrics_history_flush_every_n_steps` arg; updated
  `--metrics_history_every_n_steps` help text to describe the buffered
  scheme.
- New CLI: `--log_progress_fraction` (default `0.5`), `--time_limit hh:mm:ss`.
- `retrieve_all_args`: parses `--time_limit` into
  `args.time_limit_seconds: int | None`.
- `train_model()`: appends `EpochProgressCallback()` unconditionally,
  `StepProgressCallback(fraction=...)` if `args.log_progress_fraction > 0`,
  `TimeLimitCallback(...)` if `args.time_limit_seconds is not None`.
  Stashes `_LAST_TRAINER = trainer` right after `Trainer(**trainer_params)`.
  Removed redundant post-fit `print_gpu_initialization()` call.
- `_write_run_summary` gained `completed_epochs` / `completed_steps`
  parameters.
- `main()`'s `try/except/finally`: in `finally:`, if `exit_reason` is still
  `"completed"` but elapsed time has crossed `time_limit_seconds`, upgrade
  it to `"time_limit_exceeded"`. Then write `run_summary.json` with
  completed_epochs/completed_steps pulled from `_LAST_TRAINER`. Final log
  line: `Program exiting. Total elapsed time: H:MM:SS`.
- `MetricsHistoryCallback` construction: dropped `flush_every_n_steps=`
  kwarg.

[`tests/stage3_tests/test_callbacks.py`](tests/stage3_tests/test_callbacks.py):

- `TestMetricsHistoryJsonlStreaming.test_jsonl_written_on_each_step` and
  `test_jsonl_respects_every_n_steps`: now call `cb.on_train_epoch_end(...)`
  before reading the JSONL (records were previously expected after only
  `on_train_batch_end`s).
- New test `test_step_records_buffered_until_epoch_end` pinning the new
  contract — JSONL is empty pre-flush, contains all records post-flush,
  `_pending_train_jsonl` cleared.
- Replaced `TestMetricsHistoryFlush`:
  - `test_no_fsync_within_epoch` — the central guarantee.
  - `test_fsync_on_train_epoch_end` — flush happens at the epoch boundary.
  - `test_on_exception_flushes_pending` — crash robustness.

## Heads-up about callsites / configs

- All 7 stage 3 training configs set `metrics_history_every_n_steps: 1`
  explicitly. Under the new scheme they buffer ~`num_training_batches`
  records per epoch (~200KB at ~200 bytes/record × 1024 batches).
  Negligible memory; eliminates the per-step I/O bottleneck the user
  flagged.
- Configs that previously set `metrics_history_flush_every_n_steps` — none
  exist; the only mention was in an old `args.json` artifact (default
  null). Safe to remove.

## What did NOT get committed

- **Pre-existing working-tree edits not from this session window** (left
  alone, consistent with the prior session note's policy):
  - `configs/stage3_training/finetune_v1.json` — `limit_val_batches: 0.05 → 0.6`
  - `scripts/launchers/aurora_singlenode.sh`,
    `scripts/launchers/aurora_multinode.sh` — `GPU_BIND_SCHEME` commented
    out
- **Untracked aurora jobscripts**
  (`jobs/aurora/job_stage3_finetune_v1_ft16_n{1,2,4,8}_fluc.pbs`,
  `jobs/aurora/job_test_stage3_multinode_n2.pbs`) — user has opted not to
  commit them on prior milestones; same here.
- **Root-level scratch artifacts** (`error_log*.txt`, `prompts.csv`,
  `run_sh3_multinode.sh`, `sh3_run_config.json`,
  `job_stage3_finetune_v1_ft16_n8_flucbatch16_nowandb.pbs`, the stray `.o`
  file).

## Lingering / deferred

- **Sampling-side `device_id=` fix** for the c10d barrier warning in
  `biom3.core.distributed.init_distributed_if_launched`. Memory entry
  `todo_barrier_device_id_warning.md` still tracks this. The
  training-side surface is handled (warnings filter + filtered logger).
- **Underlying GPU-segfault crash on resume-into-new-run-id** — unresolved.
  Manifested again in run `V20260513_173414` (see
  `outputs/Stage3/finetuning/runs/.../artifacts/run_summary.json`). The
  observability surface (build_manifest pre-training, run_summary
  post-training, TimeLimitCallback, progress callbacks) is now in place
  and helping; the actual crash root cause is upstream of this commit.
- **Two side notes on `--log_every_n_steps`** — both resolved this session.
  Was: should `type=float` become `type=int`? Should `default=10001`
  become a sentinel? Now: both done.

## Commit

To be made on `addison-dev` immediately after this note is written. No
push (per branch policy).
