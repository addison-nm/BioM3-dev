# Crash-safe metrics + per-step training benchmarking

**Date:** 2026-04-24
**Branch:** `addison-training-bench-and-metrics` (worktree; not yet merged into `addison-dev`)
**Worktree:** `.claude/worktrees/training-bench-and-metrics/`

## Motivation

Two pain points on HPC (Spark / Polaris / Aurora):

1. **Metrics lost on timeout.** `MetricsHistoryCallback` accumulated train
   and val records in memory and only wrote `metrics_history.pt` from
   `on_train_end`. A walltime-killed PBS job wrote nothing — all tracked
   metrics for that run were gone.
2. **No per-step training timing.** `TrainingBenchmarkCallback` reported
   per-epoch wall clock, throughput, and peak memory, but not per-step
   — which is what you actually need to spot dataloader stalls, kernel
   recompiles, or gradient-accumulation irregularities.

Plus a third, smaller item raised mid-session: a version of
`every_n_epochs` analogous to the existing `every_n_steps` — for both
the metrics recorder and the `ModelCheckpoint` cadence.

## What shipped

Three commits on the worktree branch:

```
f490d97  feat(benchmarks): add Stage 3 training sweep driver
a7f6247  feat(stage1,stage3): wire new metrics and benchmark CLI args
f858595  feat(stage3): crash-safe metrics streaming + per-step benchmark timing
```

### Callbacks (`src/biom3/Stage3/callbacks.py`)

**`MetricsHistoryCallback`**

- Streams every recorded record to `metrics_history.train.jsonl` /
  `metrics_history.val.jsonl` as it lands. Each record is one line.
- New `flush_every_n_steps` kwarg: when set, `flush() + os.fsync()` on
  that cadence so partial-run recovery is robust.
- New `every_n_epochs` kwarg: additionally record train metrics in
  `on_train_epoch_end`, capturing epoch-averaged values. Records carry
  `source: "step"` or `source: "epoch"` to distinguish.
- `on_exception` hook: best-effort flush if training raises before
  `on_train_end`.
- `_save()` still writes `metrics_history.pt` at clean shutdown; schema
  unchanged (downstream `biom3.viz.stage3_training_runs` unaffected).
- New module-level helper `rebuild_metrics_history_pt(output_dir)`
  reconstructs `.pt` from the JSONL files after a crashed run. Tolerates
  a truncated final line.

**`TrainingBenchmarkCallback`**

- New `per_step` kwarg: writes `benchmark_steps.jsonl` with one record
  per training step (`epoch`, `global_step`, `batch_idx`,
  `step_wall_time_sec`). Rank-0 only, file-handle held open, flushed on
  `on_train_end` or `on_exception`.
- Peak memory intentionally stays at epoch granularity. Resetting the
  device peak-memory counters per step forces a device sync, which
  would dwarf the measurement it's trying to record. This is a
  documented trade-off in the docstring.

### CLI wiring (both stages' `run_PL_training.py`)

New flags on both `biom3_pretrain_stage3` and `biom3.Stage1.run_PL_training`:

| Flag                                     | Purpose |
| ---------------------------------------- | ------- |
| `--metrics_history_every_n_epochs`       | Epoch-cadence metrics recording. |
| `--metrics_history_flush_every_n_steps`  | fsync JSONL every N steps. |
| `--benchmark_per_step`                   | Stream per-step wall time. |
| `--limit_train_batches`                  | Cap train batches per epoch (new; also exposes PL's native knob). |

Also: `--checkpoint_every_n_epochs` was declared in both stages but
never passed through to `ModelCheckpoint`. Now wired as the
`every_n_epochs` kwarg on the primary checkpoint callback. This was a
latent bug, not a new feature.

### Training benchmark sweep driver (`src/biom3/benchmarks/Stage3/training.py`)

Follows the same shape as the existing
`biom3.benchmarks.Stage3.generation` module:

- Config-driven sweep over
  `(batch_size, acc_grad_batches, precision, gpu_devices, num_nodes, num_workers)`.
- Each cell launches `biom3_pretrain_stage3` as a subprocess with CLI
  overrides, then reads back that cell's `benchmark_history.json` and
  `benchmark_steps.jsonl` and summarizes into one record.
- Outputs a timestamped `<output_root>/<UTC>/` directory containing
  `config.json`, `env.json`, `results.json`, `results.npz`, per-cell
  artifact dirs, and a top-level `run.log`.
- New entrypoint: `biom3_benchmark_stage3_training` (in
  `pyproject.toml`).
- Example config: [configs/benchmark/stage3_training_example.json](../../configs/benchmark/stage3_training_example.json).

## XPU / Aurora compatibility

Verified without running on Aurora:

- **Zero CUDA-only code** in my diff: no `torch.cuda`, no hardcoded
  `"cuda"` strings, no direct `torch.cuda.*` calls. All device-sensitive
  ops route through the existing `biom3.backend.device` abstraction
  (`reset_peak_memory_stats`, `get_peak_memory_stats`, `device_sync`,
  `get_device_info`), which is already symmetric across CUDA and XPU.
- **Zero collective ops** (no `sync_dist`, `all_gather`, `all_reduce`,
  `barrier`) in the new code. The oneCCL integer-reduce bug that
  motivated `SyncSafeModelCheckpoint` can't bite us here. JSONL writes
  are rank-0-only (or `save_ranks`-gated) — no cross-rank coordination.
- **Lightning API surface**: every callback hook I use exists on
  `pl.Callback` in `pytorch_lightning` (CUDA path). `lightning` (the
  Aurora path) is the same project renamed — identical callback API.
  `ModelCheckpoint.every_n_epochs` and `Trainer.limit_train_batches`
  are standard in both packages.
- **End-to-end smoke test** (CPU, real PyTorch Lightning): ran a toy
  model for 2 epochs with all callbacks enabled. Happy path produces
  consistent JSONL + `.pt` + `benchmark_history.json` +
  `benchmark_steps.jsonl`. Crash path (RuntimeError at `batch_idx=3`,
  no `on_train_end`) leaves JSONL files on disk with partial records
  flushed via `on_exception`; `rebuild_metrics_history_pt()` reconstructs
  the `.pt` successfully.
- `pytest tests/ --quick` → 472 passed, 188 skipped, no regressions.
  48 tests in `tests/stage3_tests/test_callbacks.py` (14 new), 8 in the
  new `tests/stage3_tests/test_training_benchmark_driver.py`.

**Known scope gap** (not a bug): the sweep driver uses plain
`subprocess.run`. Single-node sweeps work on any backend. Multi-node
sweeps on Aurora would need an `mpiexec` wrapper option on the driver
— explicitly out of scope for this round.

## Follow-ups (not done this session)

- Stage 1 training benchmark sweep driver. The scaffolding is a direct
  copy of `biom3.benchmarks.Stage3.training`; defer until someone has a
  concrete Stage 1 sweep workload.
- Multi-node sweep support on Aurora (mpiexec wrapper option).
- Optional: surface `rebuild_metrics_history_pt` as a CLI command
  (`biom3_recover_metrics`) for post-incident recovery without needing
  an interactive Python session.

## Merge plan

This branch is a worktree off `addison-dev` per `CLAUDE.md`'s "Parallel
development with worktrees" convention. When ready, fast-forward-merge
`addison-training-bench-and-metrics` into `addison-dev` and remove the
worktree. Nothing on this branch depends on other in-flight feature
branches.
