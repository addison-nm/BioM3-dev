# Session: Per-epoch training benchmark callback

**Date:** 2026-04-12
**Branch:** benchmarking

## Goal

Add a way to benchmark Stage 3 training performance as a function of:
1. Effective batch size (`batch_size × acc_grad_batches × gpu_devices × num_nodes`)
2. Number of nodes/devices
3. Per-device memory usage (CUDA and XPU, optionally per-rank)

Existing `run.log` output had per-epoch loss/accuracy summaries but no
wall-clock timing, throughput, or memory metrics. Running the same training
script with different configs left no structured record to compare.

## Design

### New callback: `TrainingBenchmarkCallback`

Lives in `biom3.Stage3.callbacks` alongside `MetricsHistoryCallback`.
Kept as a separate class — different concern (timing/memory vs. loss/accuracy)
and different save cadence (per epoch boundary vs. per step).

**Hooks used:**

| Hook | Purpose |
|------|---------|
| `on_train_start` | Capture start timestamp; init interval tracking; reset peak memory counters |
| `on_train_epoch_start` | Record `time.perf_counter()` start; reset peak memory |
| `on_train_epoch_end` | Compute epoch duration, throughput, peak memory; gather across ranks if enabled |
| `on_validation_epoch_end` | For `combine` strategy only: record interval between validation checks |
| `on_train_end` | Save `benchmark_history.json`, log summary (rank 0) |

`on_train_epoch_end` fires AFTER training batches but BEFORE validation in
Lightning 2.x, so pure training time is measured cleanly (excludes val overhead).

### Output: `benchmark_history.json`

JSON (not `.pt`) because the data is small, human-readable, and trivially
aggregable across runs with `jq` or pandas.

```json
{
  "config": {
    "batch_size": 16,
    "acc_grad_batches": 4,
    "gpu_devices": 4,
    "num_nodes": 2,
    "effective_batch_size": 512,
    "num_workers": 0,
    "precision": "bf16",
    "training_strategy": "primary_only",
    "backend": "cuda",
    "hostname": "spark01",
    "train_start_timestamp": "2026-04-12T10:30:00"
  },
  "epochs": [
    {
      "epoch": 0,
      "global_step": 1250,
      "steps_in_epoch": 1250,
      "samples_in_epoch": 640000,
      "epoch_wall_time_sec": 342.7,
      "samples_per_sec": 1867.5,
      "steps_per_sec": 3.65,
      "peak_memory_allocated_gb": 12.34,
      "peak_memory_reserved_gb": 14.56,
      "peak_memory_allocated_gb_per_rank": [12.34, 12.41, 12.29, 12.38],
      "peak_memory_reserved_gb_per_rank": [14.56, 14.62, 14.51, 14.59]
    }
  ]
}
```

Per-rank fields only present when `--benchmark_all_ranks_memory True`.

### CLI flags (all off by default)

- `--save_benchmark True` — activate the callback.  Off by default so production
  jobs see zero overhead.
- `--benchmark_skip_first_epoch True` — exclude epoch 0 from the summary stats
  (JIT / DeepSpeed / memory allocator warmup).  Raw record still recorded.
- `--benchmark_all_ranks_memory False` — opt-in to all-gather peak memory from
  every rank into `peak_memory_{allocated,reserved}_gb_per_rank` lists.

### Memory tracking across backends

Two module-level helpers dispatch on `BACKEND_NAME`:

- `_reset_peak_memory_stats()` — calls `torch.cuda.reset_peak_memory_stats()`
  or `torch.xpu.reset_peak_memory_stats()`; no-op on CPU.
- `_get_peak_memory_stats()` — returns `(max_memory_allocated,
  max_memory_reserved)` in bytes, or `(None, None)` on CPU / missing XPU API.

`torch.cuda` and `torch.xpu` expose identically-named functions, so dispatch is
trivial.  Bytes → GB conversion happens in `_build_record` so the JSON is
human-readable.

### Per-rank gather

`_gather_across_ranks()` uses `pl_module.all_gather()` — Lightning's
strategy-level abstraction, which works uniformly over NCCL (CUDA) and
oneCCL (XPU).  Steps:

1. Pack `[peak_alloc, peak_reserved]` as a float64 tensor on `pl_module.device`.
2. Call `all_gather` on **all** ranks (collective; must not be gated by rank).
3. Handle both 1D `(2,)` (single-process) and 2D `(world_size, 2)`
   (distributed) return shapes.
4. Convert to GB lists and attach to the record on rank 0.

Wrapped in try/except — all-gather failure degrades gracefully to `None` with
a warning, so a transient collective issue doesn't abort training.

The rank-0 scalar `peak_memory_allocated_gb` / `peak_memory_reserved_gb`
fields remain populated even when per-rank mode is on, so existing consumers
stay compatible.

## Files changed

- `src/biom3/Stage3/callbacks.py` — added `_reset_peak_memory_stats()`,
  `_get_peak_memory_stats()`, `TrainingBenchmarkCallback` class (~250 lines).
- `src/biom3/Stage3/run_PL_training.py` — three new CLI args with `str_to_bool`
  conversions; callback instantiation in `train_model()` gated by
  `save_benchmark`; added to callbacks list conditionally.
- `tests/stage3_tests/test_callbacks.py` — new test file, 26 tests organized
  into: effective batch size, record building, hook sequences (epoch-based
  and combine), memory tracking, per-rank gather, save output, skip-first-epoch.

## Testing

All tests use mocks — no GPU or distributed setup required:

- `_get_peak_memory_stats` and `_reset_peak_memory_stats` are patched per test
  to inject synthetic memory values.
- `pl_module.all_gather` is patched with a `MagicMock` returning a tensor of
  the expected shape (1D for single-process, 2D for multi-rank).
- Uses `tmp_path` fixture for output directories.

Test coverage:
- Effective batch size computation across `(batch, acc, devices, nodes)` combos
- Record fields including bytes→GB memory conversion
- Epoch-based hook sequence produces N records for N epochs
- Combine-strategy interval tracking uses `on_validation_epoch_end` boundaries
- Per-rank gather handles single-process (1D), distributed (2D), disabled
  flag, peak=None, and all_gather failure cases
- JSON output contains config + epochs + per-rank lists (when enabled)
- Rank > 0 does not save
- `skip_first_epoch` still records raw data but excludes it from summary

All 26 tests pass on CUDA (Spark GB10).

## Usage

```bash
# Minimal benchmark
biom3_pretrain_stage3 --save_benchmark True ...

# With per-rank memory breakdown
biom3_pretrain_stage3 \
    --save_benchmark True \
    --benchmark_all_ranks_memory True \
    ...
```

Output file: `{output_root}/runs/{run_id}/artifacts/benchmark_history.json`.

Per-epoch lines also appear in `run.log`:

```
Benchmark  epoch=0  step=1250  time=342.7s  samples/sec=1867.5  steps/sec=3.650  effective_batch=512  peak_alloc_per_rank=[12.34,12.41,12.29,12.38]GB  peak_reserved_per_rank=[14.56,14.62,14.51,14.59]GB
```

Metrics also logged to TensorBoard / WandB via `pl_module.log("benchmark/...", ...)`
with `rank_zero_only=True` (never `sync_dist` — wall times shouldn't be averaged).

## Key design decisions

1. **Separate callback, not an extension of `MetricsHistoryCallback`.**
   Single-responsibility — timing/memory vs. loss/accuracy are different
   concerns with different save cadences.

2. **Off by default.**  User explicitly asked for zero overhead on
   production jobs.  All three flags default to False/True-conservative.

3. **JSON, not `.pt`.**  Small data, cross-language, `jq`-friendly,
   easy to aggregate across runs in a notebook.

4. **Lightning `all_gather` instead of raw `torch.distributed`.**
   Abstracts NCCL/CCL differences; one code path for CUDA and XPU.

5. **Collective before rank gate.**  The `_gather_across_ranks()` call must
   happen on all ranks before any rank-0 check, otherwise non-zero ranks
   would hang waiting for the collective.

6. **Rank-0 scalar fields kept alongside per-rank lists.**
   Non-breaking: existing consumers reading `peak_memory_allocated_gb`
   continue to work when per-rank mode is enabled.
