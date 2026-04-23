# Session: ProteoScribe generation benchmark + pre-unmask feature

**Date:** 2026-04-21
**Branch:** `addison-generation-benchmarking` (worktree off `addison-dev`)
**Commits:** `7f27d20`, `5c3bc5b`

## Goal

Measure wall-clock time and peak device memory for Stage 3 ProteoScribe
sequence generation across a parameter sweep, and tag each run with an
architecture ID plus full device info so data collected on different
clusters (Spark / Polaris / Aurora) stays comparable. A prerequisite
feature — starting diffusion from a partially-unmasked state — was built
in the same branch so the benchmark can sweep over a configurable
diffusion-step budget `D` without running the full 1024-step chain.

## Scope of the sweep

Per the user's spec:

- `N ∈ [4, 8, 16, 32, 64, 128]` total sequences
- `P ∈ [2, 4]` prompts (must divide `N`; `R = N // P`)
- `B ∈ [4, 8, 16, 32, 64]` batch size (runs where `B > N` are skipped)
- `token_strategy ∈ {sample, argmax}` (swept as a list inside `model_kwargs`)
- `unmasking_order = random` (fixed for v1; confidence paths still work
  with pre-unmask but are not part of this sweep)
- `D = 16` default (configurable per run)

Per-run metrics: `T_total_s`, `T_per_batch_s`, `peak_alloc_bytes`,
`peak_reserved_bytes`, `num_batches`.

## Component A: pre-unmask feature

File: `src/biom3/Stage3/run_ProteoScribe_sample.py`.

New CLI on `biom3_ProteoScribe_sample`:

- `--pre_unmask` (bool flag, default `False`)
- `--pre_unmask_config <path.json>` — required when `--pre_unmask` is set

Pre-unmask config schema (v1):

```json
{
  "strategy": "last_k",      // only supported value in v1
  "fill_with": "PAD",        // enum, resolved via _PRE_UNMASK_FILL_ALIASES
  "diffusion_budget": 16     // D
}
```

### Why a separate config file

`strategy` and `fill_with` are kept as enums so new strategies (first_k,
random_k, interleaved, …) and new fill tokens can be added without
breaking existing configs. `diffusion_budget` is the exposed knob.

### Integration point

`batch_stage3_generate_sequences()` previously initialised the starting
mask state with `torch.zeros(current_batch_size, args.diffusion_steps)`
at two sites — one per unmasking order branch. Replaced with
`_build_initial_mask_state(args, batch_size, tokens)` which returns both
`extract_digit_samples` and `sampling_path`:

- **Pre-unmask off (default)**: zero tensor shape `(B, seq_len)` and a
  permutation of `range(seq_len)` — identical to old behaviour, because
  `seq_len == args.diffusion_steps` when pre-unmask is off.
- **Pre-unmask on**: tensor shape `(B, seq_len=1024)` with positions
  `[0, D)` set to 0 (mask) and `[D, seq_len)` set to the resolved fill
  token id (23 for `<PAD>`). `sampling_path` is a permutation of
  `range(D)`, so only positions `[0, D)` are ever updated during the
  diffusion loop.

### The seq_len vs diffusion_steps split

Before this change the two were the same value (`args.diffusion_steps`
was doing double duty). To support pre-unmask without breaking the
existing single-value convention, `main()` snapshots
`config_args.sequence_length = config_args.diffusion_steps` once (the
architectural seq length the model was trained on), and then — if
pre-unmask is enabled — overrides `diffusion_steps` with the budget `D`.
`batch_generate_denoised_sampled` still reads `seq_len` from
`extract_digit_samples.size(1)` and `max_diffusion_step` from
`args.diffusion_steps`, so the existing sampling code needs no changes.

### Confidence branch (not in sweep, but still correct)

`batch_generate_denoised_sampled_confidence` picks the next position
dynamically via `is_masked = (temp_mask_realization.squeeze(1) == 0)`.
Pre-filled PAD positions have id 23, not 0, so they are automatically
excluded from candidate positions — confidence unmasking works with
pre-unmask out of the box.

## Component B: benchmark harness

File: `scripts/benchmark_generation.py`.

Standalone script (CLI entrypoint deferred). Flow per invocation:

1. Validate the sweep config (error on `P ∤ N`; skip `B > N` combos with
   a log line — those are meaningless, not misconfigured).
2. Generate the maximum-`P` synthetic prompts (`"AAAA", "AAAB", …`),
   write to a temporary CSV with a dummy protein sequence column, then
   invoke `biom3_PenCL_inference` and `biom3_Facilitator_sample` as
   subprocesses to embed them into `z_c`. The resulting tensor is
   sliced per-run; prompt embedding is never repeated across the sweep.
3. Load the ProteoScribe model once (via `prepare_model_ProteoScribe`).
4. Loop `(token_strategy, N, P, B)`. For each valid combo, build a
   fresh `Namespace` with the per-run kwargs and call
   `batch_stage3_generate_sequences` in-process.
5. Wrap each call with `reset_peak_memory_stats()` /
   `device_sync()` / `perf_counter()` / `device_sync()` /
   `get_peak_memory_stats()`.

### Why `model_kwargs` is free-form

The user explicitly wanted the benchmark to not hard-code model-specific
arguments so it stays usable when ProteoScribe is swapped for a
different architecture. `model_kwargs` values may be scalars or lists;
list values become additional sweep axes (currently only
`token_strategy` is used this way). The sweep axes `N`, `P`, `B` are
named explicitly because they'll likely apply regardless of
architecture.

### Output layout

```
<output_root>/<UTC timestamp>/
  config.json     # input config, verbatim
  env.json        # arch_id + device info (backend, GPU name, CUDA cap,
                  #   torch/python version, hostname)
  results.json    # list of per-run records (human-readable)
  results.npz     # dense numpy arrays keyed by (sweep axes + metrics)
  run.log         # captured stdout/stderr
  prompts/        # z_c cache + CSV sent through Stage 1+2
```

Numpy is preferred over pickle/torch for `results.npz` so plots and
downstream tools don't need the `biom3` environment to load it.

### Device info capture

Memory helpers were promoted from `Stage3/callbacks.py` to
`backend/device.py` (`reset_peak_memory_stats`, `get_peak_memory_stats`)
and joined by `device_sync()`, `get_device_name()`, and
`get_device_info()`. `callbacks.py` re-imports them under the existing
private names so the existing test mocks in `test_callbacks.py` keep
working without changes.

`get_device_info()` returns:

```
backend, device_name, hostname, torch_version, python_version,
platform, device_count, cuda_capability (CUDA), cuda_version (CUDA)
```

Example `env.json` from a Spark run:

```json
{
  "arch_id": "ProteoScribe_1block_v1",
  "backend": "cuda",
  "device_name": "NVIDIA GB10",
  "hostname": "spark-nm",
  "cuda_capability": "12.1",
  "cuda_version": "12.9",
  ...
}
```

## Component C: plotting

File: `scripts/plot_benchmark.py`.

Takes `--run_dir` pointing at a benchmark output directory; writes four
PNGs to `<run_dir>/images/`:

- `time_per_step_vs_batch.png` — **primary chart.** Per-step time
  (`T_total_s / (num_batches * D)`) vs batch size on log-log. Per-step
  cost is determined by `B` and `token_strategy`; `N` and `P` should
  not matter, so same-`B` points cluster — the scatter makes that
  assumption visible.
- `peak_memory_vs_batch.png` — GB vs B.
- `throughput_vs_batch.png` — sequences/sec vs B.
- `total_time_vs_N.png` — total wall-clock vs N, one line per B,
  faceted by token_strategy.

Every figure is titled `arch_id · device_name · hostname` so files
from different machines stay self-identifying when shared.

## Observations from the Spark (GB10) smoke run

Sweep: N ∈ {16,32,64}, P ∈ {2,4}, B ∈ {4,8,16,32}, D=16, both token
strategies — 44 runs.

- Per-step time is near-linear on log-log in B; ~45 ms at B=4,
  ~400 ms at B=32 on GB10.
- `sample` and `argmax` overlap — the Gumbel trick adds negligible
  overhead, consistent with the pre-allocated Gumbel buffer in
  `sampling_analysis.py`.
- Peak allocated grows with B (0.7 GB at B=4 → 2.0 GB at B=32);
  reserved is near-constant at 2.56 GB (allocator slack), which is why
  we log both.
- Throughput *decreases* as B grows on this tiny model / D=16 setup
  (4.55 → 3.97 seq/s). Kernel launch overhead amortises fast, so bigger
  batches buy less at this problem size. This flips on larger models /
  longer D and is worth revisiting once we sweep those.

## Things this session deliberately did NOT do

- Did not touch the confidence-unmasking sampling path (pre-unmask
  works with it automatically; running it in the sweep was out of
  scope for v1).
- Did not build a unified benchmark test harness under
  `pytest @pytest.mark.benchmark`. The existing
  `tests/stage3_tests/test_bench_stage3_sample.py` remains separate;
  reconciling the two is a future task.
- Did not implement additional pre-unmask strategies (first_k,
  interleaved). The enum is there to accept them; v1 only needs
  last_k.
- Did not hook `biom3_ProteoScribe_sample`'s `write_manifest` to
  record pre-unmask parameters. Manifest additions should come when a
  downstream consumer actually reads them.

## Follow-ups / open items

- **CLI entrypoint** for the benchmark script (add to `pyproject.toml`
  once the interface has stabilised).
- **Confidence-order benchmark** — the existing sweep only covers
  `unmasking_order=random`. Confidence mode has different per-step
  cost (argmax over logits to pick position) and deserves its own
  numbers.
- **Multi-run plots** — once benchmarks exist from Polaris/Aurora,
  extend `plot_benchmark.py` to overlay multiple `run_dir` args so
  GPU-vs-GPU comparisons are one command.
- **Larger `D` sweep** — default `D=16` is artificially small. A
  sweep over D ∈ {16, 64, 256, 1024} would expose whether per-step
  cost stays flat as the model is asked to do longer diffusion trajectories.
- **Per-step memory breakdown** — current code records peak over the
  full call; instrumenting the inner loop would identify which step
  allocates what (especially with `store_probabilities=True`).

## Files touched

| File | Change |
|------|--------|
| `src/biom3/Stage3/run_ProteoScribe_sample.py` | Pre-unmask CLI, `load_pre_unmask_config`, `_resolve_fill_token_id`, `_build_initial_mask_state`; `main()` snapshots `sequence_length` and overrides `diffusion_steps` when enabled. |
| `src/biom3/backend/device.py` | Added `reset_peak_memory_stats`, `get_peak_memory_stats`, `device_sync`, `get_device_name`, `get_device_info`. |
| `src/biom3/Stage3/callbacks.py` | Re-imports the memory helpers from `backend.device` under the existing private names; removed duplicate definitions. |
| `scripts/benchmark_generation.py` | New — sweep harness. |
| `scripts/plot_benchmark.py` | New — post-benchmark plotting (4 figures). |
| `configs/benchmark/stage3_generation_example.json` | New — example sweep config. |
| `tests/stage3_tests/test_pre_unmask.py` | New — 11 unit tests on initial-mask construction + config validation. |

## Verification

- `pytest tests/stage3_tests/test_pre_unmask.py -v` — 11/11 pass.
- `pytest tests/stage3_tests/test_callbacks.py -v` — 26/26 pass (existing
  `mock.patch` calls against `biom3.Stage3.callbacks._get_peak_memory_stats`
  still work because the re-import keeps the module-level name).
- `pytest tests/ --quick` — 479 passed, 135 skipped (weight/GPU-gated,
  unrelated), 0 failures.
- End-to-end smoke run on Spark (GB10): `N ∈ {4,8}, P=[2], B=[4], D=8,
  token_strategy=sample` — 2 records written, `results.npz` loads,
  `env.json` correctly names the GPU.
- Plotting smoke: 44-record sweep → 4 PNGs written to `images/`, all
  readable and sensible.
