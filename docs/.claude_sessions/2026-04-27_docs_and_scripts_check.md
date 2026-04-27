# 2026-04-27 — docs and scripts check (5-workstream bundle)

Driven by [2026-04-27_prompt_docs_and_scripts_check.md](../../2026-04-27_prompt_docs_and_scripts_check.md). Plan file at `~/.claude/plans/i-have-included-a-crispy-kettle.md`.

## What landed

### W1 — PenCL OOM mitigation
- Added `--cross_comparison_sample_limit` (int, default `-1`) to [src/biom3/Stage1/run_PenCL_inference.py](../../src/biom3/Stage1/run_PenCL_inference.py).
- Slices `z_p_tensor` / `z_t_tensor` to the first `k = min(limit, n)` rows before the O(n²) cross-comparison block (dot product, two softmaxes, `compute_homology_matrix`). Magnitudes (O(n)) still computed over the full tensor.
- New log line reports `k of n` samples used. Saved `embedding_dict` is unchanged.
- Docstring updated with a fourth example showing the OOM-mitigation use case.

### W4a — wandb arg precedence (real bug fix)
- New [scripts/_wandb_resolve.sh](../../scripts/_wandb_resolve.sh) implements the precedence matrix:
  - `--wandb False` always honored (user opt-out wins).
  - `--wandb True` requires `WANDB_API_KEY`; errors fast at the wrapper if unset.
  - No `--wandb` → defaults to True iff `WANDB_API_KEY` is set, else False (with warning).
- All 6 `scripts/stage{1,2,3}_train_{single,multi}node.sh` wrappers now `source _wandb_resolve.sh "$@"` and emit `${wandb_resolved}` in place of the old inline `${wandb_override}` block.
- All 35 templates under `jobs/{aurora,polaris,spark}/_template_*` and 19 concrete `jobs/*/job_*` files: replaced hardcoded `--wandb True` with a configurable `use_wandb=True` variable next to `epochs=`, passed as `--wandb ${use_wandb}`.
- Two stage1 pfam job files (`job_stage1_pretrain_pfam_v1_n{1,2}.pbs`) had a stale **positional** `"${wandb_api_key}"` argument from the pre-refactor signature — removed (was silently breaking the wrapper signature).
- Original bug: templates' `--wandb True` came via `"$@"` after `${wandb_override}=--wandb False`, so argparse took the last value and ignored the `WANDB_API_KEY`-empty case.

### W3 — sync script rename
- `scripts/sync_databases.sh` → [scripts/link_data.sh](../../scripts/link_data.sh) (via `git mv`). Header rewritten to be data-source-agnostic (databases or datasets), explicitly documents the load-bearing top-level-files behavior.
- `scripts/sync_weights.sh` → [scripts/link_weights.sh](../../scripts/link_weights.sh). Header refreshed with new name; explicitly documents the **deliberate** difference vs `link_data.sh`: link_weights.sh does NOT link top-level files at the source root (weights live under per-component subdirs).
- Logic of both scripts unchanged.
- All current doc/code references updated: `README.md`, `CLAUDE.md`, `weights/README.md`, `docs/setup_databases.md`, `docs/setup_shared_weights.md`, `docs/dbio_examples.md`, `docs/building_datasets_with_dbio.md`, `demos/build_sh3_dataset.sh`, `demos/build_source_datasets.sh`, `src/biom3/app/_data_browser.py`, `src/biom3/app/pages/10_Training_Run_Viewer.py`, `tests/viz_tests/test_data_browser.py`. Historical session logs and the original prompt left untouched.
- README "Reference databases" section now also shows the datasets use case.

### W2 — `limit_[val,train]_batches` reconciliation
- Stage 3 argparser default `--limit_val_batches`: `0.05` (fraction) → `200` (absolute). Help text rewritten to document the int>1=absolute, fraction(0,1] convention. Stage 1 and Stage 2 helps were also fleshed out (defaults unchanged).
- **Real bug found and fixed**: only `limit_train_batches` had the `int(x) if x>1 else float(x)` coercion before being passed to PL Trainer. `limit_val_batches` was passed raw — so the new default of `200` (with `type=float` → `200.0`) would have crashed PL Trainer (which rejects floats >1.0 for `limit_*_batches`).
- Extracted the coercion into [src/biom3/core/helpers.py:`coerce_limit_batches`](../../src/biom3/core/helpers.py); replaced 5 inline copies (Stage 1 train, Stage 1 val, Stage 2 val, Stage 3 train, Stage 3 val).
- New unit test [tests/core_tests/test_helpers.py](../../tests/core_tests/test_helpers.py) — 9 parametrized cases covering fraction (0.05, 0.5, 1.0), int (1, 2, 200), float >1 (200.0, 5.7), None.
- New end-to-end coverage: [tests/_data/entrypoint_args/training/training_args_scratch_v1_abs_val.txt](../../tests/_data/entrypoint_args/training/training_args_scratch_v1_abs_val.txt) sibling to `training_args_scratch_v1.txt` with `--limit_val_batches 5` and `--limit_train_batches 5`. Added as a second parametrize case to `test_train_from_scratch`.
- Tuning-guidance section added to [docs/stage3_training.md](../../docs/stage3_training.md) — a 4-row recipe table covering small/large dataset and step-based scenarios. Also updated the `limit_val_batches` row in "Key config parameters".

### W5 — argparser audit + docstring + CLI_reference + README
- Audited 7 entrypoints (Stage{1,2,3} training + Stage{1,2,3} inference + embedding_pipeline). Inference scripts and embedding_pipeline already well-documented; training scripts had the gaps.
- Stage 1 training argparser: added help text to **22** previously-undocumented args.
- Stage 2 training argparser: added help text to **all 53** previously-undocumented args (including the missing help on `--config_path` itself).
- Stage 3 training: argparser help was already adequate; rewrote the 4-line module docstring into ~50 lines with config location, precedence rules, and 3 example invocations (pretrain / continue with secondary / finetune).
- New [docs/CLI_reference.md](../../docs/CLI_reference.md) (~380 lines) — consolidated reference for all 7 entrypoints. Per-entrypoint synopsis, required/optional argument tables, source link, config-file pointer, representative example. Top-level note on `_base_configs` composition. Dedicated wandb-handling section reproducing the W4a precedence matrix. Bottom note pointing readers to `--help` as source of truth if drift occurs.
- README.md: added "CLI reference" callout at top of Usage; replaced 3 inline arg tables (Stages 1/2/3 inference) with one-line cross-links. Also fixed a broken pre-refactor wrapper invocation example (was passing `$WANDB_API_KEY` as positional arg, now uses `--wandb True`).
- docs/stage3_training.md and docs/embedding_pipeline.md: added top-of-doc cross-link blockquotes to CLI_reference.md.

## Things to look back into

These were intentionally left for follow-up (per user instruction: "I'm not worried about any of these not landing on this commit"):

1. **W4b — Aurora single-node generation CPU binding.** [scripts/launchers/aurora_singlenode.sh](../../scripts/launchers/aurora_singlenode.sh) hardcodes a 12-tile bind (8 cores per tile). For sequence-generation jobs that effectively use all 12 tiles as one logical pool, the strict per-tile binding may underutilize each tile. Plan: measure `biom3_ProteoScribe_sample` wall time + per-tile utilization with the existing binding vs a relaxed binding (e.g. `--cpu-bind depth -d 16` or no binding); if speedup is real, add a second launcher `aurora_singlenode_generation.sh` (or a `BIOM3_CPU_BIND_MODE` env var) for inference workloads. Do NOT change the default training binding (it's load-bearing for Stage 3 + oneCCL `CCL_WORKER_AFFINITY` in environment.sh:42). Documented in [docs/setup_aurora.md](../../docs/setup_aurora.md) when concrete recommendations land. Marked as `*(DEFERRED — return to this later)*` in the plan file.

2. **W2 empirical measurement.** New `--limit_val_batches=200` Stage 3 default was chosen heuristically (200 batches × `batch_size=16` ≈ 3200 samples per val pass — fast on 10M datasets, coverage-complete on ~10K datasets). Should be confirmed with two short Stage 3 runs (small dataset ~10K vs benchmark hdf5 ~1M+) measuring per-epoch wall time and val-loss variance.

3. **W2 production configs.** All 7 production Stage 3 training configs in `configs/stage3_training/` (`pretrain_scratch_v{1,2,3}.json`, `pretrain_start_pfam_v{1,2}.json`, `finetune_v{1,2}.json`) explicitly set `limit_val_batches: 0.05`. Per the user's instruction, these were left as-is — so the new argparser default has zero practical effect on existing flows. Three options when revisiting: (a) update in place (changes reproducibility of v1/v2/v3 configs), (b) bump versions (e.g. add `pretrain_scratch_v4.json` with new value, preserve historical configs), (c) leave indefinitely (status quo for old runs; the new default + docs guide future configs).

4. **Stringified-bool pattern.** Many training args use `type=str` with default `'True'`/`'False'` (converted via `str_to_bool` later in the script): `--wandb`, `--finetune`, `--start_secondary`, `--scale_learning_rate`, `--save_metrics_history`, etc. This works but is non-idiomatic. A future cleanup could convert them to `argparse.BooleanOptionalAction` or `type=str_to_bool` so `--help` shows the correct usage and bad values fail at parse time. Out of scope for this round.

5. **Legacy / dead args (Stage 3).** From the audit:
   - `--task` (default `'MNIST'`) — no longer relevant for a protein-only codebase.
   - `--checkpoint_dir` / `--checkpoint_prefix` — appear superseded by the `output_root/checkpoints_folder/run_id` layout.
   - `--start_pfam_trainer` — explicitly deprecated alias of `--start_secondary`; consider removing once no configs reference it.
   - `--swissprot_data_root` / `--pfam_data_root` — explicitly deprecated aliases of `--primary_data_path` / `--secondary_data_paths`; same removal candidate.
   - Code review needed before removal — these may still be referenced in test args files or live configs.

6. **Stage 2 deprecation alias.** `--num_gpus` exists alongside `--gpu_devices` for `Facilitator_Dataset` device logic (line 115). Worth checking whether the duplication is still needed or whether `Facilitator_Dataset` can read `--gpu_devices` directly.

7. **dbio + benchmark + app entrypoints.** CLI_reference.md scope was the **core 7** per the user's plan-mode answer. The remaining ~16 entrypoints in `pyproject.toml` (`biom3_build_*`, `biom3_benchmark_*`, `biom3_app`, `biom3_compile_hdf5`, `biom3_plot_benchmark`) are not in the new reference. dbio is partially covered by `docs/dbio_examples.md`. A future expansion of CLI_reference.md could cover them.

## Tests added

- [tests/core_tests/test_helpers.py](../../tests/core_tests/test_helpers.py) — `coerce_limit_batches` parametrized unit test (no GPU/data deps; runs under `--quick`).
- [tests/_data/entrypoint_args/training/training_args_scratch_v1_abs_val.txt](../../tests/_data/entrypoint_args/training/training_args_scratch_v1_abs_val.txt) — Stage 3 entrypoint args file exercising the absolute-count branch end-to-end.
- `test_train_from_scratch` parametrized to cover both fractional and absolute-count paths.

## Files changed (summary)

- New: `scripts/_wandb_resolve.sh`, `docs/CLI_reference.md`, `tests/core_tests/test_helpers.py`, `tests/_data/entrypoint_args/training/training_args_scratch_v1_abs_val.txt`, this session note.
- Renamed: `scripts/sync_databases.sh` → `scripts/link_data.sh`; `scripts/sync_weights.sh` → `scripts/link_weights.sh`.
- Modified: ~70 files (3 stage entrypoints, 6 wrapper scripts, 35 templates, 19 concrete jobs, README, CLAUDE.md, ~6 docs, 2 demos, 3 src python files for docstring updates, 1 test file for parametrize, 3 stage training scripts for argparser help-text additions).
