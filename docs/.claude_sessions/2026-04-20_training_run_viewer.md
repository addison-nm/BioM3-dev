# Stage 3 Training Run Viewer — webapp page

**Date:** 2026-04-20
**Branch:** `addison-training-viewer` (worktree at `.claude/worktrees/training-viewer/`)

## Summary

Added a 10th Streamlit page to the BioM3 webapp for visualizing Stage 3
pretraining and finetuning runs. The page reads a run's `artifacts/`
directory and renders: best-checkpoint summary, hyperparameters grouped by
category, learning curves (any of ~12 metric bases × step / epoch x-axis ×
train/val splits, with optional per-rank `val_loss` overlay), throughput and
per-epoch memory benchmarks, and a raw-metric explorer for anything the
curated UI misses.

Everything builds on the viz-module pattern established in
[dynamics.py](src/biom3/viz/dynamics.py): a typed
`load_run_artifacts` + `plot_metric(_from_dir)` surface returns
`matplotlib.figure.Figure`, then the Streamlit page composes widgets around
it. No changes to `MetricsHistoryCallback` or training internals — the page
consumes the existing artifact layout as-is.

## Commits

None yet — user will review and commit. Files touched below.

## Files changed

### `src/biom3/viz/stage3_training_runs.py` *(new)*
Loader and plotters for a Stage 3 run's `artifacts/` directory.

- `load_run_artifacts(artifacts_dir)` — reads `metrics_history.pt` (required),
  all `metrics_history.rank*.pt` siblings (glob), and the JSON metadata files
  (`args.json`, `checkpoint_summary.json`, `build_manifest.json`,
  `benchmark_history.json`). Returns a normalized
  `{"metrics": {split: {base: {"by_step": (x, y), "by_epoch": (x, y)}}}}`
  view.
  - Base names are derived by stripping `{split}_` prefix and `_step`/`_epoch`
    suffix. Split-native keys like `loss_gap` survive unchanged.
  - Bare `train_loss` and `train_loss_step` are collapsed (the callback writes
    them identically).
  - Non-finite values are masked before storage so matplotlib doesn't break
    lines on NaN/inf (common with early-epoch val losses under some
    schedulers).
  - Val has no `_step` keys but `val["global_step"]` *is* recorded — the
    loader uses it to populate `by_step` anyway, so val renders as sparse
    markers on a step x-axis.
- `discover_metric_bases(run_data)` — sorted base names per split.
- `plot_metric(run_data, metric_base, ...)` — one plot with selectable
  splits, x-axis, log-y, and per-rank overlay (only activated when
  `metric_base == "loss"` and rank files exist).
- `plot_benchmark_history(run_data, kind="throughput"|"memory", per_rank=...)`
  — per-epoch samples/sec + steps/sec (twin-y) or peak allocated/reserved
  memory, with optional per-rank memory overlay when
  `peak_memory_allocated_gb_per_rank` lists are present.
- `plot_metric_from_dir` / `plot_benchmark_from_dir` convenience wrappers
  mirror the `plot_X_from_file` pattern in dynamics.py.

### `src/biom3/viz/__init__.py`
Re-exports the six new public symbols alongside the existing viz surface.

### `src/biom3/app/pages/10_Training_Run_Viewer.py` *(new)*
Page wiring, broken into small private renderers. Layout:

1. **Run picker** — custom `_pick_run_artifacts()` (two selectboxes: data
   directory + discovered `artifacts/` folder, with substring filter).
   Bypasses the shared `browse_file` because its `Path.rglob()` does not
   follow symlinks, and worktree `weights/` is a symlink tree (see "Bugs
   noted" below). The custom walker uses `os.walk(followlinks=True)`.
2. **Summary card** — two columns: best-checkpoint metric/score/path on the
   left; git branch/hash, seed, total/trainable params, timestamp, hostname
   on the right.
3. **Hyperparameters expander** — three-column whitelist (Optimization /
   Model / Run) with a nested "All args (raw)" expander (`st.json`) for the
   ~90 remaining keys.
4. **Learning curves** — metric selectbox (default `loss`), x-axis radio,
   train/val/log-y/show-by-rank checkboxes. When "Show by rank" is on and
   no rank files exist, shows `st.info` explaining how to enable them for
   future runs.
5. **Benchmarks** (only renders when `benchmark_history.json` present) —
   throughput + memory tabs; per-rank memory checkbox that auto-disables
   when per-rank arrays are missing and warns above 16 ranks (these runs
   went up to 96 ranks, which crowds the plot).
6. **Raw metrics explorer** — flat selectbox over all raw keys (escape
   hatch for anything the curated metric-base list misses).

Bare-mode safety: whole page body wrapped in `render()` with explicit
`return` on early exits, per the webapp CLAUDE.md convention.

Caching: `@st.cache_data` keyed on `(str(artifacts_dir), metrics_mtime)`
wraps `load_run_artifacts` so widget interactions don't re-parse the `.pt`.

### `tests/viz_tests/test_stage3_training_runs.py` *(new)*
18 unit tests, all passing. Notable:

- `test_val_sparse_on_step_axis` — asserts the loader populates `by_step`
  for val even when val only has `_epoch` keys. Most likely regression site
  per the plan-agent critique.
- `test_non_finite_masked` — injects a NaN into val_loss, asserts the
  plotted line has no NaN values after masking.
- `test_rank_overlay_ignored_for_non_loss` — ranks=[0] on `prev_hard_acc`
  produces no rank lines (rank files only carry val_loss).
- Synthetic artifacts built via `tmp_path` fixture — no real weights needed.

## Verification

1. **Unit tests:** `PYTHONPATH=src pytest tests/viz_tests/` — 65 passed, 2
   skipped (existing skips), 18 new tests all pass.
2. **App smoke-import** — all 10 pages import OK, including the new one.
3. **Headless AppTest** — exercised via `streamlit.testing.v1.AppTest`:
   - Initial render with no run picked: no exceptions.
   - After filtering to `bench_ft16_bs16` and auto-loading: 4 subheaders
     render, 2 dataframes, best val_loss metric (0.1385), legend contains
     `train` + `val` lines. No exceptions or errors.
   - Metric selectbox toggles between all 12 bases (`loss`, `loss_gap`,
     `prev_hard_acc`, etc.) without error.
   - X-axis / log-y / train-only / val-only toggles all re-render cleanly.
   - "Show by rank" on a run without rank files shows the `st.info` hint
     and no overlay is attempted.
4. **Live browser test** — *not performed* (user stepped away). The AppTest
   coverage above exercises every code path exposed by the widgets; a
   human-eyes pass on layout, color choices, and plot readability is still
   outstanding. **Flag for user to do before merging.**

## Bugs noted (not fixed in this worktree)

- **`browse_file` does not follow symlinks.**
  `src/biom3/app/_data_browser.py::list_files` uses `Path.rglob("*")` which
  (in Python 3.12) does not descend into directory symlinks. All existing
  pages that call `pick_file`/`pick_pdb`/`pick_pt` are affected when their
  data lives under a symlinked tree — and `scripts/sync_weights.sh`
  deliberately produces exactly that in worktrees. On the main checkout
  this may be masked if `weights/` holds direct directories rather than
  symlinks, but on any worktree (the CLAUDE.md-recommended workflow) the
  shared picker cannot see weight files. Per CLAUDE.md's worktree rule,
  this fix belongs to its own branch. The new page ships a local
  workaround (`os.walk(followlinks=True)`) rather than consuming the
  broken helper. Filed as a follow-up.

## Design notes

- **Reuse over rebuild.** `MetricsHistoryCallback` already normalizes
  everything the UI needs; the viz module is a thin layer on top. The page
  only adds widgets, not plotting logic.
- **Suffix vs. x-axis availability.** The plan-agent critique flagged that
  keying cadence on suffix alone would mis-handle val (which has no
  `_step` keys). The loader instead populates both `by_step` and
  `by_epoch` whenever the corresponding index array exists, so val renders
  as sparse markers on a step x-axis — which is the actual per-epoch
  behavior users want to see.
- **Rank overlay is narrow.** The `metrics_history.rank*.pt` files only
  contain `val_loss` (they're a sync_dist diagnostic from a specific bug
  fix round, not a general per-rank history). The UI reflects this: rank
  overlay only activates for `metric_base == "loss"` + `val` selected.
  When the user checks "Show by rank" on other metrics, an inline caption
  explains why.
- **Mixed-type DataFrames must be stringified before `st.dataframe`.** The
  `benchmark_history.config` and build-manifest info blocks mix str + int
  values; pyarrow (used under the hood by `st.dataframe`) chokes on
  `"bf16"` in an int column. All metadata dicts go through `str(v)` before
  being boxed into a DataFrame.
- **96-rank runs produce crowded memory plots.** The user-requested
  per-rank overlay works at any rank count, but with 8 nodes × 12 GPUs =
  96 ranks, `per_rank=True` paints 98 lines. The page warns above 16
  ranks; leaving the checkbox off is the default for high-rank runs.

## Follow-ups considered but not done

- **Multi-run comparison** (overlay learning curves from several
  `artifacts/` directories). User asked about this up front; deferred to
  a follow-up now that single-run + per-rank overlay is stable.
- **Fix `_data_browser.list_files` to follow symlinks.** Separate branch
  per CLAUDE.md worktree convention. The fix itself is one-line:
  `recursive=True` → switch `Path.rglob("*")` to `os.walk(...,
  followlinks=True)` or equivalent `scandir` walk. Consider also what
  depth to cap at, since `weights/` symlinks can lead into very large
  sharded checkpoint trees.
- **`use_container_width=True` → `width="stretch"`.** Streamlit deprecated
  the former in 2025-12; the codebase uses `use_container_width` in
  several pages. Codebase-wide migration, not ad-hoc here.
- **Smoothing / moving-average overlay** for noisy training curves.
- **Export metric arrays to CSV.** Trivial if asked.
- **Live browser eyeball pass** on the rendered layout (colors, spacing,
  legend placement) — still needed before merge.
