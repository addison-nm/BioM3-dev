# Training Run Viewer merge + symlink-fix reconciliation

**Date:** 2026-04-23
**Branch:** `addison-dev`

## Summary

Wrapped the Training Run Viewer feature into `addison-dev`. Feature
development happened in the worktree `.claude/worktrees/training-viewer/`
on branch `addison-training-viewer` (2 commits) and is fully documented in
[2026-04-20_training_run_viewer.md](2026-04-20_training_run_viewer.md).

Partway through development, a latent bug was found in the shared file
picker and fixed on `addison-dev` directly. The fix interleaved with
`addison-generation-benchmarking` work landing on the same branch and a
`HEAD~1` reset — this briefly looked like the fix had been lost to an
orphaned `addison-mithril` branch. `git branch --contains` confirmed
otherwise: the fix is on `addison-dev`, just earlier in topological
history than a `git log -6` glance suggests. No porting needed.

## Commits now on `addison-dev`

```
ef062e9  fix(app): follow symlinks when browsing data directories
3e5f8fe  feat(viz): add stage3_training_runs loader and plotters
71e5c68  feat(app): add Training Run Viewer page
2146332  Merge branch 'addison-training-viewer' into addison-dev
```

## The symlink fix (`ef062e9`)

`src/biom3/app/_data_browser.py::list_files` used `Path.rglob("*")`, which
in Python 3.12 does not descend into directory symlinks.
`scripts/sync_weights.sh` (the convention for worktrees, per
`CLAUDE.md`) populates `weights/` with per-entry symlinks. Result: every
page using `pick_pt` / `pick_pdb` / `pick_file` returned zero files from
`weights/` in a worktree setup — a silent failure in the recommended
workflow.

Swapped to `os.walk(..., followlinks=True)`; non-recursive mode was
unchanged. Added [tests/viz_tests/test_data_browser.py](../../tests/viz_tests/test_data_browser.py) with 5
tests including a symlink regression test.

**Downside now visible:** "Weights → (all, recursive)" returns thousands
of `.pt` files on fully-synced trees (4,770 on the dev machine, mostly
from sharded DeepSpeed checkpoints). The existing substring filter +
subfolder-scope selector handle this; treated as newly-visible scale, not
a regression.

## Branch-topology clarification

`addison-mithril` briefly looked like it was holding an orphaned copy of
the symlink fix. It is not. The branch points at `ef062e9` because it was
created off `addison-dev` shortly after the fix landed, and its worktree
has separate uncommitted work adding a new HPC machine named "Mithril"
alongside Spark/Polaris/Aurora:

```
Modified:  CLAUDE.md, README.md, docs/stage3_training.md, environment.sh
New:       configs/stage3_training/machines/_mithril.json
           docs/setup_mithril.md
           jobs/mithril/
```

Not this session's concern — leave it to its own branch lifecycle.

## Outstanding

- **Human-eyes browser test** of page 10 (`Training Run Viewer`) against a
  real run, e.g. `weights/ProteoScribe/ft_LEGACY_SH3/LEGACY_SH3_bench_ft16_bs16/artifacts/`.
  AppTest coverage exercised every widget path without exceptions, but
  layout / colors / legend placement need human review before being
  considered production-ready.
- **Worktree cleanup.** `.claude/worktrees/training-viewer/` can be
  removed once satisfied with the merge:
  `git worktree remove .claude/worktrees/training-viewer`.

## Follow-ups deferred

- Multi-run comparison (overlay learning curves from multiple
  `artifacts/` dirs).
- `use_container_width=True` → `width="stretch"` — Streamlit deprecated
  the former in 2025-12; codebase-wide migration, not ad-hoc.
- Smoothing / moving-average overlay for noisy training curves.
- Export metric arrays to CSV.
