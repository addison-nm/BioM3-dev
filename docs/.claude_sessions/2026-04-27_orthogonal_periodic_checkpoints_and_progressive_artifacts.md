# 2026-04-27 — Orthogonal periodic checkpoints + progressive artifact saves

Pre-version-bump improvements to checkpointing and artifact persistence.
Two distinct features landed in one branch session, plus a small back-port
of metric-history plumbing to Stage 2.

## Context / why

A collaborator (Sophie) had attempted a periodic-snapshot feature on her own
fork and shared `run_PL_training.sophie.py` for reference. Reading it surfaced
a latent bug in our existing code: `--checkpoint_every_n_steps` /
`--checkpoint_every_n_epochs` were being attached to the *primary monitored*
`ModelCheckpoint` (the one with `save_top_k=2`), so periodic-trigger saves
were silently competing with — and being pruned by — the best-by-`val_loss`
ranking. The "periodic" knobs weren't actually periodic in the time-series
sense; they just changed the trigger cadence for the same top-2 buffer.

Separately, we noticed that several artifacts (`state_dict.best.pth`,
`metrics_history.pt`, `benchmark_history.json`, `checkpoint_summary.json`)
were only being written by `save_model()` *after* `trainer.fit()` returned —
which means a SIGKILL/timeout mid-run wiped them, and recovering the best
weights required manually running `convert_zero_checkpoint_to_fp32_state_dict`
on the surviving `.ckpt`.

This session fixed both.

## What shipped

### Feature 1 — Orthogonal monitored vs periodic checkpointing

| Change | File | Why |
|---|---|---|
| New `build_checkpoint_callbacks(...)` helper | `src/biom3/Stage3/callbacks.py` | Returns `(monitored_callbacks, periodic_callback)`. Monitored callbacks no longer carry `every_n_*` cadences. Periodic snapshots go to a separate `<checkpoint_dir>/periodic/` subdirectory with `monitor=None`. |
| Helper validates step-vs-epoch mutual exclusion | same | Lightning enforces this at `ModelCheckpoint.__init__`; we surface the error at helper-call time instead of at Trainer construction. |
| Helper validates `periodic_max_keep ∈ {-1, 0, 1}` | same | Lightning's `monitor=None` requires `save_top_k` in this set. Custom "keep last N" retention would need a separate pruning layer; out of scope. |
| Stage 3 wired to helper | `src/biom3/Stage3/run_PL_training.py` | Collapsed ~50 lines of inline checkpoint construction into one helper call. Preserved the `combine`-strategy default `every_n_steps = log_every_n_steps`. |
| `save_model(...)` now passes `monitored_callbacks[1:]` (not `checkpoint_callbacks[1:]`) | same | Periodic snapshots are kept as raw `.ckpt` and *not* run through the DeepSpeed→fp32 converter (avoids huge unnecessary work on every periodic snapshot at end-of-train). |
| Stage 1 + Stage 2 back-ported to use the helper | `src/biom3/Stage1/run_PL_training.py`, `src/biom3/Stage2/run_PL_training.py` | Same orthogonal-split pattern; same bug was latent in both. Stage 2 also gained `--checkpoint_every_n_epochs` (was steps-only). |
| Stage 2 gained `--metrics_history_every_n_epochs` and `--metrics_history_flush_every_n_steps` | Stage 2 | Plumbing parity with Stages 1 + 3. |
| New CLI args on all three stages: `--checkpoint_periodic_max_keep` | all three | `argparse` `choices=[-1, 0, 1]` rejects invalid values at parse time with help text explaining the constraint. |

### Feature 2 — Progressive (timeout-resilient) artifact saves

| Change | File | Why |
|---|---|---|
| `MetricsHistoryCallback.on_validation_epoch_end` now unconditionally fsyncs JSONL streams *and* writes consolidated `metrics_history.pt` | `callbacks.py` | After every val cycle there's a complete `.pt` snapshot on disk; at most one val cycle's worth of metrics is at risk on SIGKILL even with `flush_every_n_steps=None` default. |
| `TrainingBenchmarkCallback.on_train_epoch_end` (and `combine`-strategy `on_validation_epoch_end`) now writes `benchmark_history.json` per epoch | same | Was only at `on_train_end`; now per-epoch. |
| Demoted both `_save` log lines to `DEBUG` | same | Per-epoch saves would otherwise spam `run.log`. End-of-train INFO summary lines preserved. |
| Extracted `_convert_or_copy_checkpoint(...)` and `_sync_best_artifact(...)` helpers from `save_model` body | `run_PL_training.py` (Stage 3) | Idempotent: re-runnable mid-training. Returns `checkpoint_summary` dict. |
| `save_model` refactored to delegate best-side to `_sync_best_artifact`; last-side and symlinks stay inline | same | The "last" model handling only makes sense at end-of-train. |
| New `BestArtifactSyncCallback` in `callbacks.py` | `callbacks.py` | Callable-driven (sync_fn injected), so `callbacks.py` stays model-agnostic. Hooks `on_validation_epoch_end`, runs only when `primary_callback.best_model_path` has changed since the last sync. Throttled by `every_n_val`. Failures are warned, not raised. |
| New CLI args: `--artifact_sync_on_best` (default True), `--artifact_sync_every_n_val` (default 1) | `run_PL_training.py` (Stage 3) | Disable knob in case the DeepSpeed→fp32 conversion is too costly for a given workload. |

### Tests added

In `tests/stage3_tests/test_callbacks.py`:

- `TestBuildCheckpointCallbacks` (10 tests) — orthogonal-split semantics, step-vs-epoch mutual exclusion, `periodic_max_keep` legal/illegal values, JSON-string monitor parsing, sync-safe class swap.
- `TestProgressiveMetricsHistorySnapshot` (2 tests) — `metrics_history.pt` exists after a single val-epoch end; JSONL fsynced at val end even with default flush settings.
- `TestProgressiveBenchmarkHistorySnapshot` (1 test) — `benchmark_history.json` exists after one train-epoch end.
- `TestBestArtifactSyncCallback` (7 tests) — skips when no best yet, runs on first best, idempotent when best unchanged, re-runs when path changes, throttle correctness, exceptions don't propagate AND don't poison the retry, non-zero ranks skipped.

Result of `pytest tests/stage3_tests/test_callbacks.py -v`: 57/57 passing.

## Sequence of decisions

1. **Audit existing code first.** Stage 1 already had `metrics_history_every_n_epochs` and the `flush_every_n_steps` plumbing; Stage 3 had it; Stage 2 was missing both. Stage 3 had the silent `every_n_*`-attached-to-top-k bug. The pattern Sophie discovered (separate periodic callback) was the right shape — but her version dropped `flush_every_n_steps` and other recent plumbing, so we adapted the idea rather than merging her file.
2. **Orthogonalize before back-porting.** Built the `build_checkpoint_callbacks` helper in Stage 3, then back-ported Stage 1 and Stage 2 to call into it. This way the bug fix happens once and stays consistent.
3. **Validate Lightning's hidden constraints upstream.** Two of them bit us during development:
   - `every_n_train_steps` and `every_n_epochs` are mutually exclusive on `ModelCheckpoint`. Helper now raises `ValueError` with a clear message instead of letting it surface as a `MisconfigurationException` at Trainer construction.
   - `monitor=None` requires `save_top_k ∈ {-1, 0, 1}`. Helper validates; argparse `choices` enforces at parse time.
4. **Three-tier artifact-resilience scope.** User asked for "all 3" — Tier 1 (cheap progressive snapshotting), Tier 2 (best-state-dict mid-training sync via extracted helper + new callback), Tier 3 (re-emit `checkpoint_summary.json`, free with Tier 2).

## Things to watch out for going forward

- The mid-training sync hook passes `expected_dtype=None` to `_sync_best_artifact` because `PL_model.dtype` would require the model object at trainer-construction time. The end-of-train `save_model` call still passes the strict `expected_dtype=PL_model.dtype` so the dtype assertion still fires at the original spot.
- `--checkpoint_periodic_max_keep` only accepts `-1`, `0`, `1`. If/when "keep last N" becomes important, that needs either a custom pruning callback or a wrapper around `ModelCheckpoint` that injects a synthetic ranking metric.
- `BestArtifactSyncCallback` swallows exceptions intentionally — failed sync warns and retries on the next val cycle. The `.ckpt` is always preserved by `ModelCheckpoint` itself, so a sync failure is recoverable manually.
- Sophie's `run_PL_training.sophie.py` remains untracked in the repo root. Not deleted in this commit; ours diverges enough that there's no merge path back.

## Open follow-ups

1. Stale `_jobscript.sh` and `2026-04-27_xccl_hang_recap.md` still in repo root (carried over from prior session). Untouched here.
2. Same orthogonal-checkpoint pattern is now consistent across Stages 1/2/3 — when adding a Stage 4 (or any new training entrypoint), reuse `build_checkpoint_callbacks` from `Stage3.callbacks`.
3. The `mid-training best-artifact sync` is currently Stage 3 only because `_sync_best_artifact` lives in `Stage3/run_PL_training.py` and depends on `mod.get_model` + `PL_mod.PL_ProtARDM`. Replicating for Stage 1/2 would mean either extracting per-stage `_sync_best_artifact` functions or generalizing the helper to take a model-builder callable. Not done in this session.
4. Version bump (`0.1.0a4` → `0.1.0a5`?) is the next logical step now that this work is in.
