# 2026-05-12 — Stage 3 finetune-resume robustness + log hygiene + rolling artifact backups

## Context

Diagnosing a stack of small operational issues with Stage 3 finetuning runs on
Aurora (luciferase / `fluc_enriched`). Triggered by a session-start note,
`best_artifact_sync_on_resume.md`, that described a `BestArtifactSyncCallback`
failure when resuming into a fresh `run_id` and proposed a one-line fix inside
the callback's sync function. Pull-thread led to four orthogonal cleanups, all
landing on `addison-dev`:

1. `checkpoint_dir` missing-parent on resume into a new `run_id`.
2. `build_manifest.json` and `args.json` only existing on clean completion,
   so killed runs left no provenance on disk.
3. `.o` log files exploding to 78 MB / 500k+ lines from tqdm-in-non-TTY +
   repeated c10d barrier warnings.
4. `state_dict.best.pth` (~345 MB) and `checkpoint_summary.json` accumulating
   `.bak.<timestamp>` chains on every mid-training sync — ~2.8 GB of stale
   redundancy per long run.

A jobscript-organization side-quest (move root-level luciferase .pbs into
`jobs/aurora/`, add an 8-node sibling) landed in the same commit window.

State before session began:

```bash
git checkout 2edbb69  # docs(dbio): document Pfam multi-input and per-pfam output
```

## Code changes

All in [`src/biom3/Stage3/run_PL_training.py`](src/biom3/Stage3/run_PL_training.py)
plus two new jobscripts.

### 1. `checkpoint_dir` upfront makedirs

[run_PL_training.py:1742](src/biom3/Stage3/run_PL_training.py#L1742) — added
`os.makedirs(checkpoint_dir, exist_ok=True)` to the rank-0 setup block next to
the existing `logs_dir` / `artifacts_dir` mkdirs. Establishes the run-output
dir invariant in one place. The doc's suggested fix (one-line `makedirs`
inside `_sync_best_artifact`) addressed the symptom; this addresses the root
cause — Lightning's lazy dir creation was masking the missing-invariant for
fresh runs, only failing on resume into a new run_id.

### 2. Two-file split for run provenance

Two new helpers before `main()`:

- `_write_build_manifest(args, artifacts_dir, checkpoint_dir, PL_model, start_time)`
  ([run_PL_training.py:1619-1689](src/biom3/Stage3/run_PL_training.py#L1619-L1689))
  — writes `args.json` + `build_manifest.json` with everything except elapsed
  time (passed as `timedelta(0)` placeholder). Called immediately before
  `train_model()`, so the manifest lands on disk even if training is killed
  mid-run. Rank-0 guarded.
- `_write_run_summary(artifacts_dir, start_time, exit_reason, exception=None)`
  ([run_PL_training.py:1692-1710](src/biom3/Stage3/run_PL_training.py#L1692-L1710))
  — writes `run_summary.json` with `elapsed_seconds`, `end_time`, `exit_reason`
  (`"completed"` | `"interrupted"` | `"exception"`), and the exception type /
  message when applicable.

`main()` wraps `train_model()` in `try / except KeyboardInterrupt / except
BaseException / finally`
([run_PL_training.py:1846-1870](src/biom3/Stage3/run_PL_training.py#L1846-L1870))
so the run-summary write always fires except under SIGKILL. Build manifest +
args.json are now present even on hard kills.

Rationale for two files over one updated-in-place: `build_manifest` represents
*how the run was built* (immutable, always present); `run_summary` represents
*how the run ended* (best-effort). Splitting avoids the `.bak` noise of writing
the same manifest twice through `backup_if_exists`.

`get_global_rank()` is launcher-env-based in this codebase
(`biom3.core._dist_env`, reads `PALS_RANKID` / `PMI_RANK` / etc.) so the
pre-`fit()` rank-0 guard is reliable without `torch.distributed` initialized —
verified against the existing precedent at [run_PL_training.py:1739-1742](src/biom3/Stage3/run_PL_training.py#L1739-L1742).

### 3. Progress bar auto-off when stdout is non-TTY

- New CLI `--progress_bar {True,False,auto}` (default `auto`)
  ([run_PL_training.py:353-358](src/biom3/Stage3/run_PL_training.py#L353-L358)).
- Normalized in `retrieve_all_args` so `'auto'` → `None`, else
  `str_to_bool` ([run_PL_training.py:1003-1006](src/biom3/Stage3/run_PL_training.py#L1003-L1006)).
- `Trainer(enable_progress_bar=...)` wired through with
  `sys.stdout.isatty()` fallback when arg is `None`
  ([run_PL_training.py:1528-1532](src/biom3/Stage3/run_PL_training.py#L1528-L1532)).

Net: PBS submissions auto-disable the progress bar (no log spam), interactive
sessions keep it. Existing per-epoch metric logging from `MetricsHistoryCallback`
(plus the JSONL streams in `artifacts_dir/`) preserves progress visibility.

### 4. c10d `barrier()` warning — print-once

[run_PL_training.py:1721-1724](src/biom3/Stage3/run_PL_training.py#L1721-L1724)
— added a fourth `warnings.filterwarnings` entry with action `"once"` for the
`barrier(): using the device under current context` message. Originates inside
Lightning's `init_process_group` call (which doesn't forward `device_id` in
2.x); we don't own that call, so the suppression is the only practical fix on
the training side. Run still gets one warning per process (so the issue is
not silently buried), then quiet.

Sampling-side root-cause fix (passing `device_id=` to our own
`dist.init_process_group(...)` in `biom3.core.distributed`) was deliberately
deferred — recorded in user memory `todo_barrier_device_id_warning.md` with
the half that's done now marked DONE.

### 5. `--backup_artifacts` flag with rolling-backup semantics

[run_PL_training.py:331-340](src/biom3/Stage3/run_PL_training.py#L331-L340) —
new arg, default `'False'`. Normalized in `retrieve_all_args`
([run_PL_training.py:1000](src/biom3/Stage3/run_PL_training.py#L1000)).

`_sync_best_artifact` ([run_PL_training.py:653-746](src/biom3/Stage3/run_PL_training.py#L653-L746))
now uses a local `_maybe_backup(fpath)` helper that:

- Returns immediately when `--backup_artifacts False` (no backup, no chain).
- On `True`:
  1. Look up `_BACKUP_HISTORY[fpath]`. If present, `os.remove` the prior
     self-recorded backup path (silently ignore `FileNotFoundError`).
  2. Call `backup_if_exists(fpath)` — returns the new `<name>.bak.<mtime-stamp>`
     path.
  3. Record the new backup in `_BACKUP_HISTORY[fpath]`.

Module-level state `_BACKUP_HISTORY: dict[str, str] = {}`
([run_PL_training.py:124](src/biom3/Stage3/run_PL_training.py#L124)) is
`.clear()`-ed at the top of `main()`
([run_PL_training.py:1726](src/biom3/Stage3/run_PL_training.py#L1726)) so the
history is per-run. Critical safety property: we only ever remove paths
recorded by *this run*. External `.bak.*` files from prior runs or manual
operations are untouched.

Steady-state disk footprint: 1 current + 1 prior version per file. The
existing `backup_if_exists` calls in `save_model` (one-shot last-files block
at end of training) and in `_write_build_manifest` / `_write_run_summary`
(one-shot pre/post training) are intentionally left ungated — they don't
accumulate.

### 6. Aurora jobscripts for luciferase finetuning

Two new files mirroring the existing `jobs/aurora/` conventions, both keeping
the bespoke knobs from the root-level `job_stage3_finetune_v1_ft16_n8_flucbatch16_nowandb.pbs`
(`prefix=fluc_enriched`, `batch_size=16`, `checkpoint_every_n_epochs=10`,
`use_wandb=False`, the absolute fluc-enriched resume / data paths):

- [`jobs/aurora/job_stage3_finetune_v1_ft16_n1_fluc.pbs`](jobs/aurora/job_stage3_finetune_v1_ft16_n1_fluc.pbs)
  — single-node version. `select=1`, calls `scripts/stage3_train_singlenode.sh`.
- [`jobs/aurora/job_stage3_finetune_v1_ft16_n8_fluc.pbs`](jobs/aurora/job_stage3_finetune_v1_ft16_n8_fluc.pbs)
  — 8-node sibling. `select=8`, `num_nodes=8`, calls
  `scripts/stage3_train_multinode.sh` with the extra `${num_nodes}` positional
  per the multinode launcher signature.

Style alignment vs. the root original: `wandb` → `use_wandb`; dropped the
`wandb_api_key=${WANDB_API_KEY:-}` line (unused when `use_wandb=False`, and
the aurora convention is to require the env var to be exported externally).

Root file `job_stage3_finetune_v1_ft16_n8_flucbatch16_nowandb.pbs` was left
in place per user instruction.

Note: both new jobscripts exist on disk under `jobs/aurora/` but were
**not** included in the session commit — user opted to keep them untracked
for now.

## Per-run diagnostic comparison (4 fluc_enriched runs on 2026-05-12)

Each run's `build_manifest.json` presence and content was the diagnostic
window into where each run died:

| run_id  | size .o | args.json | build_manifest | checkpoints/ contents              |
|---------|---------|-----------|----------------|-------------------------------------|
| 142816  | 78 MB / 510k lines | ✗ | ✗ | epoch=34, epoch=35, last.ckpt + 3 .bak chains |
| 144048  | 122 KB  | ✗ | ✗ | (dir does not exist — the doc's bug) |
| 152703  | 123 KB  | ✓ (pre-train) | ✓ | derived only, no raw .ckpt |
| 155153  | 259 KB  | ✓ (pre-train) | ✓ | derived only, no raw .ckpt |

The fact that 152703 and 155153 *have* `build_manifest.json` written
pre-training confirmed the working-tree edits were live for those runs.
Diffing their two `build_manifest.json`s showed the only meaningful
non-noise delta was `CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD` (65536 →
10000) and `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD` (removed) — matching
the `environment.sh` edit the user had made between the two runs. Both runs
still crashed at the same point (after the first `_sync_best_artifact`),
ending with a GPU segfault → rank 0 signal 6 → SIGTERM cascade in the .o
files. Conclusion: the CCL knob change wasn't load-bearing for the
underlying crash. The build_manifest pipeline itself was the win — for the
first time a side-by-side environment diff was possible across failed runs.

## Lingering items / deferred

- **Sampling-side `device_id=` fix** for the c10d barrier warning in
  `biom3.core.distributed.init_distributed_if_launched`. User explicitly
  deferred. Tracked in memory `todo_barrier_device_id_warning.md`.
- **Heartbeat callback** (~25 lines, every-N-batches stdout heartbeat as a
  tqdm replacement). Proposed during the progress-bar discussion but user
  opted for the simpler tty-auto-detect alone. Per-epoch metric logging from
  the existing `MetricsHistoryCallback` plus `tail -f
  artifacts_dir/metrics_history.train.jsonl` is the recommended within-epoch
  progress signal under PBS.
- **Underlying GPU-segfault crash mode on resume into new run_id**. Pre-existing
  before this session and still unresolved — the manifest/checkpoint/log
  cleanups here let the user diagnose it, but don't fix it. Both 152703 and
  155153 still died with `Segmentation fault from GPU` after the first
  artifact sync. Hypothesis tested (CCL IPC handle cache threshold) did not
  prevent the crash.
- **Test coverage** for the resume-into-new-run-id manifest + checkpoint_dir
  fix was discussed but user opted out (no test coverage requested). The
  changes ship without a regression test.

## Commit

Single commit on `addison-dev`, scope:
`src/biom3/Stage3/run_PL_training.py` + this session note. The two new
aurora jobscripts and all other working-tree changes (`environment.sh`,
`configs/stage3_training/finetune_v1.json`, `scripts/launchers/aurora_*.sh`)
were left untracked / unstaged.
