# 2026-05-12 — Stage 3 default tuning + environment.sh detection refactor

## Context

Continuation of the same day (after `b8052b2`). Three orthogonal changes:

1. **Metrics-history cadence.** Default `--metrics_history_every_n_steps`
   went from `1` (one record per step) to `10`. With 1024 batches/epoch and
   long runs, per-step JSONL emissions were both noisy and large; 10 keeps
   the same "tail -f progress" UX with one-tenth the I/O. The complementary
   `--metrics_history_every_n_epochs 1` knob already provides the per-epoch
   floor when desired.
2. **Finetune-default sentinel.** The argparse defaults for
   `--finetune_last_n_blocks` / `--finetune_last_n_layers` remain `-2` (the
   sentinel meaning "unspecified"). The *coercion* of `-2` inside `main()`
   changed from `1` (last block / last layer) to `-1` (all blocks / all
   layers). Concretely: a bare `--finetune True` now finetunes the entire
   model. Explicit values in CLI or JSON config are unaffected.
3. **`environment.sh` detection refactor.** Replaced the embedded
   hostname-pattern detection with filesystem-fingerprint detection up front,
   then a single per-machine `if`/`elif` block keyed off `$BIOM3_MACHINE`.

State before this session began:

```bash
git checkout b8052b2  # feat(stage3): finetune-resume robustness + log/disk hygiene
```

## Why filesystem fingerprints for machine detection

User asked how confident we are about the `x3*` / `x4*` hostname prefixes
that the prior `environment.sh` used to discriminate Polaris vs. Aurora.
Audit answer: those were **empirical, not contractual**. The originating
commit (`c810faa`, 2026-03-26) introduced them without a recorded rationale.
We had direct in-repo observation of Aurora compute as `x4*` (multiple
`build_manifest.json` hostnames from today's fluc runs: `x4613c4s1b0n0`,
`x4616c3s0b0n0`, plus `x4309c4s0b0n0` from earlier sessions). We had **no**
direct observation of Polaris `x3*` in this repo — the prefix was inferred
from Polaris being Cray Shasta (same `xN-cN-sN-bN-nN` naming family) and
ALCF's deployment convention.

The fix: ALCF documents (and the project itself uses, per
`docs/biom3_ecosystem.md`, `docs/setup/setup_shared_weights.md`,
`docs/setup/setup_databases.md`) canonical project mounts that are 1:1 with
the cluster:

| Mount    | Cluster |
|----------|---------|
| `/flare` | Aurora  |
| `/grand` | Polaris |

These are *project-canonical* — the user's data layout, weight paths, and
database paths are all tied to them — so they're a stronger contract than
hostname prefixes that could change on a hardware refresh.

Spark stays on hostname because it's a single-node device with no
distinguishing shared mount.

## `environment.sh` structural change

Old structure: a single big `if`/`elif` chain that combined detection (by
hostname) **and** per-machine env-var setting in each branch — meaning the
two concerns were entangled and adding a new variable required knowing the
right branch.

New structure (two separate blocks, in order):

```bash
# --- Common (all machines) ---
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# --- Machine detection ---
if   [[ -d /flare ]]; then BIOM3_MACHINE=aurora
elif [[ -d /grand ]]; then BIOM3_MACHINE=polaris
elif [[ "$(hostname)" == spark* ]]; then BIOM3_MACHINE=spark
else BIOM3_MACHINE=unknown
fi
export BIOM3_MACHINE
echo "[environment.sh] Detected machine: $BIOM3_MACHINE"

# --- Machine-specific settings ---
if   [[ "$BIOM3_MACHINE" == polaris ]]; then : # no exports
elif [[ "$BIOM3_MACHINE" == aurora  ]]; then
    # NUMEXPR_MAX_THREADS, CCL_*, TMPDIR, ...
elif [[ "$BIOM3_MACHINE" == spark   ]]; then : # no exports
else
    echo "[environment.sh] Unknown machine: $(hostname) (using common settings only)"
fi
```

Per-machine settings now key purely off `$BIOM3_MACHINE`. The single per-run
"Detected machine" echo replaces the three branch-specific echoes.

Aurora export changes carried over from the working tree (in-progress edits
the user had before this refactor):

- `ONEAPI_DEVICE_SELECTOR="level_zero:gpu"` → **commented out** (user found
  it unnecessary).
- `CCL_PROCESS_LAUNCHER=pmix`, `CCL_ATL_TRANSPORT=mpi`, `CCL_KVS_MODE=mpi`,
  `FI_MR_CACHE_MONITOR=userfaultfd` → **all commented out** (user found they
  weren't necessary).
- `CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD` → `16384` → `10000`.
- `CCL_ATL_SYNC_COLL=1` → kept, with a refined comment noting that the
  "hangs" it was avoiding may have been IDE log-file refresh artifacts.

## Code changes

[`src/biom3/Stage3/run_PL_training.py`](src/biom3/Stage3/run_PL_training.py):

- L283 — `parser.add_argument('--metrics_history_every_n_steps', default=10, ...)`
  (was `default=1`).
- L227-L235 — `--finetune_last_n_blocks` / `--finetune_last_n_layers` help
  text rewritten to document the new sentinel mapping (`-2` (default) →
  coerced to `-1` (all)).
- L1836-L1842 — `-2 → 1` coercion is now `-2 → -1` for both `_blocks` and
  `_layers`, with updated inline comments.

[`tests/stage3_tests/test_stage3_run_PL_training.py`](tests/stage3_tests/test_stage3_run_PL_training.py):

- L482-L490 — mirror coercion in `test_finetuning` updated to match the
  new sentinel mapping (`-2 → transformer_blocks` / `transformer_layers`),
  so the test's `expected_changes` list correctly enumerates all blocks/
  layers for `-2` parametrize cases.
- L432-L434 — parametrize comments updated to describe new semantics
  ("unspecified → all" instead of "unspecified → last block, last layer").

[`environment.sh`](environment.sh):

- Structural refactor described above.
- User then further refined comments and embedded the ALCF-staff
  attribution / hang-cause uncertainty in-place.

## Heads-up about existing JSON configs

The default change is at the argparse layer; existing JSON configs with
explicit values will continue to override:

- `configs/stage3_training/finetune_v1.json` and `finetune_v2.json` set
  `metrics_history_every_n_steps: 1` and explicit `finetune_last_n_blocks` /
  `finetune_last_n_layers`.
- `configs/stage3_training/pretrain_scratch_v{1,2,3}.json` and
  `pretrain_start_pfam_v2.json` all set `metrics_history_every_n_steps: 1`.

These were intentionally **not** touched — "default settings" was scoped to
the argparse defaults, not to override sweeping config values.

## What did NOT get committed

- **Four aurora jobscripts** (`jobs/aurora/job_stage3_finetune_v1_ft16_n{1,2,4,8}_fluc.pbs`)
  are on disk but untracked. User opted out of committing them twice
  (initially at the previous milestone, and again here).
- `configs/stage3_training/finetune_v1.json` (M), `scripts/launchers/aurora_multinode.sh` (M),
  `scripts/launchers/aurora_singlenode.sh` (M) — pre-existing
  working-tree edits not from this session window. Left alone.
- Root-level scratch / artifact files (`error_log*.txt`, `prompts.csv`,
  `run_sh3_multinode.sh`, `sh3_run_config.json`, the stray `.o` and `.pbs`
  files) — not the user's interest right now.

## Lingering / deferred

- **Sampling-side `device_id=` fix** for the c10d barrier warning in
  `biom3.core.distributed.init_distributed_if_launched`. Still tracked in
  user memory `todo_barrier_device_id_warning.md`. Unchanged from prior
  session note.
- **GPU-segfault crash on resume-into-new-run-id** — unresolved. The
  diagnostic surface (build manifests, run summaries) is now in place; the
  user is iterating on env knobs (current value:
  `CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD=10000`).

## Commit

Pending — to be made on `addison-dev` immediately after this note is
written. No push (per branch policy).
