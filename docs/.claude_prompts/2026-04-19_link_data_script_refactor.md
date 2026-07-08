# Prompt: extract scripts/link_data.sh from sync_databases.sh

Factor the symlinking logic of `scripts/sync_databases.sh` into a new canonical
script `scripts/link_data.sh`. Keep `sync_databases.sh` and add a new
`sync_datasets.sh`, both as thin wrappers that delegate to `link_data.sh`.
**Do not modify `sync_weights.sh`** — its behavior is intentionally different.

## Why

- The name "sync" is misleading — these scripts only create symlinks; there
  is no rsync-style two-way sync. "link" is accurate.
- `sync_databases.sh` already works unchanged against `data/datasets/` (same
  pattern of "symlink per subdirectory entry plus top-level files"), but
  there is no `sync_datasets.sh`, so the repo-wide CLAUDE.md guidance
  ("repopulate `data/databases/` or `data/datasets/` via the appropriate
  `scripts/sync_*.sh`") currently has a gap for datasets.
- Adding a general script removes duplication if the pattern is ever needed
  for a third data directory.

## Read these first

- `scripts/sync_databases.sh` — source of the generalized logic.
- `scripts/sync_weights.sh` — *for comparison only*; do not modify.
- `CLAUDE.md` (repo root) — the worktree section that points users at
  `scripts/sync_*.sh` for repopulating `data/databases/` and `data/datasets/`.
- `../CLAUDE.md` (ecosystem root, `BioM3-dev-space/CLAUDE.md`) — cross-repo
  notes about sync scripts.

## What to build

### 1. `scripts/link_data.sh` (new, canonical)

Extract the exact behavior of `sync_databases.sh` into this file. CLI
interface stays the same:

```
link_data.sh <source_dir> <target_dir> [--dry-run]
```

**Key behavior to preserve** (this is where `sync_databases.sh` differs from
`sync_weights.sh`):

- Iterate subdirectories of `SRC_DIR` and symlink their entries into
  matching subdirectories of `TGT_DIR`.
- **Also iterate top-level files in `SRC_DIR`** (e.g. `provenance.tsv` in
  `data/databases/`) and symlink them directly into `TGT_DIR`. The weights
  script does not do this.
- Skip `.git*` and `README*` entries.
- When a target entry already exists, compare md5 hashes (recursively for
  directories) and print MATCH/MISMATCH rather than overwriting.
- `set -euo pipefail`, `realpath` on inputs, `--dry-run` support.

### 2. `scripts/sync_databases.sh` (rewritten as thin wrapper)

Replace the body with a delegation to `link_data.sh`, forwarding all args.
Keep the file so existing invocations in docs and scripts don't break.
One-line `exec "$(dirname "$0")/link_data.sh" "$@"` is enough; keep the
header comment pointing users at `link_data.sh` as the canonical script.

### 3. `scripts/sync_datasets.sh` (new, thin wrapper)

Same pattern as the rewritten `sync_databases.sh` — delegates to
`link_data.sh`. This closes the CLAUDE.md gap for `data/datasets/`.

### 4. Leave `sync_weights.sh` alone

Its loop is deliberately simpler (no top-level file handling), its summary
adds a torch-specific tip about mismatched archives, and it targets
`weights/` not `data/`. Folding it into `link_data.sh` would either lose
the torch tip or add a mode flag for something only weights need. Not worth
it. Flag any discovered duplication as a follow-up, don't merge them.

## References to search for and update

Search the **whole repo** (not just `scripts/`) for references to the old
names, and update docs where the wrapper names survive but the canonical
pointer should become `link_data.sh`:

- `grep -RIn 'sync_databases\.sh\|sync_weights\.sh\|sync_datasets\.sh' -- ':!**/.git/**'`
- `CLAUDE.md` at repo root (mentions `scripts/sync_*.sh` in the worktree section).
- `../CLAUDE.md` (`BioM3-dev-space/CLAUDE.md`) — ecosystem-level guidance.
- Any `docs/**/*.md` — especially setup and machine-specific guides (Polaris/Aurora/Spark).
- `README.md` at root and in `src/biom3/` subpackages if present.
- `weights/README.md`, `docs/setup_databases.md` if they exist.
- `demos/` scripts and `jobs/` HPC templates.
- `docs/.claude_sessions/*.md` — **do not retroactively edit session notes**;
  treat them as historical records. If a note references the old name, that
  was correct at the time.
- Sibling repo `CLAUDE.md` files **only** if the path is `../BioM3-*/CLAUDE.md`
  reachable from this repo's working tree. Do not walk outside the working tree.

For each live reference, decide:

- **Update** when the doc is prescriptive ("run `scripts/sync_databases.sh` to
  set up") — point at whichever wrapper is correct for the context
  (`sync_databases.sh` for databases, `sync_datasets.sh` for datasets), and
  mention `link_data.sh` as the underlying script where relevant.
- **Leave** when the doc is descriptive of the past (session notes, release
  notes, commit messages).

## Conventions

- Match the existing shell style in `sync_databases.sh`: `set -euo pipefail`,
  4-space indent, `[ ... ]` test syntax, no bashisms beyond what's already
  there.
- Wrappers should be executable (`chmod +x`).
- Commits follow Conventional Commits. Suggested split:
  1. `refactor(scripts): extract link_data.sh from sync_databases.sh`
  2. `feat(scripts): add sync_datasets.sh wrapper around link_data.sh`
  3. `docs: point docs at link_data.sh / sync_datasets.sh where appropriate`
  Include the `Co-Authored-By` trailer for the model you are.
- Do not commit without the user's explicit ask.

## Verification

Dry-run all three scripts and confirm output parity with the pre-refactor
`sync_databases.sh`:

```
# baseline (pre-refactor output — capture before touching anything)
git stash   # or checkout the original version
./scripts/sync_databases.sh /data/data-share/BioM3-data-share/databases data/databases --dry-run > /tmp/baseline_db.txt

# after refactor
./scripts/link_data.sh       /data/data-share/BioM3-data-share/databases data/databases --dry-run > /tmp/new_link.txt
./scripts/sync_databases.sh  /data/data-share/BioM3-data-share/databases data/databases --dry-run > /tmp/new_db_wrapper.txt
./scripts/sync_datasets.sh   /data/data-share/BioM3-data-share/data/datasets data/datasets --dry-run > /tmp/new_ds_wrapper.txt

# all three must be identical to the baseline in structure (paths will differ for datasets)
diff /tmp/baseline_db.txt /tmp/new_link.txt
diff /tmp/baseline_db.txt /tmp/new_db_wrapper.txt
```

The datasets output will list different entries (because SRC and TGT are
different), but the output *format* (MKDIR / LINK / MATCH / MISMATCH /
Summary) should be identical.

Do not *apply* any links during verification — the repo already has a
populated `data/databases/` and (where present) `data/datasets/`, and a
real run would append additional symlinks or trip MATCH/MISMATCH checks
against already-synced content. Dry-run only.

## Things to watch out for

- Some compute clusters have different data-share roots (see the ecosystem
  CLAUDE.md — Spark vs Polaris vs Aurora). The scripts themselves are
  machine-agnostic (you pass paths in); doc updates should preserve any
  machine-specific examples already present rather than hardcoding one root.
- The repo has several active git worktrees. `link_data.sh` lives in
  `scripts/` which is part of the tracked tree, so the new files will land
  on the branch you commit to. Verify you're on the intended branch (likely
  `addison-dev` or a fresh worktree off it) before starting.
- `data/databases/` and `data/datasets/` are gitignored except for
  `.gitkeep`. None of your changes should add tracked files under those
  directories.

## Session end

Write a session note to `docs/.claude_sessions/<YYYY-MM-DD>_link_data_script_refactor.md`:

- One-paragraph summary.
- Commits list.
- File-by-file changes (new scripts + docs touched).
- Any docs that were intentionally left alone and why (e.g. session notes as historical records).
- Any follow-ups discovered (e.g. if `sync_weights.sh` has a small overlap worth noting, log it but do not act).
