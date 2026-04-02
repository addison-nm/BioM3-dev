# Fix sync_weights.sh directory-level symlink bug

**Date:** 2026-04-02

## Problem

`scripts/sync_weights.sh` had a bug where, if a component subdirectory (e.g.
`weights/PenCL/`) did not already exist in the target, the script symlinked the
**entire directory** to the shared source rather than creating a local directory
and symlinking individual files within it.

This meant:
- Writes to `weights/<component>/` would go into the shared directory (or fail
  if the shared path was read-only).
- Local-only files could not coexist alongside shared symlinks in that directory.

The correct version already existed in `BioM3-workflow-demo/scripts/sync_weights.sh`.

## Fix

Replaced the directory-level symlink + `continue` (lines 91-99) with `mkdir -p`
and a fall-through to the per-file symlink loop, matching the workflow-demo
version. Also added an empty-directory guard (`[ -e "$entry" ] || continue`).

## Files changed

- `scripts/sync_weights.sh` — core fix
- `docs/setup_shared_weights.md` — clarified that component directories are real
  directories, not symlinks
