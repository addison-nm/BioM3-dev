# Shared weights path rename: models → weights

**Date:** 2026-03-30

## Summary

Renamed the shared data-share path pattern from `BioM3-data-share/data/models/[component]` to `BioM3-data-share/data/weights/[component]`, to mirror the local `weights/` directory structure.

## Files changed

- `scripts/sync_weights.sh` — updated comment examples
- `docs/setup_shared_weights.md` — updated machine path table, directory tree label, and DGX Spark examples

## Notes

- The sync script itself is generic (takes source/target as CLI args), so no functional code changed.
- Only 1 of 7 existing symlinks in `weights/` actually points to the BioM3-data-share (`Facilitator_MMD15.ckpt`). The other 6 point to `/data/biom3_data/weights/`.
- ~1,571 local files exist in `weights/` (DeepSpeed sharded checkpoints, training checkpoints, etc.).
