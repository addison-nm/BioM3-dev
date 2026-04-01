# 2026-04-01: Dual Logging + Manifest for Entrypoints & Taxonomy Filter Fix

**Starting point:** `git checkout 80b4453`

## Summary

Two features implemented in this session:

1. **Dual logging and `build_manifest.json` for all inference entrypoints** — following the pattern already established in `biom3.dbio.build_dataset`, every entrypoint now logs to both the console and a `run.log` file, and writes a `build_manifest.json` with version info, args, timing, and entrypoint-specific outputs.

2. **Bug fix: taxonomy filter fallback to `annot_lineage`** — the `--taxonomy-filter` option in `biom3_build_dataset` was producing 0 rows for Bacteria because the NCBI `prot.accession2taxid` index only covers ~3.8% of UniProt accessions. The filter now falls back to the `annot_lineage` column populated by UniProt enrichment.

## Changes

### New file: `src/biom3/core/run_utils.py`

Shared utilities extracted from `dbio/build_dataset.py` and generalized:
- `get_biom3_version()` / `get_git_hash()` — version and commit info
- `get_biom3_build()` — PEP 440-style build string (e.g. `0.0.1+g80b4453.dirty`)
- `get_git_dirty()` / `get_git_branch()` / `get_git_remote()` — full git context (remote URL has credentials stripped)
- `get_file_metadata(filepath)` — size, mtime, realpath (resolves symlinks) for any file
- `setup_file_logging(outdir, logger_prefix, log_filename)` — attaches a FileHandler to all matching loggers (rank-0 only for distributed safety)
- `teardown_file_logging(logger_prefix, file_handler)` — removes handler and closes
- `write_manifest(args, outdir, start_time, elapsed, outputs, resolved_paths, database_versions)` — writes `build_manifest.json`

### Modified entrypoints (dual logging + manifest)

Each `main()` function now:
1. Creates the output directory and sets up dual logging at the start
2. Prints a banner with biom3 version, git hash, and full command
3. Writes `build_manifest.json` after completion with full reproducibility info:
   - `biom3_version`, `biom3_build` (e.g. `0.0.1+g80b4453.dirty`), `git_hash`, `git_dirty`, `git_branch`, `git_remote`
   - Timing, args, resolved paths, and entrypoint-specific outputs
   - For dbio: `database_versions` with UniProt/Pfam release versions, file metadata, and provenance TSV path
4. Tears down file logging on exit

Each `main()` also accepts `_setup_logging=True` kwarg so the embedding pipeline can suppress duplicate logging when calling sub-stages in-process.

| File | Manifest `outputs` |
|------|-------------------|
| `Stage1/run_PenCL_inference.py` | `num_samples`, `embedding_dim`, `output_file` |
| `Stage2/run_Facilitator_sample.py` | `num_samples`, `embedding_dim`, MSE/MMD metrics |
| `Stage3/run_ProteoScribe_sample.py` | `num_prompts`, `num_replicas`, `seed`, `total_sequences` |
| `data_prep/compile_stage2_data_to_hdf5.py` | `num_samples`, `dataset_key` |
| `pipeline/embedding_pipeline.py` | `pencl_output`, `facilitator_output`, `hdf5_output` |
| `dbio/build_dataset.py` | Refactored to use `core.run_utils` (same behavior) |

tqdm progress bars (Stage 1) and Lightning progress bars (training) write to stderr, not through the logging module, so they are automatically excluded from log files.

The training script (`run_PL_training.py`) was intentionally deferred — Lightning manages its own logging infrastructure.

### Taxonomy filter bug fix: `dbio/build_dataset.py`

**Problem:** `_apply_taxonomy_filters` relied solely on NCBI `prot.accession2taxid` for accession-to-taxid mapping. This file covers only ~3.8% of UniProt accessions in the Pfam dataset (600/15,740 for SH3 domain PF00018). With `--taxonomy-filter superkingdom=Bacteria`, this produced 0 rows despite 4,631 Bacteria accessions being available via UniProt enrichment.

**Fix:** Two-path approach in `_apply_taxonomy_filters`:
1. NCBI index (structured rank matching) for accessions found in `prot.accession2taxid`
2. `annot_lineage` column fallback for unmapped accessions — parses the enrichment-populated lineage string and checks if filter values appear in it

The log now reports: `Taxonomy filter matched: N via NCBI index, M via annot_lineage`

After the fix, the bacteria demo produces 18 rows (down from 17,165 total).

### New test file: `tests/core_tests/test_run_utils.py`

7 unit tests covering:
- `get_biom3_version` and `get_git_hash`
- `setup_file_logging` + `teardown_file_logging` (message capture and cleanup)
- `write_manifest` structure, custom outputs, and optional key omission

### Modified test: `tests/dbio_tests/test_build_dataset.py`

Updated `test_build_log_and_manifest` to match new manifest structure where `row_counts` is nested under `outputs`.

## Not implemented / deferred

- **Training script logging** (`run_PL_training.py`): Deferred because PyTorch Lightning manages its own logging (TensorBoard, W&B) and progress bars. A future follow-up could add a custom Lightning Callback for file logging and manifest writing.
- **`annot_lineage` fallback is not exact rank matching**: The NCBI path uses structured `{rank: value}` lookups. The fallback checks if the filter value appears anywhere in the flat lineage list. This works for superkingdom/phylum/etc. but could theoretically false-match if a taxon name appears at multiple ranks. In practice this is not an issue for real taxonomy data.

## Verification

All 55 tests pass:
```
pytest tests/core_tests/ tests/dbio_tests/ tests/test_imports.py  # 55 passed
```

Bacteria demo re-run confirmed: 17,165 -> 18 rows with `superkingdom=Bacteria` filter.
