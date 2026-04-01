# Session: Add biom3.dbio subpackage and database sync infrastructure

**Date:** 2026-03-31

## Context

The project needed a formalized interface to local protein reference databases (Pfam, Swiss-Prot,
NCBI Taxonomy, SMART, ExPASy) for constructing fine-tuning datasets. A working prototype existed
at `../protein-language-data-exploration/` demonstrating the core workflow: subset Swiss-Prot and
Pfam CSVs by Pfam ID, optionally enrich text captions via the UniProt REST API. This session
ported and extended that prototype into a proper subpackage, adding local NCBI taxonomy support,
a local Swiss-Prot `.dat` parser, a symlink-based database sync mechanism, and reproducibility
logging.

## Changes

### New subpackage: `src/biom3/dbio/` (11 modules)

1. **`config.py`** — Database path resolution. Priority: `BIOM3_DATABASES_ROOT` env var >
   `configs/dbio_config.json` > CLI flags. Uses `core.helpers.load_json_config()`.

2. **`base.py`** — `DatabaseReader` ABC with abstract `query_by_pfam(pfam_ids)` method
   and `name` property. Provides extensibility for future SMART/ExPASy readers.

3. **`swissprot.py`** — `SwissProtReader(DatabaseReader)`. Loads full CSV (~570K rows) into
   memory; filters via `str.contains(regex)` on `pfam_label` column (which stores stringified
   Python lists). Lazy-loads on first query.

4. **`pfam.py`** — `PfamReader(DatabaseReader)`. Reads ~44.8M-row CSV in configurable chunks
   (default 500K). Filters via `isin()` exact match. Renames columns: `id` -> `primary_Accession`,
   `sequence` -> `protein_sequence`. Includes tqdm progress bar and
   `dtype={"family_description": str}` to suppress mixed-type warnings.

5. **`taxonomy.py`** — Two classes:
   - `TaxonomyTree`: Loads `rankedlineage.dmp` (~2.7M rows) into memory for O(1) lineage
     lookups. Methods: `get_lineage()`, `get_lineage_string()`, `filter_by_rank()`.
   - `AccessionTaxidMapper`: Maps protein accessions to NCBI tax_ids from
     `prot.accession2taxid.gz` (1.55B rows). Two strategies: streaming (no setup, ~10-15 min)
     and SQLite index (one-time build, then instant). `lookup()` auto-detects which to use.

6. **`uniprot_client.py`** — Ported from prototype. `UniProtCache` (disk-based JSON cache) +
   `UniProtClient` (batch fetch with retry/rate-limiting, batch size 25, 5 retries).

7. **`swissprot_dat.py`** — Local parser for `uniprot_sprot.dat.gz` (or `uniprot_trembl.dat.gz`).
   `SwissProtDatParser` streams the flat file line by line, extracting DE (protein name),
   CC (function, catalytic activity, etc.), OC (lineage), DR GO (gene ontology) for requested
   accessions only. Replaces API calls when local `.dat` files are available.

8. **`enrich.py`** — Two-step enrichment workflow:
   - `enrich_dataframe()`: populates individual `annot_*` columns from local `.dat` data,
     UniProt API data, and/or NCBI taxonomy. Does NOT modify `[final]text_caption`.
   - `compose_caption()`: assembles `[final]text_caption` from annotation columns using
     ALL-CAPS field labels. Accepts custom `fields` list for control over field order.
   - Applied to Pfam rows only; SwissProt rows keep their source captions.

9. **`build_dataset.py`** — Pipeline orchestrator with full CLI. Extracts from SwissProt + Pfam,
   optionally enriches, composes captions, filters by taxonomy, saves outputs. Produces:
   - `dataset.csv` — final output (standard columns, quoted)
   - `dataset_annotations.csv` — intermediate with all `annot_*` columns
   - `build.log` — full log output (dual console + file logging)
   - `build_manifest.json` — reproducibility manifest (version, git hash, command, args,
     resolved paths, row counts, python version, timestamps)
   - `pfam_ids.csv` — Pfam IDs used

10. **`__main__.py`** — Entry points: `run_build_dataset()`, `run_build_taxid_index()`.

11. **`__init__.py`** — Empty (follows repo convention).

### Database sync infrastructure

- **`scripts/sync_databases.sh`** — Mirrors `sync_weights.sh` pattern. Creates per-file symlinks
  from shared databases into `data/databases/`. Handles empty directories, md5 verification,
  `--dry-run` mode. Also links top-level files (e.g., `provenance.tsv`).

- **`configs/dbio_config.json`** — Default config pointing `databases_root` to `data/databases/`
  and `training_data_root` to the shared training CSV location.

### Documentation

- **`docs/setup_databases.md`** — Full setup guide: shared paths per machine (Spark/Polaris/Aurora),
  database inventory with sizes, path override options, enrichment strategies (API vs local `.dat`),
  TrEMBL download guidance, SQLite index instructions.

- **`README.md`** — Added "Reference databases" section with sync instructions and examples.
  Added "Dataset construction" section with full argument table and usage examples.

- **`CLAUDE.md`** — Added `dbio/` to repo layout, `data/databases/` directory, and entry points.

### Demo script

- **`demo/build_sh3_dataset.py`** — Demonstrates the full workflow: query SwissProt + Pfam,
  enrich, compose captions, save. Supports `--enrich-pfam`, `--uniprot-dat`, API fallback.

### Modified existing files

- **`pyproject.toml`** — Added `biom3_build_dataset` and `biom3_build_taxid_index` entry points.
- **`.gitignore`** — Added `.uniprot_cache/`.
- **`tests/conftest.py`** — Added `--database_files` marker and skip logic.
- **`tests/test_imports.py`** — Added `test_dbio_imports()`.

### Test suite (42 new tests, all passing)

- `tests/dbio_tests/test_swissprot.py` — 5 tests
- `tests/dbio_tests/test_pfam.py` — 7 tests
- `tests/dbio_tests/test_taxonomy.py` — 14 tests
- `tests/dbio_tests/test_enrich.py` — 9 tests (extract_annotations, enrich_dataframe, compose_caption)
- `tests/dbio_tests/test_build_dataset.py` — 8 tests (including log/manifest verification)
- Mini fixtures in `tests/_data/dbio/`

## Key design decisions

- **Package name `dbio`** — short for "database I/O", parallel to `core/io.py`.
- **API-first enrichment** — `--enrich-pfam` uses the UniProt REST API by default because Pfam
  accessions are ~100% TrEMBL (unreviewed), not in `uniprot_sprot.dat.gz` (reviewed only).
  `--uniprot-dat` accepts one or more local `.dat.gz` files for offline enrichment (Swiss-Prot
  and/or TrEMBL when downloaded).
- **Two-step enrichment** — `enrich_dataframe()` populates `annot_*` columns, `compose_caption()`
  formats them. Keeps raw annotations inspectable and caption format customizable.
- **CSV quoting** — `csv.QUOTE_NONNUMERIC` ensures commas in captions don't break CSV parsing.
- **Reproducibility** — `build_manifest.json` captures biom3 version, git hash, full command,
  resolved paths, and row counts. `build.log` captures full logging output.
- **`data/databases/` as local target** — Mirrors the `weights/` pattern. Symlinked from shared
  storage. Already gitignored via `data/*`.

## Verification

- All 46 tests pass (4 existing imports + 42 new dbio).
- `sync_databases.sh` successfully linked 29 files from Spark's shared directory.
