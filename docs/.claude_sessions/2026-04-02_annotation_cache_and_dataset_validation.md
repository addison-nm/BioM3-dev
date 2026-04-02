# 2026-04-02: Annotation Cache and Dataset Validation

## Summary

Validated the `biom3.dbio` dataset building pipeline by reproducing the legacy
SH3 (`PF00018`) and CM (`PF01817`) finetuning datasets, identified and fixed
enrichment coverage issues, optimized the `.dat` parser, and implemented a
Parquet annotation cache for fast repeated builds.

## Validation results

Reproduced both datasets with `biom3_build_dataset --enrich-pfam`:

| Metric | SH3 | CM |
|--------|-----|-----|
| Row count match | 17,165 = 17,165 | 7,189 = 7,189 |
| Accession overlap | 100% of target | 100% of target |
| SwissProt caption exact match | 100% | 100% |
| Pfam enrichment (PROTEIN NAME) | 94.9% | — |
| Pfam enrichment (LINEAGE) | 94.9% | — |

The ~5% enrichment gap is expected: some legacy Pfam accessions have been
retired in the newer UniProt release. Pfam caption text differs due to updated
annotation wording in the newer database, not due to pipeline issues.

## Key finding: TrEMBL required for Pfam enrichment

Initial builds with only `uniprot_sprot.dat.gz` achieved <0.01% Pfam
enrichment because Pfam accessions are overwhelmingly TrEMBL (unreviewed)
entries. Adding `uniprot_trembl.dat.gz` (161 GB) to the `--uniprot-dat` path
resolved coverage to ~95%.

Similarly, the NCBI `prot.accession2taxid` SQLite index contains very few
UniProt accessions (~6,250 A0A-style entries out of 1.55B total). Taxonomy
lineage for Pfam rows comes from the `.dat` file's OC lines, not from NCBI.

## Parser performance optimization

Updated `SwissProtDatParser` in `swissprot_dat.py`:
- Added `_open_gz()` helper that uses `pigz -dc` (parallel gzip) when
  available, falling back to Python's `gzip.open`
- Switched both `parse()` and `parse_all()` to binary-mode reads — only
  matched entries are UTF-8 decoded, skipping decode for 99.99% of lines
- Accession matching uses bytes comparison to avoid decode overhead

`pigz` installed system-wide via apt-get on DGX Spark.

## Annotation Parquet cache (new feature)

Created `src/biom3/dbio/build_annotation_cache.py` to solve the TrEMBL
parsing bottleneck for repeated builds:

**Builder**: `biom3_build_annotation_cache --dat <file> -o <parquet>`
- Uses `SwissProtDatParser.parse_all()` to iterate all entries
- Writes sorted chunks via PyArrow `ParquetWriter` for efficient predicate
  pushdown at read time
- Skips entries with zero annotations by default
- Schema: `primary_Accession` + 19 annotation columns

**Reader**: `load_annotation_cache(cache_paths, accessions)`
- Uses `pq.read_table(filters=...)` for accession-filtered reads
- Returns `dict[accession, dict[annot_col, value]]` — same sparse format as
  `SwissProtDatParser.parse()`

**Integration**: New `--annotation-cache` flag in `biom3_build_dataset`:
- Priority: cache > `.dat` > API
- Multiple cache files supported (e.g. Swiss-Prot + TrEMBL caches)
- Remaining accessions not in cache fall through to `--uniprot-dat` or API

## Files changed

| File | Change |
|------|--------|
| `src/biom3/dbio/swissprot_dat.py` | pigz support, binary-mode parsing |
| `src/biom3/dbio/build_annotation_cache.py` | **New** — builder, reader, CLI |
| `src/biom3/dbio/build_dataset.py` | `--annotation-cache` flag and cache-first enrichment |
| `src/biom3/dbio/__main__.py` | `run_build_annotation_cache()` entrypoint |
| `pyproject.toml` | `biom3_build_annotation_cache` script registration |
| `tests/dbio_tests/test_build_annotation_cache.py` | **New** — 10 tests |
| `docs/building_datasets_with_dbio.md` | Annotation cache usage section |
| `docs/setup_databases.md` | Annotation cache mention |

## Next steps

- Build the TrEMBL annotation Parquet cache (one-time, ~1-2 hours with pigz)
- Re-run SH3 and CM builds with `--annotation-cache` to verify instant enrichment
- Compare CM v2 results once build completes
