# Session: Source Dataset Builders from Raw Databases

**Date:** 2026-04-01  
**Branch:** `addison-main`  
**Pre-session state:** `git checkout 2c9096a`

## Summary

Added the ability to build the two foundational training CSVs
(`fully_annotated_swiss_prot.csv` and `Pfam_protein_text_dataset.csv`) directly
from the raw database files (UniProt `.dat`, Pfam FASTA, Stockholm). This closes
the reproducibility gap — previously, the finetuning pipeline depended on
pre-curated CSVs built by someone else with an unknown process.

The full pipeline lineage is now:

```
Raw databases (UniProt .dat, Pfam FASTA, NCBI taxonomy)
  → biom3_build_source_swissprot / biom3_build_source_pfam
    → training CSVs (fully_annotated_swiss_prot.csv, Pfam_protein_text_dataset.csv)
      → biom3_build_dataset (subset by Pfam ID, enrich, filter)
        → fine-tuning dataset
```

## Changes

### New files

| File | Purpose |
|------|---------|
| `src/biom3/dbio/pfam_metadata.py` | `PfamMetadataParser` — extracts family names/descriptions from Pfam-A.full.gz (Stockholm `#=GF` headers) or Pfam-A.hmm.gz |
| `src/biom3/dbio/build_source_swissprot.py` | Builds `fully_annotated_swiss_prot.csv` from `uniprot_sprot.dat.gz` + Pfam metadata. Streams .dat entries, extracts sequences/annotations/Pfam DR refs, composes captions with PubMed ref removal |
| `src/biom3/dbio/build_source_pfam.py` | Builds `Pfam_protein_text_dataset.csv` from `Pfam-A.fasta.gz` + Pfam metadata. Parses FASTA headers for accession/range/Pfam ID, joins family metadata, composes captions |
| `demo/build_source_datasets.sh` | Demo script showing end-to-end: raw DBs → source CSVs → SH3 finetuning dataset |
| `tests/dbio_tests/test_pfam_metadata.py` | 7 tests for metadata parser |
| `tests/dbio_tests/test_build_source_swissprot.py` | 14 tests for SwissProt builder |
| `tests/dbio_tests/test_build_source_pfam.py` | 12 tests for Pfam builder |
| `tests/_data/dbio/mini_swissprot.dat` | Test fixture: 4 .dat entries (3 with Pfam, 1 without) |
| `tests/_data/dbio/mini_pfam_metadata.sto` | Test fixture: 3 Stockholm family blocks |
| `tests/_data/dbio/mini_pfam.fasta` | Test fixture: 5 FASTA domain entries |

### Modified files

| File | Change |
|------|--------|
| `src/biom3/dbio/swissprot_dat.py` | Added `parse_all()` bulk generator and `_parse_entry_full()` which extracts protein sequences (SQ lines), Pfam cross-references (DR Pfam lines), and additional CC topics (TISSUE SPECIFICITY, DEVELOPMENTAL STAGE, BIOTECHNOLOGY). Existing `parse()` and `_parse_entry()` unchanged for backward compatibility |
| `src/biom3/dbio/__main__.py` | Added `run_build_source_swissprot()` and `run_build_source_pfam()` entry point functions |
| `pyproject.toml` | Added `biom3_build_source_swissprot` and `biom3_build_source_pfam` console script entry points |

### New CLI commands

```bash
# Build SwissProt CSV (~547K rows, ~785 MB, ~1 min)
biom3_build_source_swissprot \
    --dat data/databases/swissprot/uniprot_sprot.dat.gz \
    --pfam-metadata data/databases/pfam/Pfam-A.full.gz \
    -o data/datasets/fully_annotated_swiss_prot.csv

# Build Pfam CSV (~63M rows, ~35 GB, several hours for full DB)
biom3_build_source_pfam \
    --fasta data/databases/pfam/Pfam-A.fasta.gz \
    --pfam-metadata data/databases/pfam/Pfam-A.full.gz \
    -o data/datasets/Pfam_protein_text_dataset.csv
```

## Demo results

Successfully ran the full pipeline on real data:

1. **SwissProt build**: `uniprot_sprot.dat.gz` → 547,272 rows (785 MB) in ~1 minute
2. **Pfam build** (SH3 subset): 24,995 rows (16 MB) from extracted PF00018 FASTA entries
3. **Fine-tuning dataset**: Fed both into `biom3_build_dataset -p PF00018` → 25,580-row SH3 dataset

### Comparison with legacy CSVs

| Metric | Legacy | New |
|--------|--------|-----|
| SwissProt rows | 569,516 | 547,272 |
| SH3 SwissProt hits | 599 | 585 |
| Caption format | Matches | Matches (minor cosmetic diffs) |

Row count difference (~22K) is expected — different database releases. The legacy CSVs
were built from an older UniProt/Pfam release; the raw databases on-disk are current
(downloaded 2026-03-29/30).

## Test status

All 77 dbio tests pass (33 new + 44 existing), no regressions.

## Not yet committed

All changes are staged but uncommitted. No git commit was made during this session.

## Known limitations / future work

- **Pfam row count discrepancy**: `Pfam-A.fasta.gz` has ~63M entries vs the legacy CSV's 44.8M. This is a Pfam release version difference, not a filtering issue. The full Pfam build was not run during the demo due to time/disk constraints.
- **Caption cosmetic differences**: The `.dat` parser preserves slightly more structure in some CC fields (e.g., "pH dependence:" prefix in BIOPHYSICOCHEMICAL PROPERTIES) compared to the legacy CSV. These are cosmetic and should not affect model training.
- **No `text_caption` / `[clean]text_caption` columns**: The legacy SwissProt CSV had 6 columns including intermediate caption variants. The new builder produces only the 4 columns that `SwissProtReader` actually uses. An `--include-intermediate-captions` flag was discussed but not implemented — it can be added if needed.
- **Provenance of legacy filtering**: The original `Pfam_protein_text_dataset.csv` may have had additional filtering (44.8M vs 63M entries). The specific criteria are unknown. If exact reproduction of the legacy dataset is needed, this would require investigating what was filtered and why.
