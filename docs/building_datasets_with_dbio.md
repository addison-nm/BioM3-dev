# Building Datasets with `biom3.dbio`

The `biom3.dbio` package extracts protein sequences and text captions from
local database files, combines them, and optionally enriches captions with
UniProt annotations and NCBI taxonomy. The output is a CSV ready to feed into
the BioM3 embedding pipeline (Stage 1 → Stage 2 → Stage 3).

This guide walks through the process step by step, using the SH3 domain family
(`PF00018`) as a running example.

---

## Prerequisites

Install the package (editable mode):

```bash
pip install -e .
```

To run the demo showing different enrichment options:

```bash
bash demos/build_sh3_dataset.sh
```

For real datasets, you need the database files. See [setup_databases.md](setup_databases.md)
for shared paths per machine. Sync them with:

```bash
./scripts/sync_databases.sh /data/data-share/BioM3-data-share/databases data/databases
```

---

## Overview

The pipeline extracts rows from two protein databases by Pfam family ID:

| Source | File | Size | Strategy |
|--------|------|------|----------|
| **SwissProt** | `fully_annotated_swiss_prot.csv` | ~570K rows, ~1.5 GB | Loaded into memory; regex match on `pfam_label` |
| **Pfam** | `Pfam_protein_text_dataset.csv` | ~44.8M rows, ~35 GB | Chunked reading (500K rows/chunk); exact `isin()` match |

Both readers return DataFrames with the same four output columns:

```
primary_Accession, protein_sequence, [final]text_caption, pfam_label
```

The results are concatenated and saved as `dataset.csv`.

---

## Rebuilding the source CSVs from raw databases

The two source CSVs can be built from raw databases (e.g. for a fresh
UniProt/Pfam release, or when they aren't already available). Use:

| CLI | Builds | Inputs |
|-----|--------|--------|
| `biom3_build_source_swissprot` | `fully_annotated_swiss_prot.csv` | `uniprot_sprot.dat.gz` + `Pfam-A.full.gz` (for family names) |
| `biom3_build_source_pfam` | `Pfam_protein_text_dataset.csv` | `Pfam-A.fasta.gz` + `Pfam-A.full.gz` (for family metadata) |

```bash
biom3_build_source_swissprot \
    --dat data/databases/swissprot/uniprot_sprot.dat.gz \
    --pfam_metadata data/databases/pfam/Pfam-A.full.gz \
    -o data/datasets/fully_annotated_swiss_prot.csv

biom3_build_source_pfam \
    --fasta data/databases/pfam/Pfam-A.fasta.gz \
    --pfam_metadata data/databases/pfam/Pfam-A.full.gz \
    -o data/datasets/Pfam_protein_text_dataset.csv
```

Both commands accept `--chunk_size` (default 10K / 100K rows) for write buffering.

The Swiss-Prot builder has two legacy-parity flags:

- `--require_pfam` / `--no_require_pfam` (default `--no_require_pfam`): by
  default, Swiss-Prot entries without any `DR Pfam;` cross-references are
  kept and emitted with `pfam_label=['nan']`, matching the legacy
  `fully_annotated_swiss_prot.csv`. Pass `--require_pfam` to drop these
  entries instead (useful if downstream code can't handle the `['nan']`
  sentinel).
- `--keep_intermediate_captions`: emits `text_caption` (raw) and
  `[clean]text_caption` (evidence-stripped but PubMed-preserved) columns
  alongside `[final]text_caption`. Useful for auditing what the PubMed/ECO
  stripping passes remove. Without the flag, only `[final]text_caption` is
  emitted (the column set used by `SwissProtReader`).

**Caption formatting** is controlled by a `CaptionSpec` in each builder module.
Defaults reproduce the legacy CSVs: ALL-CAPS labels for Swiss-Prot (`SWISSPROT_SPEC`
in [src/biom3/dbio/build_source_swissprot.py](../src/biom3/dbio/build_source_swissprot.py)),
lowercase labels for Pfam (`PFAM_SPEC` in
[src/biom3/dbio/build_source_pfam.py](../src/biom3/dbio/build_source_pfam.py)).
See [demos/custom_caption_format.py](../demos/custom_caption_format.py) for
swapping fields or relabeling.

**Row ordering.** `biom3_build_source_swissprot` emits rows in the order they
appear in the input `.dat` file (Swiss-Prot's internal entry order), not
alphabetically sorted by accession as the legacy CSV was. Any downstream code
that does positional head/tail sampling or derives train/val splits from row
indices will see different data between the legacy and the regenerated CSV
even when the two have identical entry sets. Sort by `primary_Accession` before
such positional slicing if you want reproducibility across the two formats.

**Provenance.** Each builder writes a `<output_stem>.build_manifest.json`
next to the output CSV (e.g. `fully_annotated_swiss_prot.csv` yields
`fully_annotated_swiss_prot.build_manifest.json`) capturing the input file
sizes/mtimes, the UniProt `reldate.txt` release version and Pfam
`relnotes.txt` version when present alongside the inputs, plus the full
`args`, `git_hash`, and `biom3_version`. The stem-based name lets multiple
source CSVs coexist in the same directory without overwriting each other's
manifest. Shape is otherwise identical to the `build_manifest.json` that
`biom3_build_dataset` writes.

For per-column provenance (which `.dat` lines map to which caption fields), see
[training_csv_provenance.md](training_csv_provenance.md).

### Annotated Pfam subsets from `Pfam-A.full.gz` (non-redundant scope)

`biom3_build_source_pfam` reads `Pfam-A.fasta.gz`, which Pfam ships as a
**90% non-redundant** FASTA (see `relnotes.txt` line 224). For large
families this aggressively collapses near-duplicate sequences — SH3
(PF00018) comes out to 26,468 rows in Pfam 38.1. The full `Pfam-A.full.gz`
Stockholm alignment carries every reference-proteome hit (176,301 rows
for PF00018 in 38.1) but isn't materialized as a CSV by the main builder.

For finetuning datasets that want the full RP coverage, use
`biom3_build_annotated_pfam_subsets`. It streams `Pfam-A.full.gz`
directly, extracts only the requested families, strips Stockholm gap
characters (`.` / `-`), uppercases insertion residues, and emits a CSV
enriched with Pfam-level annotations beyond what the main builder
captures: `family_type`, `family_clan`, `family_wikipedia`, and
`family_references` (all `#=GF` header fields joined per row).

```bash
biom3_build_annotated_pfam_subsets \
    -p PF00018 PF07714 \
    --pfam_full data/databases/pfam/Pfam-A.full.gz \
    -o outputs/SH3_Pkinase_full.csv
```

| CLI | Source file | Scope | PF00018 row count |
|-----|-------------|-------|-------------------|
| `biom3_build_source_pfam` | `Pfam-A.fasta.gz` | RP, 90% non-redundant | 26,468 |
| `biom3_build_annotated_pfam_subsets` | `Pfam-A.full.gz` | RP, **non-redundant** | **176,301** |

Output schema (11 columns): `id, range, pfam_label, sequence,
family_name, family_description, family_type, family_clan,
family_wikipedia, family_references, [final]text_caption`. The extra
`family_*` columns exist as separate fields (not folded into the
caption) so downstream training code can random-subsample them
independently. Runtime on the DGX Spark data share is roughly 5–6
minutes regardless of how many families you request — the scan cost is
dominated by decompressing the 23 GB input, and multi-family requests
share the same single pass. The builder warns (but does not fail) if
any requested Pfam ID yields zero rows, e.g. a typo or an obsolete
accession.

Everything beyond the row-count lift (stats markdown, build manifest
with Pfam release version, `IncrementalStatsBuilder`-backed coverage
report) matches the other `biom3_build_source_*` tools.

---

## Step 1: Configure database paths

Paths are resolved by `biom3.dbio.config` with this priority:

1. **Explicit path arguments** (`--swissprot`, `--pfam`)
2. **Environment variable** `BIOM3_DATABASES_ROOT`
3. **Config file** `configs/dbio_config.json`

The default config looks like:

```json
{
    "databases_root": "data/databases",
    "training_data_root": "../BioM3-data-share/data/datasets",
    "databases": {
        "ncbi_taxonomy": "ncbi_taxonomy",
        "pfam": "pfam",
        "swissprot": "swissprot"
    },
    "training_datasets": {
        "swissprot_csv": "fully_annotated_swiss_prot.csv",
        "pfam_csv": "Pfam_protein_text_dataset.csv"
    }
}
```

For quick experiments, you can skip configuration entirely by passing paths
directly:

```bash
biom3_build_dataset \
    -p PF00018 \
    -o output/sh3_dataset \
    --swissprot /path/to/fully_annotated_swiss_prot.csv \
    --pfam /path/to/Pfam_protein_text_dataset.csv
```

---

## Step 2: Query SwissProt

`SwissProtReader` loads the entire CSV into memory on first query (lazy), then
filters using regex substring matching on the `pfam_label` column. This is
necessary because SwissProt stores multi-domain annotations as stringified
Python lists (e.g., `"['PF00018', 'PF07714']"`).

```python
from biom3.dbio.swissprot import SwissProtReader

reader = SwissProtReader("/path/to/fully_annotated_swiss_prot.csv")
df_sp = reader.query_by_pfam(["PF00018"])

print(df_sp.shape)           # e.g. (1234, 4)
print(df_sp.columns.tolist()) # ['primary_Accession', 'protein_sequence',
                               #  '[final]text_caption', 'pfam_label']
```

Because SwissProt entries have curated text captions, the `[final]text_caption`
column is typically descriptive (e.g., "Tyrosine-protein kinase Fyn.").

---

## Step 3: Query Pfam

`PfamReader` streams the large Pfam CSV in chunks to avoid loading ~35 GB into
memory. Each row has a single Pfam ID, so filtering uses exact `isin()` matching.

```python
from biom3.dbio.pfam import PfamReader

reader = PfamReader("/path/to/Pfam_protein_text_dataset.csv", chunk_size=500_000)
df_pfam = reader.query_by_pfam(["PF00018"])

print(df_pfam.shape)  # e.g. (15000, 4)
```

The Pfam CSV has different source columns (`id`, `sequence`, etc.) which are
automatically renamed to match the standard output columns.

To preserve family metadata for downstream enrichment, pass `keep_family_cols=True`:

```python
df_pfam = reader.query_by_pfam(["PF00018"], keep_family_cols=True)
# Adds: family_name, family_description
```

---

## Step 4: Combine and save

Concatenate the two DataFrames and write to disk:

```python
import pandas as pd

df_combined = pd.concat([df_sp, df_pfam], ignore_index=True)
df_combined.to_csv("output/sh3_dataset/dataset.csv", index=False)
```

You can also save the Pfam IDs for provenance:

```python
pd.DataFrame({"pfam_id": ["PF00018"]}).to_csv("output/sh3_dataset/pfam_ids.csv", index=False)
```

---

## Step 5 (optional): Enrich text captions

The base Pfam captions are minimal (e.g., "Protein name: SH3 domain."). For
richer training signal, you can enrich them with UniProt annotations and/or
NCBI taxonomy lineage.

Enrichment is a two-step process:
1. **`enrich_dataframe()`** — populates individual `annot_*` columns (e.g.,
   `annot_protein_name`, `annot_function`, `annot_lineage`) from one or more
   data sources.
2. **`compose_caption()`** — assembles `[final]text_caption` from those columns
   using the BioM3 ALL-CAPS field label format.

### Local `.dat` file enrichment (recommended)

Parses `uniprot_sprot.dat.gz` (and optionally `uniprot_trembl.dat.gz`) locally
— no API calls needed:

```python
from biom3.dbio.swissprot_dat import SwissProtDatParser
from biom3.dbio.enrich import enrich_dataframe, compose_caption

accessions = df_pfam["primary_Accession"].dropna().unique().tolist()

parser = SwissProtDatParser("data/databases/swissprot/uniprot_sprot.dat.gz")
local_annotations = parser.parse(accessions)

df_enriched = enrich_dataframe(df_pfam, local_annotations=local_annotations)
df_enriched = compose_caption(df_enriched)
```

**Note**: Pfam accessions are overwhelmingly from TrEMBL (unreviewed UniProt).
`uniprot_sprot.dat.gz` only covers ~568K reviewed entries and will match very
few Pfam accessions. For full coverage, also parse the TrEMBL file:

```python
# Parse Swiss-Prot first, then TrEMBL for remaining accessions
parser_sp = SwissProtDatParser("data/databases/swissprot/uniprot_sprot.dat.gz")
local_annotations = parser_sp.parse(accessions)

remaining = set(accessions) - set(local_annotations.keys())
parser_tr = SwissProtDatParser("data/databases/swissprot/uniprot_trembl.dat.gz")
local_annotations.update(parser_tr.parse(remaining))
```

### Annotation cache (recommended for repeated builds)

Parsing TrEMBL's 161 GB `.dat.gz` file takes hours, even with parallel
decompression. If you plan to build datasets for multiple Pfam families, you
can parse TrEMBL once into a Parquet cache and then use instant lookups on
every subsequent build.

**One-time: build the cache**

```bash
biom3_build_annotation_cache \
    --dat data/databases/trembl/uniprot_trembl.dat.gz \
    -o data/databases/trembl/trembl_annotations.parquet
```

This produces a Parquet file with one row per UniProt entry that has at least
one annotation (protein name, function, lineage, etc.). Entries with no
annotations are skipped by default (pass `--no_require_annotation` to include
them). The output is typically a few GB for TrEMBL.

You can also build a cache from Swiss-Prot:

```bash
biom3_build_annotation_cache \
    --dat data/databases/swissprot/uniprot_sprot.dat.gz \
    -o data/databases/swissprot/swissprot_annotations.parquet
```

**Per-build: use the cache**

```bash
biom3_build_dataset -p PF00018 -o output/sh3_enriched --enrich_pfam \
    --annotation_cache data/databases/trembl/trembl_annotations.parquet
```

Multiple cache files can be passed (e.g. Swiss-Prot + TrEMBL):

```bash
biom3_build_dataset -p PF00018 -o output/sh3_enriched --enrich_pfam \
    --annotation_cache data/databases/swissprot/swissprot_annotations.parquet \
                       data/databases/trembl/trembl_annotations.parquet
```

The cache is checked first. If `--uniprot_dat` is also provided, it serves
as a fallback — only accessions not found in the cache are looked up via the
raw `.dat` file. This lets you combine a pre-built cache with a freshly
downloaded `.dat` for maximum coverage.

### UniProt REST API enrichment (fallback)

When local `.dat` files are not available, fetch annotations via the UniProt
REST API (with disk caching and rate limiting):

```python
from biom3.dbio.uniprot_client import UniProtClient
from biom3.dbio.enrich import enrich_dataframe, compose_caption

client = UniProtClient(cache_dir=".uniprot_cache")
accessions = df_pfam["primary_Accession"].dropna().unique().tolist()
uniprot_data = client.fetch_all(accessions, batch_size=25)

df_enriched = enrich_dataframe(df_pfam, uniprot_data=uniprot_data)
df_enriched = compose_caption(df_enriched)
```

### Caption format

The enrichment adds up to 18 annotation columns. `compose_caption()` assembles
them into a single text string using BioM3's ALL-CAPS field label format:

```
FAMILY NAME: SH3 domain. FAMILY DESCRIPTION: ... PROTEIN NAME: Tyrosine-protein kinase Fyn. FUNCTION: ...
```

### NCBI taxonomy lineage

Adds organism lineage from local NCBI taxonomy files (no API needed):

```python
from biom3.dbio.taxonomy import TaxonomyTree, AccessionTaxidMapper

tree = TaxonomyTree("/path/to/ncbi_taxonomy")
tree.load()

mapper = AccessionTaxidMapper("/path/to/ncbi_taxonomy/prot.accession2taxid.gz")
acc_to_taxid = mapper.lookup(accessions)

df_enriched = enrich_dataframe(df_pfam, taxonomy_tree=tree,
                                accession_taxid_map=acc_to_taxid)
```

All enrichment sources can be combined in a single `enrich_dataframe()` call:

```python
df_enriched = enrich_dataframe(
    df_pfam,
    local_annotations=local_annotations,
    taxonomy_tree=tree,
    accession_taxid_map=acc_to_taxid,
)
df_enriched = compose_caption(df_enriched)
```

### Taxonomy filtering

After enrichment, you can filter by taxonomic rank:

```python
kept_taxids = tree.filter_by_rank(
    set(acc_to_taxid.values()), "superkingdom", include={"Bacteria"}
)
```

---

## CLI: `biom3_build_dataset`

The `biom3_build_dataset` command wraps all of the above into a single
invocation:

```bash
# Basic: extract SH3 from both databases
biom3_build_dataset -p PF00018 -o output/sh3_dataset

# Multiple families at once
biom3_build_dataset -p PF00018 PF07714 -o output/sh3_kinase_dataset

# With UniProt enrichment (via REST API)
biom3_build_dataset -p PF00018 -o output/sh3_enriched --enrich_pfam

# With offline enrichment from local .dat files
biom3_build_dataset -p PF00018 -o output/sh3_enriched --enrich_pfam \
    --uniprot_dat data/databases/swissprot/uniprot_sprot.dat.gz \
                  data/databases/trembl/uniprot_trembl.dat.gz

# With pre-built annotation cache (fastest — see "Annotation cache" section)
biom3_build_dataset -p PF00018 -o output/sh3_enriched --enrich_pfam \
    --annotation_cache data/databases/trembl/trembl_annotations.parquet

# With taxonomy lineage
biom3_build_dataset -p PF00018 -o output/sh3_taxonomy --add_taxonomy

# With taxonomy filtering (Bacteria only)
biom3_build_dataset -p PF00018 -o output/sh3_bacteria \
    --add_taxonomy --taxonomy_filter "superkingdom=Bacteria"

# Explicit paths (skip config resolution)
biom3_build_dataset -p PF00018 -o output/sh3_dataset \
    --swissprot /path/to/fully_annotated_swiss_prot.csv \
    --pfam /path/to/Pfam_protein_text_dataset.csv
```

### Full argument reference

| Argument | Default | Description |
|----------|---------|-------------|
| `-p`, `--pfam_ids` | *(required)* | One or more Pfam IDs (e.g., `PF00018 PF07714`) |
| `-o`, `--outdir` | *(required)* | Output directory |
| `--swissprot` | from config | Path to SwissProt CSV |
| `--pfam` | from config | Path to Pfam CSV |
| `--databases_root` | from config | Override database root path |
| `--config` | `configs/dbio_config.json` | Path to config JSON |
| `--chunk_size` | `500000` | Chunk size for Pfam CSV reading |
| `--enrich_pfam` | off | Enrich captions with UniProt annotations (API by default) |
| `--annotation_cache` | none | Pre-built annotation Parquet cache(s) for fast enrichment (see above) |
| `--uniprot_dat` | none | Use local `.dat.gz` file(s) instead of API (accepts multiple paths) |
| `--add_taxonomy` | off | Add NCBI taxonomy lineage |
| `--taxonomy_filter` | none | Filter by rank (e.g., `"superkingdom=Bacteria"`) |
| `--uniprot_cache_dir` | `.uniprot_cache` | Cache directory for API responses |
| `--uniprot_batch_size` | `25` | Batch size for UniProt API requests |

### Output files

```
output/sh3_dataset/
├── dataset.csv                 # Final dataset (standard 4-column format, quoted)
├── dataset_annotations.csv     # Intermediate with all annot_* columns
├── pfam_ids.csv                # Pfam IDs used for extraction
├── build.log                   # Full log output
└── build_manifest.json         # Reproducibility manifest (version, args, paths, row counts)
```

---

## What comes next

The output `dataset.csv` feeds directly into the BioM3 embedding pipeline:

```
dataset.csv
  → Stage 1: PenCL inference (biom3_PenCL_inference)
      Encodes sequences (ESM-2) and captions (BioBERT) into joint embeddings
  → Stage 2: Facilitator sampling (biom3_Facilitator_sample)
      Maps text embeddings to protein-aligned space
  → Stage 3: ProteoScribe sampling (biom3_ProteoScribe_sample)
      Generates novel protein sequences via diffusion
```

See `demos/SH3/` for example shell wrappers that run SH3 data through
this pipeline.
