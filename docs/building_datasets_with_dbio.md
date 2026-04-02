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
annotations are skipped by default (pass `--no-require-annotation` to include
them). The output is typically a few GB for TrEMBL.

You can also build a cache from Swiss-Prot:

```bash
biom3_build_annotation_cache \
    --dat data/databases/swissprot/uniprot_sprot.dat.gz \
    -o data/databases/swissprot/swissprot_annotations.parquet
```

**Per-build: use the cache**

```bash
biom3_build_dataset -p PF00018 -o output/sh3_enriched --enrich-pfam \
    --annotation-cache data/databases/trembl/trembl_annotations.parquet
```

Multiple cache files can be passed (e.g. Swiss-Prot + TrEMBL):

```bash
biom3_build_dataset -p PF00018 -o output/sh3_enriched --enrich-pfam \
    --annotation-cache data/databases/swissprot/swissprot_annotations.parquet \
                       data/databases/trembl/trembl_annotations.parquet
```

The cache is checked first. If `--uniprot-dat` is also provided, it serves
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
biom3_build_dataset -p PF00018 -o output/sh3_enriched --enrich-pfam

# With offline enrichment from local .dat files
biom3_build_dataset -p PF00018 -o output/sh3_enriched --enrich-pfam \
    --uniprot-dat data/databases/swissprot/uniprot_sprot.dat.gz \
                  data/databases/trembl/uniprot_trembl.dat.gz

# With pre-built annotation cache (fastest — see "Annotation cache" section)
biom3_build_dataset -p PF00018 -o output/sh3_enriched --enrich-pfam \
    --annotation-cache data/databases/trembl/trembl_annotations.parquet

# With taxonomy lineage
biom3_build_dataset -p PF00018 -o output/sh3_taxonomy --add-taxonomy

# With taxonomy filtering (Bacteria only)
biom3_build_dataset -p PF00018 -o output/sh3_bacteria \
    --add-taxonomy --taxonomy-filter "superkingdom=Bacteria"

# Explicit paths (skip config resolution)
biom3_build_dataset -p PF00018 -o output/sh3_dataset \
    --swissprot /path/to/fully_annotated_swiss_prot.csv \
    --pfam /path/to/Pfam_protein_text_dataset.csv
```

### Full argument reference

| Argument | Default | Description |
|----------|---------|-------------|
| `-p`, `--pfam-ids` | *(required)* | One or more Pfam IDs (e.g., `PF00018 PF07714`) |
| `-o`, `--outdir` | *(required)* | Output directory |
| `--swissprot` | from config | Path to SwissProt CSV |
| `--pfam` | from config | Path to Pfam CSV |
| `--databases-root` | from config | Override database root path |
| `--config` | `configs/dbio_config.json` | Path to config JSON |
| `--chunk-size` | `500000` | Chunk size for Pfam CSV reading |
| `--enrich-pfam` | off | Enrich captions with UniProt annotations (API by default) |
| `--annotation-cache` | none | Pre-built annotation Parquet cache(s) for fast enrichment (see above) |
| `--uniprot-dat` | none | Use local `.dat.gz` file(s) instead of API (accepts multiple paths) |
| `--add-taxonomy` | off | Add NCBI taxonomy lineage |
| `--taxonomy-filter` | none | Filter by rank (e.g., `"superkingdom=Bacteria"`) |
| `--uniprot-cache-dir` | `.uniprot_cache` | Cache directory for API responses |
| `--uniprot-batch-size` | `25` | Batch size for UniProt API requests |

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
