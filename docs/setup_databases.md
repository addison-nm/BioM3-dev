# Shared Reference Databases

Reference databases used by `biom3.dbio` for constructing fine-tuning datasets are stored in a shared directory on each machine. This avoids duplicating large files (tens of GB) across users and project copies. The local `data/databases/` directory in each clone of BioM3-dev is populated with symlinks that point to these shared files.

## Shared database locations

| Machine | Shared databases path | Permissions |
|---------|-----------------------|-------------|
| DGX Spark | `/data/data-share/BioM3-data-share/databases` | read only |
| Polaris (ALCF) | `/grand/NLDesignProtein/sharepoint/BioM3-data-share/databases` | read only |
| Aurora (ALCF) | `/flare/NLDesignProtein/sharepoint/BioM3-data-share/databases` | read only |

The shared directory contains the following database subdirectories:

```
databases/
  ncbi_taxonomy/     # NCBI taxonomy tree + accession-to-taxid mapping (~13 GB)
  pfam/              # Pfam 38.1 HMMs, FASTA, and full alignments (~29 GB)
  swissprot/         # UniProt/Swiss-Prot sequences and annotations (~750 MB)
  smart/             # SMART domain descriptions (~356 KB)
  expasy/            # ExPASy enzyme nomenclature (~24 MB)
  provenance.tsv     # Download timestamps and MD5 checksums for all files
```

## Populating the local `data/databases/` directory

The script `scripts/sync_databases.sh` creates symlinks in your local `data/databases/` directory for any files present in the shared directory that you don't already have locally. Existing local files are left untouched and verified against the shared copy via md5 checksum.

```bash
# Preview what will be linked (recommended before first run)
./scripts/sync_databases.sh <shared_databases_path> data/databases --dry-run

# Create symlinks
./scripts/sync_databases.sh <shared_databases_path> data/databases
```

For example, on DGX Spark:

```bash
./scripts/sync_databases.sh /data/data-share/BioM3-data-share/databases data/databases --dry-run
./scripts/sync_databases.sh /data/data-share/BioM3-data-share/databases data/databases
```

The script will report `MATCH`, `MISMATCH`, or `LINK` for each entry.

## Overriding database paths

The `biom3.dbio` package resolves database paths in the following order:

1. **Environment variable**: set `BIOM3_DATABASES_ROOT` to point to any directory containing the database subdirectories.
2. **Config file**: the default config at `configs/dbio_config.json` sets `databases_root` to `data/databases` (relative to the repo root).
3. **CLI flags**: the `biom3_build_dataset` command accepts `--databases-root` and per-database path overrides (`--swissprot`, `--pfam`).

## Available databases

### NCBI Taxonomy (`ncbi_taxonomy/`)

Provides organism classification and protein-to-taxonomy mapping.

| File | Size | Description |
|------|------|-------------|
| `rankedlineage.dmp` | 347 MB | Pre-computed ranked lineage for each tax_id (species through superkingdom) |
| `nodes.dmp` | 246 MB | Taxonomy tree structure (parent-child relationships and ranks) |
| `names.dmp` | 241 MB | Scientific names, synonyms, and common names for each tax_id |
| `prot.accession2taxid.gz` | 11 GB | Maps ~1.55B protein accessions to taxonomy IDs |

**Performance note**: The `prot.accession2taxid.gz` file is very large. By default, lookups stream the file in chunks (~10-15 min per query). For faster repeated lookups, build a SQLite index once:

```bash
biom3_build_taxid_index data/databases/ncbi_taxonomy/prot.accession2taxid.gz
```

This creates an indexed SQLite database (~25 GB) alongside the source file, reducing lookup time to seconds.

### Pfam (`pfam/`)

Pfam 38.1 protein family database (January 2026).

| File | Size | Description |
|------|------|-------------|
| `Pfam-A.hmm.gz` | 367 MB | Profile HMMs for 25,545 Pfam-A families |
| `Pfam-A.fasta.gz` | 5.8 GB | Seed sequences for each family |
| `Pfam-A.full.gz` | 23 GB | Full-length sequence alignments |

### Swiss-Prot (`swissprot/`)

UniProt/Swiss-Prot release 2026_01 (January 2026). Manually curated protein sequences.

| File | Size | Description |
|------|------|-------------|
| `uniprot_sprot.fasta.gz` | 90 MB | Swiss-Prot FASTA sequences (~568K entries) |
| `uniprot_sprot.dat.gz` | 661 MB | Full flat-file with annotations, GO terms, cross-references |

The `uniprot_sprot.dat.gz` file is used by `--enrich-pfam` to extract protein annotations (protein name, function, catalytic activity, GO terms, lineage, etc.) locally, without requiring UniProt API access.

### SMART (`smart/`)

| File | Size | Description |
|------|------|-------------|
| `SMART_domains.txt` | 352 KB | Tab-delimited domain descriptions (signalling and extracellular domains) |

### ExPASy Enzyme (`expasy/`)

| File | Size | Description |
|------|------|-------------|
| `enzyme.dat` | 9.1 MB | EC number entries with reaction descriptions and synonyms |
| `enzyme.rdf` | 15 MB | RDF/OWL semantic version |

## Databases required by `biom3.dbio`

The `biom3_build_dataset` command requires the pre-processed training CSV files (not the raw database files above). These CSVs live in the shared data directory:

| File | Source location | Description |
|------|-----------------|-------------|
| `fully_annotated_swiss_prot.csv` | `BioM3-data-share/data/datasets/` | ~570K annotated Swiss-Prot entries with text captions |
| `Pfam_protein_text_dataset.csv` | `BioM3-data-share/data/datasets/` | ~44.8M Pfam domain entries with text captions |

Optional flags require additional database files to be synced in `data/databases/`:

| Flag | Required files |
|------|----------------|
| `--enrich-pfam` | None (uses UniProt REST API; results are cached in `--uniprot-cache-dir`) |
| `--enrich-pfam --uniprot-dat <paths>` | One or more local `.dat.gz` files (no API needed) |
| `--add-taxonomy` | `ncbi_taxonomy/rankedlineage.dmp`, `ncbi_taxonomy/prot.accession2taxid.gz` |
| `--taxonomy-filter` | Same as `--add-taxonomy` |

**Note on local enrichment**: Pfam domain accessions are overwhelmingly from TrEMBL (unreviewed UniProt), not Swiss-Prot (reviewed). Using `--uniprot-dat` with only `uniprot_sprot.dat.gz` will match very few Pfam accessions. For full local coverage, also include the TrEMBL flat file:

```bash
biom3_build_dataset -p PF00018 --enrich-pfam \
    --uniprot-dat data/databases/swissprot/uniprot_sprot.dat.gz \
                  data/databases/swissprot/uniprot_trembl.dat.gz \
    -o outputs/SH3_dataset
```

The TrEMBL file (`uniprot_trembl.dat.gz`, ~130 GB) must be downloaded separately — see the [UniProt downloads page](https://www.uniprot.org/help/downloads).

## Provenance tracking

All database downloads are logged in `provenance.tsv` with timestamps and MD5 checksums. This file is synced alongside the databases for reproducibility tracking.
