# Demo Scripts

End-to-end demonstrations of the `biom3.dbio` dataset building pipeline.
Run these to verify the installation works and to examine outputs. All
results go in `demos/outputs/` — nothing is written to the shared
`data/datasets/` directory.

## Shell scripts (run-and-check)

### `build_sh3_dataset.sh` — Finetuning dataset from legacy CSVs

Subsets the pre-existing SwissProt and Pfam training CSVs to build SH3
domain (PF00018) finetuning datasets with four levels of enrichment.

```bash
bash demos/build_sh3_dataset.sh
```

**What it demonstrates:**
- Basic Pfam ID extraction from SwissProt and Pfam source CSVs
- Parquet conversion for faster repeated queries
- UniProt REST API enrichment (protein name, function, GO terms)
- Local `.dat` file enrichment (offline, no API calls)
- NCBI taxonomy lineage + rank-based filtering (bacteria only)

**Outputs** → `demos/outputs/sh3_dataset/`

| Subdirectory | Description |
|-------------|-------------|
| `basic/` | Raw extraction, no enrichment |
| `enriched_api/` | Pfam rows enriched via UniProt REST API |
| `enriched_local/` | Pfam rows enriched via `uniprot_sprot.dat.gz` |
| `enriched_bacteria/` | Enriched + filtered to bacteria only |

Runtime: ~5-10 min (first run includes Parquet conversion).

---

### `build_source_datasets.sh` — Source CSVs from raw databases

Builds the training CSVs themselves from the raw database files, then feeds
them into `biom3_build_dataset` to produce a finetuning dataset — demonstrating
the full raw-to-finetuning pipeline.

```bash
bash demos/build_source_datasets.sh
```

**What it demonstrates:**
- SwissProt CSV construction from `uniprot_sprot.dat.gz`
- Pfam CSV construction from `Pfam-A.fasta.gz` (SH3 subset for speed)
- Newly-built CSVs working as drop-in inputs to `biom3_build_dataset`

**Outputs** → `demos/outputs/source_datasets/`

| File | Description |
|------|-------------|
| `fully_annotated_swiss_prot.csv` | ~547K SwissProt proteins with captions |
| `pfam_sh3_subset.csv` | ~25K SH3 domain sequences with captions |
| `sh3_from_source/dataset.csv` | SH3 finetuning dataset built from above |

Runtime: ~5 min.

---

## Python demos (library API examples)

### `custom_caption_format.py` — Custom CaptionSpec

Shows how `CaptionSpec` controls caption composition — different field
selections, labels, separators, and formatting — without modifying library code.

```bash
python demos/custom_caption_format.py
```

**Outputs** → `demos/outputs/custom_captions/`

| File | Caption style |
|------|--------------|
| `swissprot_minimal.csv` | Protein name + function only |
| `swissprot_lowercase.csv` | Lowercase labels, semicolon separators |

Runtime: ~2 min (two full SwissProt .dat scans).

---

### `minimal_pfam_extract.py` — Direct FASTA iteration

Shows how to use `iter_pfam_fasta()` directly for custom extraction
without the full CSV builder.

```bash
python demos/minimal_pfam_extract.py
```

**Outputs** → `demos/outputs/pfam_extract/sh3_domains.csv`

Runtime: ~2.5 min (scans full Pfam FASTA for SH3 entries).

---

## Output column format

All final `dataset.csv` files share the same 4-column schema expected by
the BioM3 embedding pipeline:

| Column | Description |
|--------|-------------|
| `primary_Accession` | UniProt accession ID |
| `protein_sequence` | Amino acid sequence |
| `[final]text_caption` | Natural language description of the protein |
| `pfam_label` | Pfam family ID(s) |

These are ready to feed into `scripts/embedding_pipeline.sh` for Stage 1
(PenCL) → Stage 2 (Facilitator) → Stage 3 (ProteoScribe) processing.

## See also

- `scripts/dataset_building/` — Python recipes for building custom datasets
  (taxonomy depth studies, finetuning workflows). Meant to be copied and modified.
