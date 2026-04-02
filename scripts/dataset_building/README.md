# Dataset Building Scripts

Python scripts that compose `biom3.dbio` building blocks into dataset
"recipes" for research workflows. Copy and modify for new experiments.

For introductory demos, see `demo/` instead.

## Directory layout

```
scripts/dataset_building/
├── README.md                            ← this file
├── build_finetuning_dataset.py          — subset + enrich for a Pfam family
└── build_taxonomy_variants.py           — taxonomy depth study (any family)
```

## Architecture

These scripts import library primitives from `biom3.dbio` and compose them
into specific workflows:

```
biom3.dbio (library)                     scripts/dataset_building/ (recipes)
├── swissprot_dat.parse_all()             ├── build_finetuning_dataset.py
├── pfam_metadata.PfamMetadataParser      └── build_taxonomy_variants.py
├── build_source_pfam.iter_pfam_fasta()
├── build_source_swissprot.build_swissprot_csv()
├── caption.CaptionSpec
├── caption.compose_row_caption()
├── caption.build_lineage_string()
├── taxonomy.TaxonomyTree
└── enrich.compose_caption()
```

## Scripts

### `build_finetuning_dataset.py` — Subset + enrich for a Pfam family

Wraps `biom3_build_dataset` for scripted workflows. Edit the configuration
at the top, then run:

```bash
python scripts/dataset_building/build_finetuning_dataset.py
```

Reads from the legacy CSVs in `data/datasets/` and writes to `outputs/`.

### `build_taxonomy_variants.py` — Taxonomy depth study

Builds multiple versions of the same dataset with different levels of
taxonomic information in the captions. Works with any Pfam family.

```bash
# Build all 6 variants for Chorismate Mutase
python scripts/dataset_building/build_taxonomy_variants.py \
    --pfam PF01817 --name CM \
    -o outputs/taxonomy_study/CM/

# Build all 6 variants for SH3 domain
python scripts/dataset_building/build_taxonomy_variants.py \
    --pfam PF00018 --name SH3 \
    -o outputs/taxonomy_study/SH3/

# Build only specific variants
python scripts/dataset_building/build_taxonomy_variants.py \
    --pfam PF01817 --name CM \
    --variants no_taxonomy shallow full \
    -o outputs/taxonomy_study/CM_subset/
```

### Available taxonomy variants

| Variant | Taxonomy depth | Example lineage |
|---------|---------------|-----------------|
| `no_taxonomy` | None | (no LINEAGE field) |
| `domain_only` | Superkingdom | `Bacteria` |
| `shallow` | Superkingdom + Phylum | `Bacteria, Pseudomonadota` |
| `medium` | Superkingdom → Order | `Bacteria, Pseudomonadota, Gammaproteobacteria, Enterobacterales` |
| `full` | All NCBI ranks | All 8 ranks from superkingdom to species |
| `oc_lineage` | Raw OC from .dat | Unranked, matches legacy format |

Each variant produces a `dataset.csv` ready for the embedding pipeline.
A `README.md` is auto-generated in the output directory with row counts
and caption samples.

## Customizing captions

To change which annotation fields appear in captions, create a `CaptionSpec`:

```python
from biom3.dbio.caption import CaptionSpec

my_spec = CaptionSpec(
    fields=[
        ("PROTEIN NAME", "annot_protein_name"),
        ("FUNCTION", "annot_function"),
    ],
    strip_pubmed=True,
    family_names_label=None,
)
```

Then pass it to a builder:

```python
build_swissprot_csv(..., caption_spec=my_spec)
```

See `demo/custom_caption_format.py` for a complete working example.
