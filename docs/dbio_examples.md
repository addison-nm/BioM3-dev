# `biom3.dbio` — usage examples for FT dataset construction

Example-driven walkthrough of the features added in the
`addison-dbio-ft-datasets` branch: new source builders (TrEMBL,
ExPASy, SMART, BRENDA), enrichment joins, per-Pfam output,
`<stem>.stats.md` reports, and the structured `annot_ec_numbers`
column.

See [docs/database_linkage.md](database_linkage.md) for the canonical
cross-reference spec and [docs/building_datasets_with_dbio.md](building_datasets_with_dbio.md)
for the original Pfam-centric workflow.

## Prerequisites

Source databases symlinked under `data/databases/` via
`scripts/link_data.sh`:

```bash
./scripts/link_data.sh /data/data-share/BioM3-data-share/databases \
                            data/databases
```

Default paths used by every example below are resolved from
[configs/dbio_config.json](../configs/dbio_config.json). Override any
of them with explicit CLI flags if the local layout differs.

## Part 1 — building the source CSVs

Every source builder emits three artifacts beside its output CSV:
`<stem>.csv`, `<stem>.build_manifest.json`, `<stem>.stats.md`.

### ExPASy enzyme nomenclature

```bash
biom3_build_source_expasy \
    --dat data/databases/expasy/enzyme.dat \
    -o data/datasets/expasy_enzyme.csv
```

Expected: ~1–2 seconds, ~8.4k rows. Columns: `ec`, `annot_name`,
`annot_alternative_names`, `annot_catalytic_activity`, `annot_cofactor`,
`annot_comments`, `annot_uniprot_accessions`, `uniprot_count`,
`transferred_to`, `deleted`, `[final]text_caption`.

Stats file shows 100% EC coverage, ~82% catalytic-activity coverage,
68% of ECs have at least one UniProt cross-reference (avg ~30,
max ~4000 for ubiquitous enzymes).

### SMART domain descriptions

```bash
biom3_build_source_smart \
    --input data/databases/smart/SMART_domains.txt \
    -o data/datasets/smart_domains.csv
```

Expected: <1 second, ~1.4k domains. Columns: `domain_id`,
`annot_domain_name`, `annot_definition`, `annot_description`,
`[final]text_caption`. 100% name, ~85% definition, ~75% description.

### BRENDA per-organism kinetics

```bash
biom3_build_source_brenda \
    --input data/databases/brenda/brenda_2026_1.txt \
    -o data/datasets/brenda_kinetics.csv
```

Expected: ~1 minute, ~112k rows (one per `(EC, organism)` pair),
covering ~6.9k ECs × ~55k organisms. Output CSV ~480 MB with
per-field character caps applied (reactions capped at 2000 chars per
row, substrates at 1500, kinetics at 1000, pH/temp at 400). Per-field
coverage in stats: 100% name/reactions, 99.8% synonyms, 81%
substrates, 29–33% kinetics, 26% temperature.

### TrEMBL (long-running, optional)

```bash
biom3_build_source_trembl \
    --dat data/databases/trembl/uniprot_trembl.dat.gz \
    --pfam_metadata data/databases/pfam/Pfam-A.full.gz \
    -o data/datasets/fully_annotated_trembl.csv \
    --evidence_filter lenient \
    --limit 10000  # dev: cap the row count for a smoke test
```

Evidence filter choices:
- `strict` — require at least one `ECO:0000269` (experimental) code.
- `lenient` (default) — require any ECO code other than `ECO:0000256`
  / `ECO:0000313` (pure-automatic).
- `any` — no filter; includes fully automatic entries (max coverage).

Full TrEMBL (~150 GB `.dat.gz`) takes hours; `--limit N` stops after
N rows for iteration. Output includes the new `annot_ec_numbers`
column (see Part 4).

### Rebuild SwissProt with EC extraction

The legacy SwissProt source CSV has 4 columns and no EC numbers.
Regenerating with this branch's builder adds `annot_ec_numbers`:

```bash
biom3_build_source_swissprot \
    --dat data/databases/swissprot/uniprot_sprot.dat.gz \
    --pfam_metadata data/databases/pfam/Pfam-A.full.gz \
    -o data/datasets/fully_annotated_swiss_prot.csv
```

Expected: ~1 minute, ~570k rows, 5 columns (`primary_Accession`,
`protein_sequence`, `[final]text_caption`, `pfam_label`,
`annot_ec_numbers`). The EC column pulls from CC CATALYTIC ACTIVITY
xrefs *and* DE lines, comma-joined (`"3.1.1.74"` or
`"2.7.11.1, 3.1.3.16"` for bifunctional enzymes). Captions themselves
stay prose-clean — EC xrefs are *not* embedded in
`annot_catalytic_activity`.

## Part 2 — building an FT dataset

### Minimal: single Pfam, no enrichment

```bash
biom3_build_dataset \
    --pfam_ids PF00128 \
    --outdir out/PF00128-minimal
```

Expected: ~15 seconds (Parquet path) or ~2 minutes (CSV path) on
PF00128 (alpha-amylase). Produces:

```
out/PF00128-minimal/
├── build.log
├── build_manifest.json
├── dataset.csv              ← primary_Accession, protein_sequence,
│                              [final]text_caption, pfam_label
├── dataset_annotations.csv  ← superset with annot_* columns
├── dataset.stats.md
└── pfam_ids.csv
```

The `.stats.md` reports per-source row counts (swissprot vs pfam),
sequence-length distribution, annotation coverage percentages, and
pfam-family distribution.

### With ExPASy + SMART joins

```bash
biom3_build_dataset \
    --pfam_ids PF00128 \
    --outdir out/PF00128-enriched \
    --use_expasy \
    --use_smart
```

No `--expasy_csv` / `--smart_csv` flags needed — both resolve to
config defaults under `data/datasets/`. Output manifest now includes a
`join_stats` key:

```json
"join_stats": {
  "ec_extraction_rate": 0.47,
  "expasy_hit_rate": 0.45,
  "smart_hit_rate": 0.12
}
```

Hit rates reflect the fraction of rows where an EC was found (in
`annot_ec_numbers` → `annot_catalytic_activity` → `[final]text_caption`
in that precedence order) and where that EC matched an ExPASy entry.
Using a freshly-rebuilt SwissProt source CSV (with
`annot_ec_numbers`) dramatically raises these rates versus the legacy
4-column CSV.

### With BRENDA per-organism kinetics

```bash
biom3_build_dataset \
    --pfam_ids PF00128 \
    --outdir out/PF00128-brenda \
    --use_expasy \
    --use_brenda \
    --organism_match strict
```

`--organism_match` choices:
- `strict` (default) — species-level match from the row's
  `annot_lineage` (last taxon).
- `relaxed` — falls back to genus when strict misses.
- `ec_only` — uses EC-level BRENDA data even when the organism
  doesn't match any BRENDA record.

Strict matching is ideal for downstream training where you want each
row annotated with the specific organism's measured kinetics. Use
`ec_only` for broad coverage (every enzyme row gets *some* BRENDA
context) at the cost of organism specificity.

Added columns: `annot_brenda_substrates`, `annot_brenda_km_values`,
`annot_brenda_ph_optimum`, `annot_brenda_temperature_optimum`.

### Multi-Pfam, per-Pfam output

```bash
biom3_build_dataset \
    --pfam_ids PF00128 PF00169 PF00001 \
    --outdir out/multi-pfam \
    --per_pfam_output \
    --use_expasy --use_smart
```

**Only** per-Pfam subdirectories are emitted — no aggregate top-level
dataset. Each subdir is fully self-contained:

```
out/multi-pfam/
├── build.log
├── PF00001/
│   ├── build_manifest.json
│   ├── dataset.csv
│   ├── dataset_annotations.csv
│   ├── dataset.stats.md
│   └── pfam_ids.csv
├── PF00128/
│   └── ... (same structure)
└── PF00169/
    └── ...
```

Rows are routed to every subdirectory whose Pfam ID appears in their
`pfam_label`, so multi-domain proteins can appear in multiple dirs
(intentional — each subdir represents "all rows associated with
Pfam X"). Each subdir's `dataset.stats.md` reports stats computed on
its own row subset.

### Taxonomy filtering

```bash
biom3_build_dataset \
    --pfam_ids PF00128 \
    --outdir out/PF00128-bacteria \
    --add_taxonomy \
    --taxonomy_filter superkingdom=Bacteria \
    --use_expasy
```

`--add_taxonomy` hits the local NCBI `prot.accession2taxid.gz` +
`TaxonomyTree` (no network). `--taxonomy_filter` can take multiple
rank=value pairs (e.g., `superkingdom=Bacteria class=Gammaproteobacteria`).

## Part 3 — reading the outputs

### Stats markdown

```markdown
# dataset — coverage stats

**Rows:** 17,165

## Sequence length (residues)

| min | mean | median | p95  | max    |
|-----|------|--------|------|--------|
|  20 | 71.5 |    47  |   57 | 18,141 |

## Rows by source

| source    | rows   |
|-----------|--------|
| pfam      | 16,566 |
| swissprot |    599 |

## Annotation coverage

| column                | populated | %      | mean chars | max chars |
|-----------------------|-----------|--------|------------|-----------|
| `annot_family_name`   |    16,566 | 96.5%  |     10.0   |    10     |
| `annot_ec_numbers`    |       412 |  2.4%  |      7.0   |    31     |
| ...                                                               |

## Enrichment join hit rates

| join                | hit rate |
|---------------------|----------|
| ec_extraction_rate  |    47.2% |
| expasy_hit_rate     |    45.1% |
| smart_hit_rate      |    11.8% |
```

Quick-diff across runs to spot regressions in annotation coverage.

### Manifest JSON

```python
import json
with open("out/PF00128-enriched/build_manifest.json") as f:
    m = json.load(f)

print(m["biom3_version"], m["git_hash"])
print(m["database_versions"]["uniprot_release"])
print(m["outputs"]["row_counts"])
print(m["outputs"]["join_stats"])
print(m["stats"])
```

The manifest mirrors the `.stats.md` numbers under a `stats` key
(same dict shape), captures all resolved paths + upstream version
strings (UniProt reldate, Pfam relnotes, source CSV mtime/size), and
records the full CLI command + args for exact reproduction.

## Part 4 — the `annot_ec_numbers` contract

**Canonical:** `annot_ec_numbers` is a comma-separated string of
EC numbers (e.g. `"3.1.1.74"` or `"2.7.11.1, 3.1.3.16"`). Populated
by the source builder from both CC CATALYTIC ACTIVITY xrefs and DE
lines. Empty string when none found.

**Not in captions:** the BioM3 paper's caption format keeps EC xrefs
out of `annot_catalytic_activity` (prose only). `annot_ec_numbers`
is a parallel structured column meant for enrichment joins, not for
BioBERT tokenization.

**Precedence in the enrichment layer:**

1. Source-supplied `annot_ec_numbers` (set by SwissProt/TrEMBL
   builder).
2. EC regex extraction from `annot_catalytic_activity` (caption-prose
   fallback for other data sources).
3. EC regex extraction from `[final]text_caption` (last resort).

Source-populated rows hit ExPASy/BRENDA joins natively with zero
parsing cost. The fallback paths keep the joins useful even when
running against legacy CSVs or captions from non-UniProt sources.

**Backward compatibility:** the legacy 4-column SwissProt CSV
(without `annot_ec_numbers`) still loads via `SwissProtReader` — the
optional column filter in `query_by_pfam` silently omits missing
columns. Rebuild the source CSV with the new builder to pick up the
new column; no downstream code change required.

## Common patterns

### "Rebuild everything from scratch"

```bash
biom3_build_source_swissprot \
    --dat data/databases/swissprot/uniprot_sprot.dat.gz \
    --pfam_metadata data/databases/pfam/Pfam-A.full.gz \
    -o data/datasets/fully_annotated_swiss_prot.csv

biom3_build_source_pfam \
    --fasta data/databases/pfam/Pfam-A.fasta.gz \
    --pfam_metadata data/databases/pfam/Pfam-A.full.gz \
    -o data/datasets/Pfam_protein_text_dataset.csv

biom3_build_source_expasy \
    --dat data/databases/expasy/enzyme.dat \
    -o data/datasets/expasy_enzyme.csv

biom3_build_source_smart \
    --input data/databases/smart/SMART_domains.txt \
    -o data/datasets/smart_domains.csv

biom3_build_source_brenda \
    --input data/databases/brenda/brenda_2026_1.txt \
    -o data/datasets/brenda_kinetics.csv
```

### "Convert source CSVs to Parquet for faster build_dataset"

```bash
biom3_convert_to_parquet data/datasets/fully_annotated_swiss_prot.csv
biom3_convert_to_parquet data/datasets/Pfam_protein_text_dataset.csv
```

`SwissProtReader` and `PfamReader` auto-detect the `.parquet`
sibling — no CLI change needed. Expected speedup on build_dataset:
~10× (2m18s → 13s for a two-Pfam + enrichment build).

### "Build one self-contained FT dataset per enzyme family"

```bash
biom3_build_dataset \
    --pfam_ids PF00001 PF00069 PF00128 PF00150 \
    --outdir out/enzyme-ft-sets \
    --per_pfam_output \
    --use_expasy --use_brenda --use_smart \
    --organism_match strict \
    --add_taxonomy
```

Each subdirectory `out/enzyme-ft-sets/<pfam>/` becomes a drop-in FT
dataset with its own manifest (for reproducibility) and stats
(for quality inspection).
