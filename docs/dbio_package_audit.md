# `biom3.dbio` package audit

This document is a structural audit of the `biom3.dbio` subpackage —
what each file does, how the layers fit together, and where the
backward-compatibility surfaces, tensions, and open items live. It is
intended as an onboarding map for anyone making non-trivial changes to
dbio, complementing the user-facing
[building_datasets_with_dbio.md](building_datasets_with_dbio.md) and the
column-level reference in
[training_csv_provenance.md](training_csv_provenance.md).

The audit reflects the package state after the v0.1.0a7 refactor
(2026-05-06): 25 source files plus a parallel test layout, organized
into per-layer subpackages. The pre-refactor v0.1.0a6 layout had 26
files in a single flat directory.

## Purpose

`biom3.dbio` turns raw biological reference databases (UniProt
SwissProt/TrEMBL `.dat`, Pfam Stockholm/FASTA, ExPASy/SMART/BRENDA
flatfiles, NCBI taxonomy `.dmp`) into versioned, captioned CSVs that
the BioM3 training and finetuning stages consume. Every output CSV
ships with a `<stem>.build_manifest.json` (provenance) and a
`<stem>.stats.md` (per-column coverage report) sidecar.

## Architectural layers

The four conceptual layers are now expressed in the directory tree —
each subpackage is one layer. Cross-cutting modules stay at the
top level. Reading order tracks the dependency graph, low layers first:

```
src/biom3/dbio/
├── __init__.py             ← empty
├── __main__.py             ← CLI dispatch
├── config.py               ← cross-cutting: path resolution
├── caption.py              ← cross-cutting: CaptionSpec + composers
├── stats.py                ← cross-cutting: coverage report
├── enrich.py               ← cross-cutting: annotation + join layer
│
├── parsers/                ← Layer 1: raw-DB format parsers
├── readers/                ← Layer 2: filtered/indexed CSV readers
├── builders/               ← Layer 3: primary source-CSV builders
├── helpers/                ← Layer 3 (derived performance artifacts)
└── pipelines/              ← Layer 4: combine multiple sources
```

The legacy `uniprot_client.py` (UniProt REST API client) was removed
in v0.1.0a7 in favor of the local `.dat` parser + Parquet annotation
cache, which together cover every workflow that previously needed
the API.

---

## Cross-cutting modules

### [`__init__.py`](../src/biom3/dbio/__init__.py) (0 lines)

Empty. Follows the repo convention that subpackage `__init__.py` files
don't re-export.

### [`__main__.py`](../src/biom3/dbio/__main__.py) (90 lines)

Pure dispatch table for the CLI entry points registered in
`pyproject.toml`. Each `run_*()` function lazily imports the relevant
module's `parse_arguments` + `main` and runs them. Lazy imports keep
startup cheap when only one CLI is being invoked. Nine entry points
are wired here:

- `run_build_dataset` → `biom3_build_dataset`
- `run_build_taxid_index` → `biom3_build_taxid_index` (inlined rather
  than delegating)
- `run_convert_to_parquet` → `biom3_convert_to_parquet`
- `run_build_source_{swissprot,pfam,trembl,expasy,smart,brenda}` —
  six per-database source-CSV builders
- `run_build_annotated_pfam_subsets` → `biom3_build_annotated_pfam_subsets`
- `run_build_annotation_cache` → `biom3_build_annotation_cache`

### [`config.py`](../src/biom3/dbio/config.py) (86 lines)

Database path resolution. Three layers of priority, low → high:

- `configs/dbio_config.json` defaults
- env var `BIOM3_DATABASES_ROOT`
- explicit argparse flags

Public functions: `get_databases_root()`,
`get_database_path(database_name)` (e.g., `"swissprot"` →
`data/databases/swissprot/`), `get_training_data_root()`,
`get_training_data_path(dataset_name)`. Lets the same code run on
Spark / Polaris / Aurora without hard-coded paths.

### [`caption.py`](../src/biom3/dbio/caption.py) (135 lines)

Configurable caption composition — the data-augmentation surface area.
Two pieces:

- **`CaptionSpec`** — a dataclass parameterizing `fields` (label +
  annotation key), `field_template`, `separator`, `strip_pubmed`,
  `strip_evidence`, `family_names_label`, `family_names_template`,
  `trailing_period`. Different builders subclass this with different
  specs (e.g., `SWISSPROT_SPEC` strips PubMed and ECO; `PFAM_SPEC`
  doesn't).
- **`compose_row_caption(annotations, spec, pfam_family_names=None)`**
  — pure caption-composition function. Walks `spec.fields`, applies
  stripping, joins.

Also exports `strip_pubmed_refs()`, `strip_evidence_tags()`,
`build_lineage_string()` (for taxonomic prefixes), and a canonical
`TAXONOMY_RANKS` ordering.

### [`stats.py`](../src/biom3/dbio/stats.py) (379 lines)

Coverage-stats infrastructure. Every builder calls into this to emit a
`<stem>.stats.md` next to the CSV and embed the same dict under
`manifest.stats`. Three public surfaces:

- `compute_coverage_stats(df, …)` — one-shot stats from a DataFrame.
- `IncrementalStatsBuilder` — streaming version for builders that
  don't keep the whole frame in memory.
- `format_stats_markdown(stats)` / `write_stats_markdown(stats, path,
  title=…)` — pretty-print to `.md`.

Tracks: row count, sequence-length quantiles, caption coverage +
length, per-`annot_*`-column populated/% / mean-chars / max-chars,
top-N Pfam families. The `_EMPTY_SENTINELS` set explicitly handles
`pd.NA` stringified to `"<na>"`, plus `"['nan']"` (the legacy
SwissProt sentinel for Pfam-less rows).

### [`enrich.py`](../src/biom3/dbio/enrich.py) (~580 lines)

The two-step enrichment pipeline that turns a thin `(accession,
sequence, pfam_label)` row into a fully-captioned training example.

- **Step 1: `enrich_dataframe(df, …)`** populates per-row `annot_*`
  columns from local `.dat` parser output / annotation Parquet
  cache, NCBI taxonomy lookup, and ExPASy/BRENDA/SMART source-CSV
  joins. Returns `(df, join_stats)` so callers can record hit rates
  in the manifest.
- **Step 2: `compose_caption(df, …)`** assembles `[final]text_caption`
  from the populated `annot_*` columns using the `ANNOTATION_FIELDS`
  ordering — ALL-CAPS labels matching the BioM3 paper.

Key constants:

- **`ANNOTATION_FIELDS`** — the canonical ordering. Original 18 fields
  + 7 source-join fields (`annot_ec_names` / `annot_ec_description` /
  `annot_smart_domains` / `annot_brenda_*`) appended at the end so
  legacy captions don't shift.
- **`ANNOTATION_COLUMNS`** = caption-feeding columns;
  **`EXTRA_ANNOTATION_COLUMNS`** = `["annot_ec_numbers"]`
  (structured, not in caption); **`ALL_ANNOTATION_COLUMNS`** = both.

Helpers for EC-number extraction (`extract_ec_numbers` with
caption-text fallback) and the three private join helpers
(`_join_expasy`, `_join_brenda`, `_join_smart`) with hit-rate stats.
The legacy UniProt JSON parsers (`parse_protein_name`,
`parse_gene_ontology`, `parse_subcellular_location`,
`extract_annotations`) were removed in v0.1.0a7 along with the API
client.

---

## Layer 1 — `parsers/` (raw-database parsers)

Pure parsers — they read a file and yield structured records. Zero
pandas, zero IO orchestration, zero opinions about output shape. File
naming: `<dbname>_<format>` (e.g., `swissprot_dat`, `pfam_stockholm`).

### [`parsers/swissprot_dat.py`](../src/biom3/dbio/parsers/swissprot_dat.py) (628 lines)

The workhorse. Parses UniProt SwissProt/TrEMBL `.dat` flat files. Handles
all the quirks:

- **`pigz` path optimization** — `_open_gz()` shells out to `pigz -dc`
  if available (4–8× faster than Python's `gzip` on TrEMBL's 161 GB
  file).
- **Binary-mode reads** — accession matching uses `bytes` comparison
  so 99.99% of lines never get UTF-8 decoded.
- **Two API surfaces**: `parse(accessions)` (targeted, used by
  enrichment) vs `parse_all()` (bulk generator, used by source
  builders).
- **`_parse_entry_full()`** — extracts protein sequence (`SQ`), Pfam
  DR cross-refs (`DR Pfam;`), GO terms (`DR GO;`), OX tax_id, OC
  lineage, plus all CC topics (`FUNCTION`, `CATALYTIC ACTIVITY`,
  `COFACTOR`, `SUBCELLULAR LOCATION`, `TISSUE SPECIFICITY`,
  `DEVELOPMENTAL STAGE`, `BIOTECHNOLOGY`, etc.).
- **EC harvesting** — `_extract_ec_numbers_from_lines()` scans `DE …
  EC=N.N.N.N` and `CC -!- CATALYTIC ACTIVITY` blocks for cross-refs to
  build the structured `annot_ec_numbers` column.
- **Cross-reference side-channels** — captures `DR SMART;` /
  `DR InterPro;` / `DR PDB;` into `xref_smart_ids` /
  `xref_interpro_ids` / `xref_pdb_ids` for the enrichment join layer.

### [`parsers/pfam_stockholm.py`](../src/biom3/dbio/parsers/pfam_stockholm.py) (168 lines)

Parses `Pfam-A.full.gz` (Stockholm) or `Pfam-A.hmm.gz` for **family-level
metadata only** — not per-protein hits. Returns a `PF_ID → {short_id,
family_name, family_description, family_type, family_clan,
family_wikipedia, family_references}` dict. Stockholm is preferred
because it carries `#=GF CC` (description) and `#=GF TP/CL/WK/RT` lines
that the HMM file lacks.

### [`parsers/expasy_dat.py`](../src/biom3/dbio/parsers/expasy_dat.py) (148 lines)

Streaming parser for `enzyme.dat`. Yields `EnzymeEntry` dataclass
instances with `ec`, `name`, `alternative_names`, `catalytic_activities`,
`cofactors`, `comments`, `uniprot_accessions`, `transferred_to`,
`deleted`. Handles section-delimited `//` records and the 2-character
line codes (`ID`, `DE`, `AN`, `CA`, `CF`, `CC`, `DR`). Detects
"Transferred entry: X.Y.Z.W" / "Deleted entry" markers in the DE
field.

### [`parsers/smart_tsv.py`](../src/biom3/dbio/parsers/smart_tsv.py) (44 lines)

Trivial TSV reader for `SMART_domains.txt`. `SmartReader.iter_domains()`
yields dicts with `domain_name`, `accession`, `definition`,
`description`. Skips header line and dashed separator. Smallest parser
in the package.

### [`parsers/brenda_flatfile.py`](../src/biom3/dbio/parsers/brenda_flatfile.py) (241 lines)

Parser for the BRENDA flatfile. Section-delimited per EC, with the most
complex format in dbio (40+ section codes, per-organism record fanout
via `#N#` / `#N,M,…#` tags, reference tail-tags `<3,7,12>`). Currently
captures only the high-value sections in `TRACKED_CODES = {"PR", "RN",
"SN", "SY", "RE", "SP", "KM", "PHO", "TO"}` — protein,
recommended/systematic name, synonyms, reaction, substrate-product,
KM, pH optimum, temperature optimum. Yields `BrendaEntry` with
per-organism dicts for the kinetic fields.

---

## Layer 2 — `readers/` (filtered/indexed readers)

`DatabaseReader` subclasses that consume the **output** CSVs of the
source builders, not the raw databases. File naming: `<dbname>_csv`
for CSV/Parquet readers.

### [`readers/base.py`](../src/biom3/dbio/readers/base.py) (26 lines)

Defines the `DatabaseReader` ABC: `name` property + abstract
`query_by_pfam(pfam_ids, **kwargs)`. Provides extensibility for future
readers (PDB / SCOPe / CATH per
[database_linkage.md](database_linkage.md)). Currently only
`SwissProtReader` and `PfamReader` subclass it; the other parsers
(BRENDA, ExPASy, SMART) don't because they're consumed via the join
layer rather than `query_by_pfam`.

### [`readers/swissprot_csv.py`](../src/biom3/dbio/readers/swissprot_csv.py) (84 lines)

`SwissProtReader.query_by_pfam(pfam_ids)`. Loads the entire
`fully_annotated_swiss_prot.csv` (~570K rows) into memory once, then
regex-matches `pfam_label` (which stores stringified Python lists).
Auto-detects a Parquet sibling and prefers it.
**`OPTIONAL_OUTPUT_COLS = ["annot_ec_numbers"]`** is the flag that
lets newer 5-column CSVs flow through `query_by_pfam` while older
4-column CSVs still load — this is what makes the `--no_emit_ec_numbers`
toggle (commit `46106f3`) backward compatible.

### [`readers/pfam_csv.py`](../src/biom3/dbio/readers/pfam_csv.py) (124 lines)

`PfamReader.query_by_pfam(pfam_ids, keep_family_cols=False)`. Reads
the much-larger `Pfam_protein_text_dataset.csv` (~63M rows). Two
paths:

- **Parquet**: `pq.read_table(filters=[("pfam_label", "in", pfam_ids)])`
  — instant predicate pushdown.
- **CSV**: chunked read with `chunksize=500_000` and tqdm progress
  bar. Slow but works without the Parquet conversion.

The `keep_family_cols` flag is for the enrichment pipeline, which
needs `family_name`/`family_description` to compose Pfam captions.

### [`readers/taxonomy.py`](../src/biom3/dbio/readers/taxonomy.py) (282 lines)

NCBI taxonomy. Two classes — splitting these into separate files is
deferred:

- **`TaxonomyTree`** — loads `rankedlineage.dmp` (~2.7M rows, 300–500 MB
  RAM) into a dict for O(1) lineage lookups. `get_lineage(tax_id)`
  returns a per-rank dict; `get_lineage_string(tax_id)` formats it for
  caption inclusion; `filter_by_rank(tax_ids, rank, include=, exclude=)`
  filters by superkingdom / phylum / class / etc.
- **`AccessionTaxidMapper`** — maps protein accessions to NCBI
  tax_ids. Two strategies: streaming `prot.accession2taxid.gz`
  (~1.55B rows, ~10–15 min per scan) or a one-time SQLite index
  (`build_sqlite_index()`) for instant subsequent lookups. `lookup()`
  auto-detects which to use. SQLite query batch size is 500 to stay
  under SQLite's variable limit.

---

## Layer 3 — `builders/` (primary source-CSV builders)

Per-database CLI builders. Each writes a CSV plus
`<stem>.build_manifest.json` (provenance) and `<stem>.stats.md`
(coverage). All emit ALL-CAPS-prefixed `[final]text_caption` via
`caption.py`. File naming: `source_<dbname>` for primary (whole-DB)
builders; bare names for selective per-family builders.

### [`builders/source_swissprot.py`](../src/biom3/dbio/builders/source_swissprot.py) (399 lines)

Foundational builder #1. Streams `uniprot_sprot.dat.gz` via
`SwissProtDatParser.parse_all()`, joins family names from
`PfamMetadataParser`, composes captions via `SWISSPROT_SPEC`
(PubMed-stripped, ECO-stripped). Output schema is one of four:

| `--keep_intermediate_captions` | `--emit_ec_numbers` (default) | `--no_emit_ec_numbers` |
|---|---|---|
| **off** (default) | 5 cols (modern) | 4 cols (legacy parity) |
| **on** | 7 cols | 6 cols (full legacy parity) |

### [`builders/source_pfam.py`](../src/biom3/dbio/builders/source_pfam.py) (289 lines)

Foundational builder #2. Streams `Pfam-A.fasta.gz` (90%-NR, RP-scoped
since Pfam 37.1), parses the FASTA header (3 whitespace-separated
parts), joins family metadata from `Pfam-A.full.gz`, composes a
2-field caption via `PFAM_SPEC`
(`"Protein name: {family_name}. Family description: {family_description}"`,
no PubMed/ECO stripping). Always emits 8 columns. ~63M rows, ~52 GB
on Pfam 38.1.

### [`builders/source_trembl.py`](../src/biom3/dbio/builders/source_trembl.py) (339 lines)

Same schema as the SwissProt builder, but parses
`uniprot_trembl.dat.gz` (~250M entries). Adds `--evidence_filter
{strict,lenient,any}` because TrEMBL is unreviewed and the codes in
`AUTOMATIC_ECO_CODES = {"ECO:0000256", "ECO:0000313"}` (per the in-code
definitions) dominate the file. Default `lenient` drops entries whose
only evidence codes are those two. `--strict` requires
`ECO:0000269` (`EXPERIMENTAL_ECO_CODE`); `--any` keeps all rows.

### [`builders/source_expasy.py`](../src/biom3/dbio/builders/source_expasy.py) (203 lines)

Builds `expasy_enzyme.csv` from `enzyme.dat`. Per-row schema: EC,
name, reactions, cofactors, UniProt cross-refs. Has
`--include_obsolete` / `--exclude_obsolete` to keep or drop
"Transferred entry" / "Deleted entry" rows. ~8,441 rows on the current
ExPASy release.

### [`builders/source_smart.py`](../src/biom3/dbio/builders/source_smart.py) (126 lines)

Builds `smart_domains.csv` from `SMART_domains.txt`. ~1,405 rows, 100%
domain-name coverage, 85% definition coverage. Smallest builder.

### [`builders/source_brenda.py`](../src/biom3/dbio/builders/source_brenda.py) (260 lines)

Builds `brenda_kinetics.csv` — per-organism kinetic data joined from
each EC entry. Output is rows of `(ec, organism_id, organism_name,
recommended_name, …substrates, km_values, ph_optimum,
temperature_optimum)`. ~111,854 rows (6,910 ECs × ~16 organisms each).
Truncates oversized join cells via `_truncate_join` to keep CSV cells
under a char cap.

### [`builders/pfam_subsets.py`](../src/biom3/dbio/builders/pfam_subsets.py) (324 lines)

Streams `Pfam-A.full.gz` (the **non-NR** Stockholm alignment) directly,
extracts only requested families. Unlocks ~6.7× more rows per family
than `source_pfam` — PF00018 goes from 26,468 → 176,301 rows.
11-column schema with `family_type` / `family_clan` /
`family_wikipedia` / `family_references` as separate side fields
(deliberately not folded into the caption, so training-time
augmentation can subsample them independently). Single-pass streaming,
multi-family requests share one scan, early-exits when all targets
are found. The bare name (vs. `source_*` prefix) marks this as a
selective builder rather than a whole-DB one.

---

## Layer 3 — `helpers/` (derived performance artifacts)

These produce derived performance artifacts (Parquet caches /
conversions) rather than primary training-data CSVs. They sit
alongside the source builders because they're CLI-driven and consume
the same raw inputs, but downstream consumers only see them
indirectly — `pipelines/build_dataset.py` uses `--annotation_cache`
paths the first one writes, and the readers in Layer 2 auto-detect
Parquet siblings the second one writes.

### [`helpers/annotation_cache.py`](../src/biom3/dbio/helpers/annotation_cache.py) (204 lines)

Writes a Parquet annotation cache from a `.dat` file. Schema:
`primary_Accession` + 19 `annot_*` columns (sparse, `pa.string()`
nullable), sorted by accession per row group for fast
predicate-pushdown reads. The point is amortizing the multi-hour TrEMBL
`.dat` parse: build the cache once, then any subsequent
`biom3_build_dataset --annotation_cache` invocation gets instant
Pfam-row enrichment without re-streaming TrEMBL. Tightly coupled to
`SwissProtDatParser` and the `enrich.ANNOTATION_COLUMNS` schema.

### [`helpers/csv_to_parquet.py`](../src/biom3/dbio/helpers/csv_to_parquet.py) (108 lines)

Generic CSV → Parquet converter (`biom3_convert_to_parquet`). Reads
the CSV in chunks, writes a single Parquet file with row groups for
partial reads. Database-agnostic at its core; auto-detects the Pfam
CSV by checking for `family_description` in the header and forces it
to `dtype=str` to silence pandas mixed-type warnings, but otherwise
will happily convert any CSV. Used to turn the 35 GB
`Pfam_protein_text_dataset.csv` into a 5–8 GB Parquet for the
`PfamReader` fast path.

---

## Layer 4 — `pipelines/` (the dataset orchestrator)

### [`pipelines/build_dataset.py`](../src/biom3/dbio/pipelines/build_dataset.py) (~660 lines)

The CLI most users actually invoke (`biom3_build_dataset`). Stitches
everything above into a finetuning dataset:

1. Parse args; resolve SwissProt / Pfam / ExPASy / BRENDA / SMART CSV
   paths via `config.py`.
2. Subset SwissProt and Pfam by Pfam IDs (`SwissProtReader.query_by_pfam`
   + `PfamReader.query_by_pfam`).
3. Optionally enrich Pfam rows via `enrich.enrich_dataframe()` using
   the `--annotation_cache` → `--uniprot_dat` chain.
   (`--enrich_pfam` without either of those raises a clear `ValueError`
   — the legacy UniProt REST API path was removed in v0.1.0a7.)
4. Optionally add NCBI taxonomy lineage and filter by
   `--taxonomy_filter "rank=value"`.
5. Optionally run ExPASy/BRENDA/SMART joins (`--use_expasy/brenda/smart`)
   on the combined frame.
6. Compose final captions via `compose_caption()`.
7. Write outputs:
   - `dataset.csv` — the standard finetuning columns
   - `dataset_annotations.csv` — full enriched annotations
   - `build.log` — dual console + file logging
   - `<stem>.build_manifest.json` — git hash, biom3 version, full
     command, resolved paths, row counts, manifest stats
   - `<stem>.stats.md` — per-column coverage report
   - `pfam_ids.csv` — the IDs used

Has `--per_pfam_output` mode that emits one self-contained
subdirectory per Pfam ID instead of an aggregate dataset.

---

## Test layout

`tests/dbio_tests/` mirrors the source layout:

```
tests/dbio_tests/
├── parsers/         test_{swissprot_dat,pfam_stockholm,expasy,smart,brenda}.py
├── readers/         test_{swissprot_csv,pfam_csv,taxonomy}.py
├── builders/        test_{source_swissprot,source_pfam,pfam_subsets}.py
├── helpers/         test_annotation_cache.py
├── pipelines/       test_build_dataset.py
├── test_caption.py
├── test_enrich.py
├── test_enrich_joins.py
├── test_ec_extraction.py
└── test_stats.py
```

Cross-cutting tests (`caption`, `enrich`, `enrich_joins`,
`ec_extraction`, `stats`) stay at the top level alongside the modules
they cover. 245 tests pass under `--quick`.

## Cross-cutting design observations

### Strengths

- **Clean layering, expressed in the directory tree.** Layer = subpackage.
  Each layer can be tested independently and the seams are stable.
- **Provenance everywhere.** Every CSV gets a
  `<stem>.build_manifest.json` + `<stem>.stats.md` sidecar. The
  2026-04-16 legacy-vs-CLI audit workflow is the reason this exists,
  and it's load-bearing — see
  [training_csv_provenance.md](training_csv_provenance.md) for the
  column-level reference and the 2026-04-16 session note for the audit
  context.
- **Filename suffix declares role.** `_dat` / `_stockholm` / `_tsv` /
  `_flatfile` for parsers; `_csv` for readers; `source_` prefix for
  primary builders, bare name for selective ones. New contributors
  can map a file to a layer without reading it.
- **Backward-compat surfaces are explicit.** `OPTIONAL_OUTPUT_COLS` in
  `readers/swissprot_csv.py`, the `_LEGACY` schema constants in
  `builders/source_swissprot.py`. Every "what schema is this CSV?"
  question has an answer.
- **Parquet as a sibling format.** Readers auto-detect `.parquet` next
  to a `.csv` and prefer it. Lets the Pfam path go from
  "chunked-CSV-with-tqdm" to "predicate-pushdown" without changing
  call sites.

### Tensions and open items

- **`pipelines/build_dataset.py` enriches `df_pfam` but not `df_sp`.**
  SwissProt rows skip the join layer because they come from a
  pre-built source CSV. Closing this gap (e.g., re-running
  `enrich_dataframe` on SwissProt rows keyed by `primary_Accession`)
  is the "df_sp enrichment in build_dataset" item still listed in the
  2026-04-18 session note.
- **`enrich.py` is the largest top-level module (~580 lines)** even
  after removing the JSON parsers. A natural seam exists between
  `enrich_dataframe` (the populator) and the join helpers
  (`_join_expasy`, `_join_brenda`, `_join_smart`) — splitting them
  would make the file easier to navigate, but breaking the public API
  of `enrich_dataframe` would ripple. Deferred.
- **`readers/taxonomy.py` mixes two related but distinct classes**
  (`TaxonomyTree` + `AccessionTaxidMapper`). Splitting into
  `readers/tree.py` + `readers/accession_index.py` was considered
  during the v0.1.0a7 refactor and deferred — the two classes are
  always imported together by `pipelines/build_dataset.py`'s
  `_load_taxonomy`, so the split would force two imports for no
  conceptual gain at present.

## Related docs

- [building_datasets_with_dbio.md](building_datasets_with_dbio.md) —
  user-facing CLI usage and recipes.
- [training_csv_provenance.md](training_csv_provenance.md) — column-level
  source mapping for `fully_annotated_swiss_prot.csv` and
  `Pfam_protein_text_dataset.csv`.
- [database_linkage.md](database_linkage.md) — cross-reference spec
  and identifier-family map across DBs.
- [setup_databases.md](setup_databases.md) — database download and
  symlink layout per cluster.
- [dbio_examples.md](dbio_examples.md) — concrete recipe gallery.
