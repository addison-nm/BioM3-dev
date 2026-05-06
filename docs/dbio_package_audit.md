# `biom3.dbio` package audit

This document is a structural audit of the `biom3.dbio` subpackage —
what each file does, how the layers fit together, and where the
backward-compatibility surfaces, tensions, and open items live. It is
intended as an onboarding map for anyone making non-trivial changes to
dbio, complementing the user-facing
[building_datasets_with_dbio.md](building_datasets_with_dbio.md) and the
column-level reference in
[training_csv_provenance.md](training_csv_provenance.md).

The audit reflects the package state at git `46106f3` (2026-05-05),
covering 26 files and ~6,300 lines of Python.

## Purpose

`biom3.dbio` turns raw biological reference databases (UniProt
SwissProt/TrEMBL `.dat`, Pfam Stockholm/FASTA, ExPASy/SMART/BRENDA
flatfiles, NCBI taxonomy `.dmp`) into versioned, captioned CSVs that
the BioM3 training and finetuning stages consume. Every output CSV
ships with a `<stem>.build_manifest.json` (provenance) and a
`<stem>.stats.md` (per-column coverage report) sidecar.

## Architectural layers

The 26 files split cleanly into four conceptual layers plus two
cross-cutting utilities. Reading order roughly tracks the dependency
graph (low layers first):

```
┌───────────────────────────────────────────────────────────┐
│ Layer 4: Orchestrator                                     │
│   build_dataset.py — biom3_build_dataset                  │
└─────────────────────┬─────────────────────────────────────┘
                      │
┌─────────────────────┴─────────────────────────────────────┐
│ Layer 3: Source-CSV builders + enrichment                 │
│   build_source_{swissprot,pfam,trembl,expasy,smart,brenda}│
│   build_annotated_pfam_subsets.py                         │
│   build_annotation_cache.py                               │
│   convert.py                                              │
│   enrich.py                                               │
└─────────────────────┬─────────────────────────────────────┘
                      │
┌─────────────────────┴─────────────────────────────────────┐
│ Layer 2: Filtered/indexed readers                         │
│   swissprot.py · pfam.py · taxonomy.py                    │
└─────────────────────┬─────────────────────────────────────┘
                      │
┌─────────────────────┴─────────────────────────────────────┐
│ Layer 1: Raw-database parsers                             │
│   swissprot_dat.py · pfam_metadata.py                     │
│   expasy.py · smart.py · brenda.py                        │
│   uniprot_client.py (legacy REST API)                     │
└───────────────────────────────────────────────────────────┘

Cross-cutting:
  base.py · config.py · caption.py · stats.py · __main__.py · __init__.py
```

---

## Layer 0 — infrastructure & utilities

Small files that everything else depends on.

### [`__init__.py`](../src/biom3/dbio/__init__.py) (0 lines)

Empty. Follows the repo convention that subpackage `__init__.py` files
don't re-export.

### [`__main__.py`](../src/biom3/dbio/__main__.py) (90 lines)

Pure dispatch table for the CLI entry points registered in
`pyproject.toml`. Each `run_*()` function lazily imports the relevant
builder module's `parse_arguments` + `main` and runs them. Lazy imports
keep startup cheap when only one CLI is being invoked. Nine entry
points are wired here:

- `run_build_dataset` → `biom3_build_dataset`
- `run_build_taxid_index` → `biom3_build_taxid_index` (inlined rather
  than delegating)
- `run_convert_to_parquet` → `biom3_convert_to_parquet`
- `run_build_source_{swissprot,pfam,trembl,expasy,smart,brenda}` —
  six per-database source-CSV builders
- `run_build_annotated_pfam_subsets` → `biom3_build_annotated_pfam_subsets`
- `run_build_annotation_cache` → `biom3_build_annotation_cache`

### [`base.py`](../src/biom3/dbio/base.py) (26 lines)

Defines the `DatabaseReader` ABC: `name` property + abstract
`query_by_pfam(pfam_ids, **kwargs)`. Provides extensibility for future
readers (PDB / SCOPe / CATH per
[database_linkage.md](database_linkage.md)). Currently only
`SwissProtReader` and `PfamReader` subclass it; the other parsers
(BRENDA, ExPASy, SMART) don't because they're consumed via the join
layer rather than `query_by_pfam`.

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

---

## Layer 1 — raw-database parsers (one per source format)

Pure parsers — they read a file and yield structured records. Zero
pandas, zero IO orchestration, zero opinions about output shape.

### [`swissprot_dat.py`](../src/biom3/dbio/swissprot_dat.py) (628 lines)

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

### [`pfam_metadata.py`](../src/biom3/dbio/pfam_metadata.py) (168 lines)

Parses `Pfam-A.full.gz` (Stockholm) or `Pfam-A.hmm.gz` for **family-level
metadata only** — not per-protein hits. Returns a `PF_ID → {short_id,
family_name, family_description, family_type, family_clan,
family_wikipedia, family_references}` dict. Stockholm is preferred
because it carries `#=GF CC` (description) and `#=GF TP/CL/WK/RT` lines
that the HMM file lacks. The four extras (`type`, `clan`, `wikipedia`,
`references`) were added in commit `dc911ce` to feed
`biom3_build_annotated_pfam_subsets`.

### [`expasy.py`](../src/biom3/dbio/expasy.py) (148 lines)

Streaming parser for `enzyme.dat`. Yields `EnzymeEntry` dataclass
instances with `ec`, `name`, `alternative_names`, `catalytic_activities`,
`cofactors`, `comments`, `uniprot_accessions`, `transferred_to`,
`deleted`. Handles section-delimited `//` records and the 2-character
line codes (`ID`, `DE`, `AN`, `CA`, `CF`, `CC`, `DR`). Detects
"Transferred entry: X.Y.Z.W" / "Deleted entry" markers in the DE
field.

### [`smart.py`](../src/biom3/dbio/smart.py) (44 lines)

Trivial TSV reader for `SMART_domains.txt`. `SmartReader.iter_domains()`
yields dicts with `domain_name`, `accession`, `definition`,
`description`. Skips header line and dashed separator. Smallest parser
in the package.

### [`brenda.py`](../src/biom3/dbio/brenda.py) (241 lines)

Parser for the BRENDA flatfile. Section-delimited per EC, with the most
complex format in dbio (40+ section codes, per-organism record fanout
via `#N#` / `#N,M,…#` tags, reference tail-tags `<3,7,12>`). Currently
captures only the high-value sections in `TRACKED_CODES = {"PR", "RN",
"SN", "SY", "RE", "SP", "KM", "PHO", "TO"}` — protein,
recommended/systematic name, synonyms, reaction, substrate-product,
KM, pH optimum, temperature optimum. Yields `BrendaEntry` with
per-organism dicts for the kinetic fields.

### [`uniprot_client.py`](../src/biom3/dbio/uniprot_client.py) (172 lines)

**REST API client** for UniProt — the legacy enrichment path before
local `.dat` parsing existed. Two classes:

- `UniProtCache` — disk-backed JSON cache keyed by accession.
- `UniProtClient` — batch fetcher with retry/backoff (batch=25,
  retries=5, delay=0.5s) for the 429/5xx storms the UniProt search
  endpoint hands out.

Mostly superseded by `swissprot_dat.py` + the annotation Parquet
cache, but kept as a fallback for accessions that aren't in any local
`.dat` (rare TrEMBL holdouts, recent additions).

---

## Layer 2 — filtered/indexed readers

`DatabaseReader` subclasses that consume the **output** CSVs of the
source builders, not the raw databases.

### [`swissprot.py`](../src/biom3/dbio/swissprot.py) (84 lines)

`SwissProtReader.query_by_pfam(pfam_ids)`. Loads the entire
`fully_annotated_swiss_prot.csv` (~570K rows) into memory once, then
regex-matches `pfam_label` (which stores stringified Python lists).
Auto-detects a Parquet sibling and prefers it.
**`OPTIONAL_OUTPUT_COLS = ["annot_ec_numbers"]`** is the flag that
lets newer 5-column CSVs flow through `query_by_pfam` while older
4-column CSVs still load — this is what makes the `--no_emit_ec_numbers`
toggle (commit `46106f3`) backward compatible.

### [`pfam.py`](../src/biom3/dbio/pfam.py) (124 lines)

`PfamReader.query_by_pfam(pfam_ids, keep_family_cols=False)`. Reads
the much-larger `Pfam_protein_text_dataset.csv` (~63M rows). Two
paths:

- **Parquet**: `pq.read_table(filters=[("pfam_label", "in", pfam_ids)])`
  — instant predicate pushdown.
- **CSV**: chunked read with `chunksize=500_000` and tqdm progress
  bar. Slow but works without the Parquet conversion.

The `keep_family_cols` flag is for the enrichment pipeline, which
needs `family_name`/`family_description` to compose Pfam captions.

### [`taxonomy.py`](../src/biom3/dbio/taxonomy.py) (282 lines)

NCBI taxonomy. Two classes:

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

## Layer 3 — source-CSV builders + enrichment

Per-database CLI builders. Each writes a CSV plus
`<stem>.build_manifest.json` (provenance) and `<stem>.stats.md`
(coverage). All emit ALL-CAPS-prefixed `[final]text_caption` via
`caption.py`.

### [`build_source_swissprot.py`](../src/biom3/dbio/build_source_swissprot.py) (399 lines)

Foundational builder #1. Streams `uniprot_sprot.dat.gz` via
`SwissProtDatParser.parse_all()`, joins family names from
`PfamMetadataParser`, composes captions via `SWISSPROT_SPEC`
(PubMed-stripped, ECO-stripped). Output schema is one of four:

| `--keep_intermediate_captions` | `--emit_ec_numbers` (default) | `--no_emit_ec_numbers` |
|---|---|---|
| **off** (default) | 5 cols (modern) | 4 cols (legacy parity) |
| **on** | 7 cols | 6 cols (full legacy parity) |

Commit `46106f3` (2026-05-05) added the EC-numbers toggle and the two
`_LEGACY` schema constants.

### [`build_source_pfam.py`](../src/biom3/dbio/build_source_pfam.py) (289 lines)

Foundational builder #2. Streams `Pfam-A.fasta.gz` (90%-NR, RP-scoped
since Pfam 37.1), parses the FASTA header (3 whitespace-separated
parts), joins family metadata from `Pfam-A.full.gz`, composes a
2-field caption via `PFAM_SPEC`
(`"Protein name: {family_name}. Family description: {family_description}"`,
no PubMed/ECO stripping). Always emits 8 columns. ~63M rows, ~52 GB
on Pfam 38.1.

### [`build_source_trembl.py`](../src/biom3/dbio/build_source_trembl.py) (339 lines)

Same schema as the SwissProt builder, but parses
`uniprot_trembl.dat.gz` (~250M entries). Adds `--evidence_filter
{strict,lenient,any}` because TrEMBL is unreviewed and the codes in
`AUTOMATIC_ECO_CODES = {"ECO:0000256", "ECO:0000313"}` (per the in-code
definitions) dominate the file. Default `lenient` drops entries whose
only evidence codes are those two. `--strict` requires
`ECO:0000269` (`EXPERIMENTAL_ECO_CODE`); `--any` keeps all rows.

### [`build_source_expasy.py`](../src/biom3/dbio/build_source_expasy.py) (203 lines)

Builds `expasy_enzyme.csv` from `enzyme.dat`. Per-row schema: EC,
name, reactions, cofactors, UniProt cross-refs. Has
`--include_obsolete` / `--exclude_obsolete` to keep or drop
"Transferred entry" / "Deleted entry" rows. ~8,441 rows on the current
ExPASy release.

### [`build_source_smart.py`](../src/biom3/dbio/build_source_smart.py) (126 lines)

Builds `smart_domains.csv` from `SMART_domains.txt`. ~1,405 rows, 100%
domain-name coverage, 85% definition coverage. Smallest builder.

### [`build_source_brenda.py`](../src/biom3/dbio/build_source_brenda.py) (260 lines)

Builds `brenda_kinetics.csv` — per-organism kinetic data joined from
each EC entry. Output is rows of `(ec, organism_id, organism_name,
recommended_name, …substrates, km_values, ph_optimum,
temperature_optimum)`. ~111,854 rows (6,910 ECs × ~16 organisms each).
Truncates oversized join cells via `_truncate_join` to keep CSV cells
under a char cap.

### [`build_annotated_pfam_subsets.py`](../src/biom3/dbio/build_annotated_pfam_subsets.py) (324 lines)

Streams `Pfam-A.full.gz` (the **non-NR** Stockholm alignment) directly,
extracts only requested families. Unlocks ~6.7× more rows per family
than `build_source_pfam` — PF00018 goes from 26,468 → 176,301 rows.
11-column schema with `family_type` / `family_clan` /
`family_wikipedia` / `family_references` as separate side fields
(deliberately not folded into the caption, so training-time
augmentation can subsample them independently). Single-pass streaming,
multi-family requests share one scan, early-exits when all targets
are found.

### Helpers

These produce derived performance artifacts (Parquet caches /
conversions) rather than primary training-data CSVs. They sit
alongside the source builders because they're CLI-driven and consume
the same raw inputs, but downstream consumers in Layer 4 only see them
indirectly — `build_dataset.py` uses `--annotation_cache` paths the
first one writes, and the readers in Layer 2 auto-detect Parquet
siblings the second one writes.

#### [`build_annotation_cache.py`](../src/biom3/dbio/build_annotation_cache.py) (204 lines)

Writes a Parquet annotation cache from a `.dat` file. Schema:
`primary_Accession` + 19 `annot_*` columns (sparse, `pa.string()`
nullable), sorted by accession per row group for fast
predicate-pushdown reads. The point is amortizing the multi-hour TrEMBL
`.dat` parse: build the cache once, then any subsequent
`biom3_build_dataset --annotation_cache` invocation gets instant
Pfam-row enrichment without re-streaming TrEMBL. Tightly coupled to
`SwissProtDatParser` and the `enrich.ANNOTATION_COLUMNS` schema.

#### [`convert.py`](../src/biom3/dbio/convert.py) (108 lines)

Generic CSV → Parquet converter (`biom3_convert_to_parquet`). Reads
the CSV in chunks, writes a single Parquet file with row groups for
partial reads. Database-agnostic at its core; auto-detects the Pfam
CSV by checking for `family_description` in the header and forces it
to `dtype=str` to silence pandas mixed-type warnings, but otherwise
will happily convert any CSV. Used to turn the 35 GB
`Pfam_protein_text_dataset.csv` into a 5–8 GB Parquet for the
`PfamReader` fast path.

### [`enrich.py`](../src/biom3/dbio/enrich.py) (738 lines)

The two-step enrichment pipeline that turns a thin `(accession,
sequence, pfam_label)` row into a fully-captioned training example.

- **Step 1: `enrich_dataframe(df, …)`** populates per-row `annot_*`
  columns from any combination of: UniProt JSON entries (legacy API
  path), local `.dat` parser output, the annotation Parquet cache,
  NCBI taxonomy lookup, and ExPASy/BRENDA/SMART source-CSV joins.
  Returns `(df, join_stats)` so callers can record hit rates in the
  manifest.
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

Helpers for parsing UniProt JSON (`parse_protein_name`,
`parse_gene_ontology`, …), EC-number extraction (`extract_ec_numbers`
with caption-text fallback), and the three private join helpers
(`_join_expasy`, `_join_brenda`, `_join_smart`) with hit-rate stats.

---

## Layer 4 — the dataset orchestrator

### [`build_dataset.py`](../src/biom3/dbio/build_dataset.py) (666 lines)

The CLI most users actually invoke (`biom3_build_dataset`). Stitches
everything above into a finetuning dataset:

1. Parse args; resolve SwissProt / Pfam / ExPASy / BRENDA / SMART CSV
   paths via `config.py`.
2. Subset SwissProt and Pfam by Pfam IDs (`SwissProtReader.query_by_pfam`
   + `PfamReader.query_by_pfam`).
3. Optionally enrich Pfam rows via `enrich.enrich_dataframe()` using
   the `--annotation_cache` → `--uniprot_dat` → API fallback chain.
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

## Cross-cutting design observations

### Strengths

- **Clean layering.** Parsers → readers → builders → orchestrator.
  Each layer can be tested independently and the seams are stable.
- **Provenance everywhere.** Every CSV gets a
  `<stem>.build_manifest.json` + `<stem>.stats.md` sidecar. The
  2026-04-16 legacy-vs-CLI audit workflow is the reason this exists,
  and it's load-bearing — see
  [training_csv_provenance.md](training_csv_provenance.md) for the
  column-level reference and the 2026-04-16 session note for the audit
  context.
- **Backward-compat surfaces are explicit.** `OPTIONAL_OUTPUT_COLS` in
  `swissprot.py`, the `_LEGACY` schema constants in
  `build_source_swissprot.py`, the `OPTIONAL_*` columns in the
  enrichment layer. Every "what schema is this CSV?" question has an
  answer.
- **Parquet as a sibling format.** Readers auto-detect `.parquet` next
  to a `.csv` and prefer it. Lets the Pfam path go from
  "chunked-CSV-with-tqdm" to "predicate-pushdown" without changing
  call sites.

### Tensions and open items

- **`uniprot_client.py` is mostly legacy.** The local `.dat` parser +
  annotation cache cover almost all enrichment now, but the API client
  is still wired into `build_dataset.py` as a fallback. Worth deciding
  whether to deprecate it or keep it for the rare TrEMBL holdouts.
- **`build_dataset.py` enriches `df_pfam` but not `df_sp`.** SwissProt
  rows skip the join layer because they come from a pre-built source
  CSV. Closing this gap (e.g., re-running `enrich_dataframe` on
  SwissProt rows keyed by `primary_Accession`) is the
  "df_sp enrichment in build_dataset" item still listed in the
  2026-04-18 session note.
- **Naming asymmetry between readers and parsers.** `swissprot.py`
  (reader for the *output* CSV) vs `swissprot_dat.py` (parser for the
  *raw* `.dat`) is occasionally confusing — same for `pfam.py`
  (reader) vs `pfam_metadata.py` (raw `.full` / `.hmm` parser) and
  `expasy.py` (raw `.dat` parser, no reader).
- **`enrich.py` is the biggest file (738 lines)** and mixes UniProt
  JSON parsing, EC extraction, taxonomy joining, ExPASy/BRENDA/SMART
  joining, and `compose_caption`. A natural seam exists between the
  JSON parsers and the join helpers — splitting them would make the
  file easier to navigate, but breaking the public API of
  `enrich_dataframe` would ripple.

### Test coverage

`tests/dbio_tests/` has 210+ tests across all of these (`test_imports`,
`test_swissprot`, `test_pfam`, `test_taxonomy`, `test_enrich`,
`test_build_dataset`, `test_build_source_swissprot`,
`test_build_source_pfam`, `test_build_source_trembl`,
`test_pfam_metadata`, `test_caption`, `test_build_annotation_cache`,
`test_build_annotated_pfam_subsets`, `test_ec_extraction`,
`test_expasy_parser`, `test_smart`, `test_brenda_parser`,
`test_stats`). Coverage is thorough on the parser layer and the
builder schemas, lighter on the orchestrator's branchier code paths.

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
