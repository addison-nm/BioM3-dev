# 2026-04-18 — `biom3.dbio` expansion for Stage 3 FT datasets

## Goal

Enable rich Stage 3 finetuning datasets built from multiple databases
(SwissProt, TrEMBL, Pfam, NCBI Taxonomy + new: ExPASy, SMART, BRENDA)
with cross-DB annotation enrichment, per-Pfam output modes, and
coverage stats for every output.

Plan lives at `~/.claude/plans/let-s-carefully-plan-this-jazzy-walrus.md`.

## Branch

`addison-dbio-ft-datasets` (worktree at
`.claude/worktrees/dbio-ft-dataset-expansion/`), pushed to origin. 13
commits ahead of `addison-dev`.

## What shipped

### Phase 0 — foundation
- [`docs/database_linkage.md`](../../docs/database_linkage.md):
  authoritative cross-reference spec. Per-DB sections, identifier
  families, cross-reference cheat sheet, canonical CSV contract
  (`annot_*` = ground truth, `[final]text_caption` = convenience),
  planned DBs (PDB / SCOPe / CATH).

### Stats infrastructure
- [`src/biom3/dbio/stats.py`](../../src/biom3/dbio/stats.py):
  `compute_coverage_stats()`, `IncrementalStatsBuilder` (for streaming
  builders), `format_stats_markdown()` + `write_stats_markdown()`.
- [`src/biom3/core/run_utils.py`](../../src/biom3/core/run_utils.py):
  `write_manifest()` gained a `stats` kwarg.
- Every source builder and `build_dataset` emits `<stem>.stats.md`
  alongside the CSV and embeds the same dict under `manifest.stats`.

### Source builders (Phases 1-4)
| Builder | Rows | Coverage highlights |
|---------|------|---------------------|
| TrEMBL (`build_source_trembl.py`) | smoke: 50 with `--limit` | `--evidence_filter {strict,lenient,any}`, default `lenient` drops pure-automatic ECO:0000256/0000313 |
| ExPASy (`build_source_expasy.py`) | 8,441 | 100% name, 82% reactions, 68% UniProt xrefs (30/EC avg, max 3,980) |
| SMART (`build_source_smart.py`) | 1,405 | 100% name, 85% definition |
| BRENDA (`build_source_brenda.py`) | 111,854 (6,910 ECs × 55K organisms) | 100% name+reactions, 81% substrates, 29–33% kinetics, 483 MB CSV |
- SwissProt + Pfam builders retrofitted to emit stats (commit `36d1cb0`).
- CLI entry points: `biom3_build_source_{trembl,expasy,smart,brenda}`.

### Phase 5 — enrichment join layer
- [`SwissProtDatParser`](../../src/biom3/dbio/swissprot_dat.py) now
  captures `DR SMART;`, `DR InterPro;`, `DR PDB;` under
  `xref_smart_ids` / `xref_interpro_ids` / `xref_pdb_ids` side-channel
  keys (flow through `enrich_dataframe` into DataFrame columns).
- [`enrich.py`](../../src/biom3/dbio/enrich.py): `extract_ec_numbers()`,
  `load_expasy_lookup()`, `load_brenda_lookup()`, `load_smart_lookup()`,
  and three private join helpers `_join_expasy` / `_join_brenda` /
  `_join_smart` with hit-rate reporting.
- `enrich_dataframe()` now returns `(df, join_stats)`; 2 tests updated.
- `ANNOTATION_FIELDS` extended with 7 new caption fields (EC, SMART,
  BRENDA substrates/KM/pH/temp), ordered at the end so legacy
  captions don't shift.
- `build_dataset` flags: `--use_expasy --expasy_csv`, `--use_brenda
  --brenda_csv --organism_match {strict,relaxed,ec_only}`, `--use_smart
  --smart_csv`. All opt-in.
- Smoke: PF00018 with `--use_expasy --use_smart` → 17,165 rows in
  2m16s, join stats in manifest, 0% hit rates as expected without
  `--uniprot_dat` to populate `annot_catalytic_activity`.

### Phase 6 — per-Pfam output + stats fix
- `--per_pfam_output` flag: emits one self-contained subdirectory per
  Pfam ID (dataset.csv, dataset_annotations.csv, build_manifest.json,
  dataset.stats.md, pfam_ids.csv). **Mutually exclusive with aggregate
  output** — when set, NO top-level dataset is written, only the
  subdirectories + build.log.
- Top-level `dataset.stats.md` always emitted in aggregate mode, with
  source-breakdown (swissprot vs pfam rows) and join hit rates.
- [`_row_has_pfam()`](../../src/biom3/dbio/build_dataset.py) helper
  uses `PF\d{5}` regex on `pfam_label` — robust to list-stringified
  and bare formats.
- Fix: `pd.NA` stringifies to `"<NA>"`; added to `_EMPTY_SENTINELS`
  in `stats.py` so coverage percentages are accurate on pandas-mixed
  DataFrames (commit `4e6852d`).
- Smoke: `--pfam_ids PF00018 PF00169 --per_pfam_output` in 2m18s →
  clean per-Pfam subdir structure verified.

## Bugs caught and fixed

1. `pd.NA` in boolean context (`extract_ec_numbers`): reorder checks
   + try/except around `pd.isna()`. Commit included in `5ac63af`.
2. `_truncate_join` in BRENDA builder: single record > char_cap slipped
   through. Fixed to truncate mid-string when budget exhausted.
3. `pd.NA` stringified to `"<NA>"` not matching empty sentinels in
   stats.py. Separate `fix(dbio)` commit `4e6852d` for clean bisect.

## Verified

All 119 dbio tests pass throughout. End-to-end smoke tests run on
legacy SwissProt + Pfam CSVs (symlinked from
`/data/data-share/BioM3-data-share/data/datasets/LEGACY_*.csv` since
the non-legacy paths were overwritten per earlier project memory).

## Phase 7 — deferred / closed

Original concern (independent subprocess builds re-streaming the 35 GB
Pfam CSV per Pfam ID) is already handled by the existing architecture:
`query_by_pfam(ids)` filters all requested IDs in a single read pass,
enrichment joins run once on the combined DataFrame, and per-Pfam
output sequentially writes subdirectories (~1 sec/Pfam after the
initial scan). The real throughput win is the Parquet path (next
item), not multiprocessing.

## Additional work (since initial writeup)

- **Config resolvers done.** `configs/dbio_config.json` gained
  `training_datasets.{brenda,expasy,smart,trembl}_csv` entries plus
  `databases.{brenda,trembl}` entries. `build_dataset.py` has
  `_resolve_source_csv()` so `--expasy_csv` / `--brenda_csv` /
  `--smart_csv` fall back to the config-resolved path when not
  supplied. Verified end-to-end: `--use_expasy --use_smart` with no
  `--*_csv` flags runs against Parquet sources in 5s.
- **Unit tests added (76 new; 195 total in dbio).** Covers:
  ExPASy parser (8), SMART reader (6), BRENDA parser (9), EC
  extraction + ExPASy/BRENDA/SMART joins via `enrich_dataframe` (18),
  stats module including IncrementalStatsBuilder (35). All green.
- **Verification pass with --uniprot_dat on PF00128 (alpha-amylase).**
  Ran end-to-end (18–34s depending on whether .dat parsing was
  triggered); hit rates were 0% because:
  - SwissProt source CSV only has the legacy 4-column schema (no
    `annot_catalytic_activity`), so the 608 SwissProt rows can't be
    EC-extracted directly.
  - Pfam rows need local annotation via `--uniprot_dat`. Only
    2/38,849 Pfam accessions were in `uniprot_sprot.dat.gz` (most
    are TrEMBL).
  - Legacy SwissProt captions have `CATALYTIC ACTIVITY:` describing
    reactions in prose but with EC-number xrefs stripped, so the
    caption fallback added during this session doesn't help those
    specific rows either.

  Fixes landed during the verification pass (commit `f7af1f6`):
  - **pd.NA hardening** in `enrich_dataframe`,
    `_strip_lineage_prefix`, `_row_organism_candidates`, `_row_smart`.
    Factored `_is_missing()` helper; replaced ad-hoc checks.
  - **Object-column pre-creation** in `enrich_dataframe` so list
    values like `xref_smart_ids` can be stored in DataFrame cells
    via `df.at[idx, col] = [...]` without pandas raising.
  - **Caption fallback for EC extraction** in ExPASy/BRENDA joins
    (`_extract_row_ec_numbers` tries `annot_catalytic_activity`
    first, then `[final]text_caption`). Future-proof for source CSVs
    that carry EC in the caption.

  Real hit rates are deterministically validated by the integration
  tests (`test_expasy_join_hit_rate_stats`: 3/4 EC extraction, 2/4
  ExPASy matches). The 0% end-to-end reflects the legacy CSV's data
  shape, not a code defect.

## Open items (remaining)

- **EC preservation in SwissProt source builder.** The current
  `_map_cc_catalytic_activity` in `swissprot_dat.py` drops EC xrefs
  during composition. If the source CSV preserved EC numbers in
  `annot_catalytic_activity`, the ExPASy/BRENDA joins would hit
  natively on SwissProt rows without needing `--uniprot_dat`.
- **df_sp enrichment in build_dataset.** Today only df_pfam flows
  through `enrich_dataframe`. SwissProt rows skip the join layer
  because they come from the pre-built source CSV with just 4
  columns. Extending enrichment to df_sp would require re-parsing
  `uniprot_sprot.dat.gz` for those accessions — a separate design
  decision.
- **TrEMBL annotation cache for fast Pfam-row enrichment.** Large
  Pfam families are TrEMBL-dominated; populating
  `annot_catalytic_activity` for them requires either
  `--uniprot_dat <trembl.dat.gz>` (hours) or the existing
  `biom3_build_annotation_cache` Parquet path (instant lookup once
  built).

## Commits

```
890e702 refactor(dbio): --per_pfam_output skips aggregate output entirely
c3b19c5 feat(dbio): --per_pfam_output mode + dataset.stats.md for build_dataset
4e6852d fix(dbio): treat pandas <NA> string as empty in stats coverage
e7addba feat(dbio): wire source-CSV enrichment flags into build_dataset
5ac63af feat(dbio): ExPASy/BRENDA/SMART join layer in enrich.py
9cd5585 feat(dbio): capture SMART/InterPro/PDB cross-refs in .dat parser
10d3380 chore(dbio): register new source-builder CLI entry points
49b1eac feat(dbio): add BRENDA source builder with per-organism kinetics
f537095 feat(dbio): add ExPASy and SMART source builders
36d1cb0 refactor(dbio): emit .stats.md from SwissProt + Pfam builders
6fd4ee1 feat(dbio): add TrEMBL source builder with evidence filtering
3cd4f10 docs(dbio): add database_linkage.md cross-reference spec
3fbc098 feat(dbio): add stats module for coverage reports
```
