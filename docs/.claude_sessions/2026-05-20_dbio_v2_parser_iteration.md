# 2026-05-20 — dbio_v2 parser: JSON emission, structured DE, predict-and-fix loop

Continuation of the `dbio_v2` parser refactor on branch `dbio-refactor`.
Initial scaffold + UniProt parser + 8 hand-curated expectations had landed
in commit `9e65bc0`; this session iterated the parser, added the JSON
emission feature, and grew the expectation suite to 40 entries through a
hand-prediction-and-verify loop.

## What landed

### Parser features (`src/biom3/dbio_v2/parsers/uniprot_dat.py`)

- **Structured `description` (Shape C)** — `parse_description` parses the
  DE block grammar from userman.html into a typed dict
  `{rec_name, sub_name, alt_names, includes, contains, flags}` with
  recursive `Includes:`/`Contains:` NameBlock nesting. Polymorphic
  AltName variants (`Allergen=` / `Biotech=` / `CD_antigen=` / `INN=`)
  carry a single variant-keyed value plus `evidence`. Standard NameBlocks
  carry `full / short / ec_numbers / evidence`.
- **`_join_wrap` helper** — UniProt wraps multi-line continuations
  mid-word on hyphens with no joining space (e.g. `'Ser-` + `241'` →
  `'Ser-241'`, not `'Ser- 241'`). One helper applied at every multi-line
  join site: `_flush_cc`, `_h_ref` (RP/RC/RX/RG/RA/RT/RL), `_h_ft`
  qualifier continuations, and `parse_entry` finalize for `OS`.
- **`cross_references` raw-string shape** — `_h_dr` now stores each DR
  line verbatim (cols 6+, trailing `.` and any optional `[P12345-N]`
  isoform tag intact), grouped by database name as the dict key:
  `{"WormBase": ["WormBase; F26C11.2; CE01560; WBGene…; unc-4."]}`.
  Replaces the earlier nested-list shape; `_DR_ISOFORM_RE` removed.
- **`gene_names` split on `;`** — `_h_gn` now emits one element per
  `Key=value` sub-token (e.g. `Name=…; Synonyms=…; OrderedLocusNames=…`
  → three list entries), consistent with how `_h_kw` handles `KW`.
- **Trailing-terminator policy: strip `;`, keep `.`** — semicolons (the
  UniProt field-terminator) are stripped from `gene_names`, references
  (`RC`/`RG`/`RX`/`RA`/`RT`), DE-block per-name values, and `Flags`
  tokens. Periods are preserved everywhere (often part of the data, e.g.
  abbreviation initials like `Essani K.` in `RA`).

### JSON emission (`src/biom3/dbio_v2/parsers/base.py`)

- **`Record.to_json(*, indent=None, ensure_ascii=False, sort_keys=False)`**
  — serialize a record to a JSON string. Compact one-line default for
  JSONL streaming; `indent=2` for pretty-print. Mirrors `as_dict` —
  `record_type` is a ClassVar tag and is not emitted into the payload.
- **`dump_jsonl(records, stream, *, ensure_ascii=False) -> int`**
  — module-level helper that writes a stream of `Record` instances as
  JSONL (one compact JSON object per line). Lazy over any
  `Iterable[Record]`, returns the count written.
- Both re-exported from `biom3.dbio_v2.parsers`.

### Test suite

- `tests/dbio_v2_tests/parsers/test_uniprot_dat.py` — added
  `TestJsonExpectations` (parametrized per-entry comparison against
  hand-curated JSONs under
  `tests/_data/dbio_v2/parser_expectations/uniprot_sprot_mini/`) and
  `TestParseDescriptionUnit` (focused DE-grammar unit tests).
- `tests/dbio_v2_tests/parsers/test_base.py` — added `TestJsonEmission`
  (round-trip, compact/pretty, JSONL writer).
- **Test count grew from 38 (post-scaffold) → 130 today.**
- 40 hand-curated `sprot_exp_*.json` expectation files (entries 0–39).

## Methodology: predict-and-fix loop

For entries beyond the initial scaffold's 8, the workflow was:

1. Read the kth source entry from `tests/_data/dbio_v2/uniprot_sprot_mini.dat`.
2. **Hand-derive** the expected JSON from the documented rules (NOT by
   invoking the parser under test).
3. Run the parametrized expectation test for `idx=k`.
4. On mismatch, diagnose whether the parser or the prediction is wrong
   and fix accordingly.

This loop surfaced ~5 real parser refinements (RC/RG strip, GN split,
DR raw-string policy, hyphen-wrap rule, ECO aggregation choice) and
caught several hand-prediction errors (off-by-one terminator strips,
inconsistent hyphen-wrap treatment). Across entries 12–39 the parser
required zero changes for ~22 of 28 entries — good evidence the rules
are well-codified.

## Key design decisions (all confirmed with the user this session)

| Decision | Choice |
|---|---|
| **Trailing terminator policy** | Strip `;` (field delimiter); keep `.` (often part of data) |
| **DR storage** | Raw line strings per `;`-key, not nested-list of fields |
| **DE description shape** | Structured Shape C dict (recursive, polymorphic AltNames) |
| **Hyphen-wrap on continuation** | No joining space when prev chunk ends with `-` (Ser-241 model) |
| **ECO aggregation in NameBlock** | Option 1: flat deduped list of every ECO across Full/Short/EC sub-fields |
| **Reference fields stripping `;`** | RC, RG, RX, RA, RT (all five UniProt R-codes that terminate with `;`) |

## Verification

- `pytest tests/dbio_v2_tests -q` → **130 passed**.
- `pytest tests/test_imports.py -q` → 5/5 passed, no regression from
  pre-session state.
- All 40 expectation files match the parser via both `as_dict` equality
  and `to_json` JSON round-trip.

## Files modified / created

Modified (since `9e65bc0`):
- `src/biom3/dbio_v2/parsers/base.py` — JSON emission helpers
- `src/biom3/dbio_v2/parsers/__init__.py` — re-export `dump_jsonl`
- `src/biom3/dbio_v2/parsers/uniprot_dat.py` — `parse_description`,
  `_join_wrap`, raw-string DR, GN split, terminator strips
- `tests/dbio_v2_tests/parsers/test_base.py` — `TestJsonEmission`
- `tests/dbio_v2_tests/parsers/test_uniprot_dat.py` — parametrized
  expectations, new unit tests, updated assertions
- 8 existing expectation JSONs (0–7) refreshed for the new shapes

Created:
- 32 new expectation JSONs (8–39) under
  `tests/_data/dbio_v2/parser_expectations/uniprot_sprot_mini/`

## What's still on deck

- Remaining four parsers: `pfam_stockholm`, `expasy_enzyme`,
  `brenda_flatfile`, `smart_tsv` — each gets its own `<Db>Record(Record)`
  annotated subclass, `iter_records`, fail-loud reads, and an analogous
  expectation suite under `tests/_data/dbio_v2/parser_expectations/`.
- Builder rewire — once all five parsers are in, replace the legacy
  `biom3.dbio` builders / `enrich.py` / `annotation_cache.py` consumers
  with calls into `biom3.dbio_v2`. The handoff explicitly scoped this
  as a later, separate pass.
- (Optional) Extend the predict-and-fix loop to the TrEMBL mini fixture
  to validate `Unreviewed` / `SubName:` / abundant `ECO:0000256/0313`
  patterns against hand-curated expectations.
