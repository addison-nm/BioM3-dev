# 2026-05-05 — optional `annot_ec_numbers` column in SwissProt source builder

## Context

`biom3_build_source_swissprot` has unconditionally emitted the structured
`annot_ec_numbers` column since 2026-04-18 (commit `468dcd7`), giving the
ExPASy/BRENDA enrichment joins a clean source of EC numbers without
re-parsing `.dat` files at join time. That default is correct for modern
pipelines, but it leaves no single-flag way to produce a CSV byte-compatible
with the legacy 4/6-column SwissProt schema. The legacy-vs-CLI audit
workflow (per the 2026-04-16 session) currently has to drop the column
post-hoc to diff cleanly against `LEGACY_fully_annotated_swiss_prot.csv`.

This session adds an opt-in toggle to drop `annot_ec_numbers` at build
time, mirroring the existing `--require_pfam` / `--keep_intermediate_captions`
legacy-parity axes.

## Changes

### `src/biom3/dbio/build_source_swissprot.py`

- Added two new schema constants:
  - `OUTPUT_COLUMNS_LEGACY` — 4-column legacy schema (`primary_Accession`,
    `protein_sequence`, `[final]text_caption`, `pfam_label`).
  - `OUTPUT_COLUMNS_WITH_INTERMEDIATES_LEGACY` — 6-column legacy schema
    (the four above plus `text_caption` / `[clean]text_caption`).
- `build_swissprot_csv()` gained an `emit_ec_numbers=True` keyword. When
  `False`, the per-row append of `annot_ec_numbers` is skipped and the
  schema constant selection drops down to the `_LEGACY` variant in both
  the default and `--keep_intermediate_captions` paths.
- New mutually-exclusive CLI flag pair `--emit_ec_numbers` /
  `--no_emit_ec_numbers`, defaulting to emit. Help text on the negative
  flag warns explicitly that ExPASy/BRENDA EC-based joins fall back to
  caption-text extraction (typically 0% hit rate) when the column is
  dropped — surfacing the silent-degradation risk discussed during the
  design conversation.
- `main()` threads `args.emit_ec_numbers` into `build_swissprot_csv()`.

### `tests/dbio_tests/test_build_source_swissprot.py`

New `TestEmitEcNumbersToggle` class with three tests:

- 4-column header when `emit_ec_numbers=False` and intermediate captions
  are off.
- 6-column header when `emit_ec_numbers=False` and
  `keep_intermediate_captions=True`.
- Per-row CSV widths equal the header width (no off-by-one when the
  column is dropped).

All 31 SwissProt-builder tests pass under
`conda run -n biom3-env pytest tests/dbio_tests/test_build_source_swissprot.py`
(28 prior + 3 new).

### `docs/training_csv_provenance.md`

Documented the new flag pair under the existing `annot_ec_numbers` note
in the SwissProt section, including the legacy-parity use case and the
explicit warning that disabling the column collapses EC join hit rates
to ~0% (since `annot_catalytic_activity` strips EC xrefs in this
builder's output, exactly as the legacy CSV did).

## Backward compatibility

- Default invocations (no new flags) produce the same 5-column /
  7-column output as before. No behavior change for existing pipelines.
- `swissprot.py:OPTIONAL_OUTPUT_COLS = ["annot_ec_numbers"]` already
  handles absence in `query_by_pfam`, so reading a
  `--no_emit_ec_numbers` CSV needs no additional changes downstream.
- `_extract_row_ec_numbers` precedence (`annot_ec_numbers` →
  `annot_catalytic_activity` → `[final]text_caption`) means joins remain
  well-defined when the column is missing — they just hit at 0% on
  legacy-formatted SwissProt rows, which is the deterministic and
  documented behavior.

## Why opt-in (not opt-out)

The modern path keeps EC-based enrichment joins working out-of-the-box;
defaulting `emit_ec_numbers=False` would push every workspace into
needing to remember the flag. The asymmetry with `--require_pfam`
(which defaults to legacy parity) is intentional: skipping rows is a
filter the legacy CSV applied implicitly, whereas the EC column is a
strict superset of the legacy schema that adds no rows or row-level
divergence.
