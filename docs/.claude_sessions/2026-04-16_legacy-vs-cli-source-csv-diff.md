# Session: Legacy vs CLI source CSV diff

**Date:** 2026-04-16
**Branch:** `worktree-legacy-vs-cli-source-csv-diff`

## Context

A sibling project (`~/Projects/BioM3-projects/dataset-construction/`)
regenerated both foundational source CSVs —
`fully_annotated_swiss_prot.csv` and `Pfam_protein_text_dataset.csv` —
from the current raw databases using `biom3_build_source_swissprot` and
`biom3_build_source_pfam`, and compared them against the legacy reference
files in `/data/data-share/BioM3-data-share/data/datasets/LEGACY_*`.
All figures below were verified with `wc -l`, `head`, `tail`, and `grep`
on the actual files; they are reproduced verbatim from the sibling
project's report.

## Findings

### Swiss-Prot (`fully_annotated_swiss_prot.csv`)

| | LEGACY | CLI-generated |
|-|--------|---------------|
| Rows | 569,517 | 547,273 |
| Columns | 6 | 4 |
| Ordering | alphabetical by accession | `.dat` file order |
| Entries with `['nan']` pfam_label | 31,144 | 0 |

Three material differences between legacy and CLI output:

1. **`require_pfam=True` is hard-wired.** The call site is
   [`src/biom3/dbio/build_source_swissprot.py:96`](../../src/biom3/dbio/build_source_swissprot.py#L96)
   (`parser.parse_all(require_pfam=True)`). The legacy CSV kept 31,144
   Swiss-Prot entries with no Pfam cross-reference and stamped them
   `pfam_label = ['nan']`; the CLI silently drops them. The net row-count
   delta reconciles as `-31,144 (filtered) + ~8,900 (new entries since
   LEGACY was built) = -22,244`. Recovering legacy parity required a code
   change — there was no CLI flag to disable the filter.

2. **Intermediate caption columns removed.** The legacy CSV carried
   `text_caption` (raw, with `(PubMed:NNN)` refs and `{ECO:...}` tags),
   `[clean]text_caption` (stripped), and `[final]text_caption`. The CLI
   emits only `[final]text_caption`, which loses the provenance needed to
   audit exactly what the PubMed and ECO-tag stripping passes removed.

3. **Row order differs.** Legacy is alphabetically sorted by accession
   post-build (`A0A009IHW8` → `X6R8R1`); CLI writes in `.dat` file order
   (`Q6GZX4` → `A9JR22`, i.e. Swiss-Prot's internal entry order). Any
   downstream code doing positional head/tail sampling or deriving
   train/val splits from row indices will see different data from the two
   files even when the entry sets overlap.

### Pfam (`Pfam_protein_text_dataset.csv`)

| | LEGACY | CLI-generated |
|-|--------|---------------|
| Rows | 44,767,155 | 63,237,515 |
| Columns | 8 (identical schema) | 8 |
| Max Pfam ID observed | `PF17xxx` range | `PF26733` |

The ~41% row-count jump is **a Pfam release version difference, not a
builder difference.** Legacy Pfam IDs top out in the PF10xxx–PF17xxx
range; the fresh build has `PF26733` — consistent with Pfam v37+, which
picked up thousands of new families and many additional sequence hits
per family. The caption format is byte-identical (lowercase
`Protein name:` / `Family description:` labels). No builder-side change
would close this gap; it requires rebuilding from the same DB release the
legacy file was cut from.

## Recommendations

1. **Add a `--require_pfam` / `--no_require_pfam` flag** to
   `biom3_build_source_swissprot`. The default should flip to `False`
   (legacy parity): Pfam-less entries are kept with
   `pfam_label=['nan']`. `--require_pfam` becomes an opt-in filter for
   downstream code that can't handle the sentinel. See
   [`build_source_swissprot.py:96`](../../src/biom3/dbio/build_source_swissprot.py#L96).

2. **Add a `--keep_intermediate_captions` flag** that also emits
   `text_caption` and `[clean]text_caption` alongside `[final]text_caption`,
   so reviewers can audit what the PubMed/ECO stripping removes without
   re-running the full build.

3. **Document the row-order difference** in
   [`docs/building_datasets_with_dbio.md`](../building_datasets_with_dbio.md)
   so downstream code that relies on positional slicing isn't silently
   broken when the CSV is regenerated.

4. **Capture DB release version in a build manifest** so each CSV carries
   a manifest next to it recording the UniProt/Pfam release version and
   source file sizes/mtimes. Today there is no trace in the output of
   which DB release produced the file; reconciling `(LEGACY, CLI)` row
   counts required cross-referencing file mtimes by hand. The manifest
   filename should be disambiguated per output (not a single shared
   `build_manifest.json`) so back-to-back source builds into the same
   directory don't clobber each other's provenance.

## Implementation notes

All four recommendations were implemented in this same session. The
changes are uncommitted in the `worktree-legacy-vs-cli-source-csv-diff`
worktree:

- [`src/biom3/dbio/build_source_swissprot.py`](../../src/biom3/dbio/build_source_swissprot.py)
  grew a `--require_pfam`/`--no_require_pfam` mutually-exclusive flag pair
  (default `--no_require_pfam` for legacy parity — Pfam-less entries are
  kept and stamped `pfam_label=['nan']`), a `--keep_intermediate_captions`
  flag (off by default, switches the output to the legacy 6-column
  schema), and writes a build manifest via
  `biom3.core.run_utils.write_manifest`.
- [`src/biom3/dbio/build_source_pfam.py`](../../src/biom3/dbio/build_source_pfam.py)
  also writes a manifest with Pfam FASTA + metadata file provenance and
  Pfam release version (read from `relnotes.txt`).
- **Manifest naming:** both builders write
  `<output_stem>.build_manifest.json` next to the output CSV (e.g.
  `fully_annotated_swiss_prot.build_manifest.json`), not a shared
  `build_manifest.json`. This lets multiple source CSVs coexist in one
  `datasets/` directory without overwriting each other's provenance.
- [`docs/building_datasets_with_dbio.md`](../building_datasets_with_dbio.md)
  gained a "Row ordering" note and a "Provenance" note under the
  rebuild-from-raw section, plus docs for the new flags. Also corrected
  an earlier inaccurate claim that the two source CSVs are "normally
  distributed pre-built via `BioM3-data-share`" — they aren't guaranteed
  to be.
- [`tests/dbio_tests/test_build_source_swissprot.py`](../../tests/dbio_tests/test_build_source_swissprot.py)
  and
  [`tests/dbio_tests/test_build_source_pfam.py`](../../tests/dbio_tests/test_build_source_pfam.py)
  cover the new flag behavior, the manifest write, and the stem-based
  filename. All 38 tests pass under `conda run -n biom3-env pytest`.

### `['nan']` format verified against the legacy file

The `pfam_label` sentinel format was verified by reading
`/data/data-share/BioM3-data-share/data/datasets/LEGACY_fully_annotated_swiss_prot.csv`
directly: 31,144 rows match the literal bare string `['nan']` (no CSV
quoting, since the cell contains no comma or double-quote). This is
exactly what `repr(["nan"])` emits from `_format_pfam_label`. Multi-entry
lists like `['PF03639', 'PF00949']` get CSV-quoted automatically because
of the embedded comma — also matching `repr(pfam_ids)` output.

The row counts in the tables above were not re-derived in this session;
the original databases are too large to re-verify in-session and the
figures are reproduced from the sibling project's report verbatim.
