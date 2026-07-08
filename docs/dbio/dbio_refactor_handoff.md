# dbio refactor — handoff

## Goal

`biom3.dbio` parses biological reference databases (UniProt
SwissProt/TrEMBL `.dat`, Pfam Stockholm/FASTA, ExPASy, BRENDA, SMART)
and turns them into CSVs used for model training. The current parsing
and storage is fragile and ad hoc. Review the basic structure of this repository and familiarize yourself with the current dbio subpackage. Be critical. I think it is very poor code and overly complicated.

## What to do now (scope: parsers only)

Rewrite the parsers in `src/biom3/dbio/parsers/` to be:

- **Simple and self-contained.** One parser per database, in its own
  file. A shared ABC record class is fine, that abstracts the idea of each row being an "entry" with a number of fields, but no abstract base class hierarchy, no
  cross-database "schema" machinery. Adding a new database = adding one
  new self-contained file.
- **Simple iterator output.** Each parser yields one Record per entry.
  Each parser documents its own keys. No shared typed-record type.
- **Reuse libraries.** Don't reimplement what already exists. Example:
  UniProt `.dat` entries carry a CRC64 in the `SQ` line; use
  Biopython's `Bio.SeqUtils.CheckSum.crc64` (biopython is already a
  dependency) rather than hand-writing CRC64.
- **Fail loud.** A truncated or corrupt input must raise, not silently
  return a short/partial result. Two known cases: (1) `pigz`/gzip
  decompression dying mid-stream; (2) a UniProt entry whose assembled
  sequence doesn't match its `SQ` length/CRC64.
- **Capture faithfully.** Parse the fields the source provides; don't
  silently drop information at the parse layer.
- **Review Uniprot data layout.** Read the document at https://web.expasy.org/docs/userman.html,
  focusing on the description of each field type (ID) and whether there are any differences in 
  convention between TrEMBL and SwissProt.

## Testing

- Create small **real** mini fixtures: a few real Swiss-Prot entries
  and a few real TrEMBL entries (e.g. fetched from
  `https://rest.uniprot.org/uniprotkb/<ACC>.txt`), committed under
  `tests/_data/dbio/`.
- Write focused tests that pull individual entries from the mini
  fixtures and assert each field/mapping is parsed correctly.
- Keep the existing legacy `tests/dbio_tests/` aside; build the new
  tests fresh under `tests/dbio_tests_v2` and we will later rename.

## What NOT to do

- Do not build a large multi-phase plan. Earlier attempts over-engineered
  this and were discarded.
- Do not change builders, readers, the orchestrator, or Stage 1 yet.
  Parsers only.

## Current state

- Branch `dbio-refactor` was reset to a clean slate (`35b4a98`); no
  parser refactor code exists yet.
