# `biom3.dbio` refactor — execution plan

Branch: `dbio-refactor` (based on `dev`, merges back to `dev`; no worktree).
Resolves the findings in [dbio_fragility_audit.md](dbio_fragility_audit.md).

## Goals (from the audit + user decisions)

1. **Solid, fail-loud parsers** for UniProt (SwissProt + TrEMBL), Pfam,
   ExPASy, BRENDA, SMART. One abstract base so every parser returns a
   consistent typed record and raises on corrupt/truncated input.
   Capture UniProt comprehensively: add gene names (`GN`), keywords
   (`KW`), protein existence (`PE`), organism species (`OS`), the full
   `DE` name hierarchy, expanded `CC` topics and `DR` cross-refs. The
   feature table (`FT`) is **deferred** to a follow-up.
2. **Builders just convert a raw DB into one consistent CSV per
   source.** No caption composition in builders. Consistent annotation
   fields + coverage stats.
3. **One reader per source CSV.** Schema is fixed per source, so each
   reader validates its columns and fails loud on mismatch. Two schema
   *families*, not one shape:
   - **sequence-bearing** (SwissProt/TrEMBL/Pfam): a row is a protein.
   - **reference** (ExPASy/BRENDA/SMART): a row is an EC/domain record.
     These join onto sequence rows by EC number / SMART id.
4. **A separate prompt-construction layer** reads source CSVs and
   composes the training text prompt (`[final]text_caption`) + the
   final training CSV.

## Settled decisions

- **Pfam IDs in CSVs:** one simple format everywhere — semicolon-
  separated tokens (`PF00018;PF00169`), empty cell when none. One
  `join_pfam(ids) -> str` writer, one `split_pfam(cell) -> list[str]`
  reader. The `['nan']` magic-string sentinel is removed. Stage 1 stops
  matching the literal `'nan'` and treats "no Pfam IDs" as empty
  generally.
- **Downstream contract:** the final training CSV keeps
  `primary_Accession`, `protein_sequence`, `[final]text_caption`,
  `pfam_label`. Stage 1 / RL / benchmarks need no logic change *except*
  the lockstep edits forced by the Pfam-source rename and the
  `pfam_label` storage change (Commit 4).
- **Reuse, don't reinvent:** `core.run_utils.write_manifest` /
  `get_file_metadata`, `core.helpers.load_json_config`,
  `configs/dbio_config.json`, `biom3.backend.device.setup_logger`,
  the `<stem>.build_manifest.json` + `<stem>.stats.md` sidecars.

## Commit sequence (each individually green under `pytest tests/ --quick`)

| # | Lands | Gate |
|---|---|---|
| 1 | **C1 fail-loud**: pigz decompression raises on truncation. `parsers/base.py` exceptions; `_open_gz` wrapper. | new truncated-`.gz` test |
| 2 | `schema.py`: two-family frozen schemas + `split_pfam`/`join_pfam`. Additive, no callers. | schema/helper unit tests |
| 3 | Collapse the two `.dat` entry parsers into one (H2); add `GN/KW/PE/OS`/full-`DE`/expanded `CC`+`DR`. FT deferred. | parser-equivalence + expanded-fixture tests |
| 4 | **Atomic breaking commit**: simple `pfam_label` storage in every producer/consumer; drop `['nan']`; rename Pfam `id`/`sequence` → `primary_Accession`/`protein_sequence`; lockstep Stage 1 (`preprocess.py` ast.literal_eval / `'nan'` check / groupby / direct Pfam read); rewrite schema-coupled tests. | round-trip + schema-mismatch-raises |
| 5 | Builders emit annotation-only CSVs. | per-source schema tests |
| 6 | Prompt-construction layer (single caption engine; kill dual composer M1; vectorize joins M2). | cross-composer equivalence |
| 7 | Orchestrator builds final CSV via prompt layer; no Layer-1 parsing in the pipeline. | end-to-end mini build |
| 8 | Derived-artifact builders (annotation cache, csv_to_parquet) consume `schema.py`. | round-trip |
| 9 | Verify 11 console_scripts + entrypoint smoke. | full `pytest tests/` |
| 10 | Rewrite affected docs, cross-link audit, session note, update memory. | — |

## Lockstep downstream edits (Commit 4, outside dbio)

- `src/biom3/Stage1/preprocess.py`: `:451`/`:625`/`:829`
  `ast.literal_eval` → `split_pfam`; `:647` `if 'nan' in pfam_labels`
  → `if not pfam_labels`; `:458`/`:602`/`:887` Pfam-source `pfam_label`
  groupby (single-token, still works); `:605-606`/`:876` Pfam-source
  `id`/`sequence` → `primary_Accession`/`protein_sequence`.
- `run_PenCL_inference.py:206-229` inline fixture `pfam_label` form.
- `rl/grpo.py`, `benchmarks/Stage3/generation.py`: no change (final-CSV
  column names preserved); re-verify under full tests.

## Out of scope here (operational, user-run on the right machine)

The one-time rebuild of the 63M-row source/derived CSVs is **not**
executed as part of this branch. It writes through `data/datasets/`
symlinks into shared cluster storage (see memory
`project_overwritten_shared_csvs`) — it must be run deliberately into a
scratch path, parity-validated against the prior build, and promoted
behind a backup. Documented in Commit 10; performed by the user.
</content>
