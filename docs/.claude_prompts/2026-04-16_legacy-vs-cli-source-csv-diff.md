# Prompt: document legacy vs. CLI source CSV differences

Please write a session note to `docs/.claude_sessions/2026-04-16_legacy-vs-cli-source-csv-diff.md` documenting a comparison between the legacy source CSVs and fresh CLI-generated versions. This came out of a dataset-construction sibling project (`~/Projects/BioM3-projects/dataset-construction/`) where we regenerated both source CSVs from scratch using `biom3_build_source_swissprot` and `biom3_build_source_pfam` on current databases and compared them against the legacy reference files in `/data/data-share/BioM3-data-share/data/datasets/LEGACY_*`.

## Findings to capture

All numbers verified via `wc -l`, `head`, `tail`, and `grep` on the actual files.

### Swiss-Prot (`fully_annotated_swiss_prot.csv`)

| | LEGACY | CLI-generated |
|-|--------|---------------|
| Rows | 569,517 | 547,273 |
| Columns | 6 | 4 |
| Ordering | alphabetical by accession | `.dat` file order |
| Entries with `['nan']` pfam_label | 31,144 | 0 |

Three material differences:

1. **`require_pfam=True` is hard-wired** in `src/biom3/dbio/build_source_swissprot.py:96`. LEGACY kept 31,144 Swiss-Prot entries with no Pfam cross-ref and stamped `pfam_label = ['nan']`; the CLI silently drops them. Net row-count delta reconciles as: `-31,144 (filtered) + ~8,900 (new entries since LEGACY was built) = -22,244`. The builder currently has no CLI flag to disable this filter — recovering legacy parity requires a code change.

2. **Intermediate caption columns removed.** LEGACY carried `text_caption` (raw, with `(PubMed:NNN)` refs and `{ECO:...}` tags), `[clean]text_caption` (stripped), and `[final]text_caption`. CLI emits only `[final]text_caption`. This loses the provenance needed to audit what PubMed/ECO stripping is removing.

3. **Row order differs.** LEGACY is alphabetically sorted by accession post-build (`A0A009IHW8` → `X6R8R1`); CLI writes in `.dat` file order (`Q6GZX4` → `A9JR22`, matching Swiss-Prot's internal ordering). Any downstream code doing positional head/tail sampling or train/val splits by row index will see different data.

### Pfam (`Pfam_protein_text_dataset.csv`)

| | LEGACY | CLI-generated |
|-|--------|---------------|
| Rows | 44,767,155 | 63,237,515 |
| Columns | 8 (identical schema) | 8 |
| Max Pfam ID observed | `PF17xxx` range | `PF26733` |

The ~41% row-count jump is **a Pfam release version difference, not a builder difference.** LEGACY's Pfam IDs top out in the PF10xxx-PF17xxx range; the fresh build has `PF26733` — that's ~Pfam v37+ picking up thousands of new families and many more sequence hits per family. Caption format is byte-identical (lowercase `Protein name:` / `Family description:`).

## Recommended follow-ups to note in the session document

- Consider adding a `--require_pfam / --no_require_pfam` flag to `biom3_build_source_swissprot` so the hard-coded default can be overridden without a code change.
- Consider optional flags (e.g. `--keep_intermediate_captions`) to emit the `text_caption` and `[clean]text_caption` columns for auditing PubMed/ECO stripping.
- Document expected row-order difference (`.dat` order vs. alphabetical) and its effect on any downstream positional splits.
- Capture the Pfam release version (and UniProt release date) used to build each CSV in `build_manifest.json` or a sibling provenance file — today there is no trace in the output of which DB release produced it.

## Format guidance

- Use the session-note style already present in `docs/.claude_sessions/` (date-prefixed filename, short context paragraph, bullet findings, clear separation between observations and recommendations).
- Do not attempt to run the builders or re-derive the numbers — use the figures above verbatim. The inputs are too large to re-verify in-session.
- Cross-link to `src/biom3/dbio/build_source_swissprot.py:96` and `docs/building_datasets_with_dbio.md` where relevant.
