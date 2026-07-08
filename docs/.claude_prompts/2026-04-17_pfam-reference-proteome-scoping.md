# Prompt: document Pfam reference-proteome scoping in dbio docs

Please write a session note to `docs/.claude_sessions/2026-04-17_pfam-reference-proteome-scoping.md`
and update `docs/building_datasets_with_dbio.md` to capture a user-coverage
question that came out of the sibling dataset-construction project
(`~/Projects/BioM3-projects/dataset-construction/`).

## Context

The user built a fresh Pfam source CSV with `biom3_build_source_pfam` against
Pfam release **38.1** (`Pfam-A.fasta.gz` + `Pfam-A.full.gz` from
`/data/data-share/BioM3-data-share/databases/pfam/`) and observed that
**PF00018 (SH3)** contained only **26,468 rows**, whereas external sources
(InterPro's SH3 page, casual googling) report *~215,000* SH3-domain proteins
across UniProtKB. They asked whether the builder was losing ~88% of hits.

It is not. The row count reflects Pfam's upstream scoping decision, not a bug
or filter in the dbio builder.

## The finding to document

All of this is verifiable from `relnotes.txt` shipped alongside `Pfam-A.full.gz`
in the Pfam 38.1 release (present at
`/data/data-share/BioM3-data-share/databases/pfam/relnotes.txt`, see lines
92–132 for the scope table and discontinuation note).

### Pfam upstream scope has changed over time

| Release range | Upstream sequence scope | Files shipped |
|---|---|---|
| ≤ 28 | All of UniProtKB | `Pfam-A.full` (full UniProtKB) |
| 29 – 37.0 | **UniProt Reference Proteomes** (default for `Pfam-A.full`) | `Pfam-A.full` (RP) + `Pfam-A.uniprot` (full UniProtKB HMM matches, separate file) |
| **37.1+** (Nov 2024 onward) | UniProt Reference Proteomes **only** | `Pfam-A.full` (RP); `Pfam-A.uniprot` **discontinued** |

Release-notes confirmation (line 132): *"Matches to UniProtKB were
discontinued as of Pfam 37.1."*

### What this means for `biom3_build_source_pfam`

- The builder parses `Pfam-A.full.gz` and `Pfam-A.fasta.gz` directly.
- In Pfam 38.1, those files cover **~69.6M sequences across ~27,500 families** — all from UniProt Reference Proteomes 2025_03 (`relnotes.txt` line 104).
- Per-family row counts are therefore bounded by RP coverage, not full UniProtKB. PF00018 (SH3) = 26,468 rows in this release; that is the complete Pfam 38.1 figure.
- The "~215,000 SH3 proteins" number surfaces on InterPro's family page. InterPro computes that on the fly by running the PF00018 HMM against UniProtKB/UniParc via InterProScan — it is a **display-time** number, not a downloadable Pfam artifact as of 37.1+.

### Workarounds if broader-than-RP coverage is needed

1. **Run `hmmscan` locally** with `Pfam-A.hmm.gz` (already on the data share) against a target DB (Swiss-Prot, TrEMBL, or UniProtKB). This reproduces what InterProScan does and produces a family-membership table equivalent to the old `Pfam-A.uniprot.gz`. This is the direction the Pfam team now points downstream users.
2. **Pin to an older Pfam release (≤ 37.0)** and parse the legacy `Pfam-A.uniprot.gz` file. Trades release freshness for coverage breadth; row counts per family will be several times larger but family IDs will not include the ~5,500 new entries added in 37.1–38.1.

The dbio builder supports neither workaround today and probably shouldn't try —
both are "pick your scope" decisions that belong upstream of the CSV builder.

## What to update

### 1. `docs/building_datasets_with_dbio.md`

Add a short "Pfam coverage scope" subsection under the Pfam source-CSV
description explaining that:

- `biom3_build_source_pfam` row counts mirror the `Pfam-A.full` / `Pfam-A.fasta` scope of whichever Pfam release is fed in.
- Since Pfam 29 that scope is UniProt Reference Proteomes, not all of UniProtKB.
- `Pfam-A.uniprot.gz` is no longer published as of Pfam 37.1, so even swapping in that file isn't an option for recent releases.
- Users who need full UniProtKB coverage should run `hmmscan` / InterProScan themselves rather than expecting the CSV to contain it.

Keep this terse — one paragraph plus a pointer to `relnotes.txt`. The audience
is future users hitting the same "why are my PF-X counts lower than InterPro says"
question.

### 2. Session note `docs/.claude_sessions/2026-04-17_pfam-reference-proteome-scoping.md`

Standard session-note format: short context paragraph, the scope-table above,
a note that this was *not* a builder bug, and a pointer to the doc update.

## Format guidance

- Match existing `docs/.claude_sessions/` conventions (date-prefixed filename, concise bullets).
- Do not attempt to re-run the builder or re-derive counts — use the figures above verbatim. Inputs are too large to re-verify in-session.
- Cross-link to `src/biom3/dbio/build_source_pfam.py` where it reads `Pfam-A.full` / `Pfam-A.fasta`, and to `relnotes.txt` on the Pfam FTP for the discontinuation evidence.
- Do **not** suggest adding an RP-vs-UniProtKB flag to the builder. The fix, if any, is documentation + a sibling pipeline that runs `hmmscan`, not a builder-level option.
