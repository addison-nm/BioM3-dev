# `biom3.dbio` refactor plan

This plan executes a coordinated cleanup of the `biom3.dbio` subpackage:
deprecate the legacy UniProt REST client, tag a stable release as a
fallback point, and reorganize the package into per-layer subpackages
with consistent naming. The architectural rationale is in
[dbio_package_audit.md](dbio_package_audit.md); this document is the
execution plan.

## Status

| Phase | Status | Notes |
|---|---|---|
| 0. Promote TrEMBL annotation cache to data share | **done** | 2026-05-06; cache available at `dbio_constructs/trembl_annotation_cache_v20260506.parquet` |
| 1. Deprecate `uniprot_client.py` | not started | Phase 1 of this plan |
| 2. Tag release | not started | Phase 2 |
| 3. Subpackage refactor | not started | Phase 3 — the bulk of the work |
| 4. Remove deprecation shims | future | Phase 4 (≥ 2 releases after Phase 3 lands) |

## Goals

- **Eliminate the external API dependency.** `uniprot_client.py` was
  the original enrichment path before local `.dat` parsing existed;
  the TrEMBL annotation cache (built 2026-05-06, promoted to the data
  share) closes the last gap that made the API necessary. Keeping it
  wired in adds maintenance surface (rate-limit changes, schema
  changes, retry tuning) for a path that no current workflow needs.
- **Make file roles unambiguous from filename alone.** Today's naming
  has parsers and readers sharing bare-name patterns (`expasy.py` is a
  parser, `swissprot.py` is a reader, `taxonomy.py` is two classes
  fused). New contributors have to read each file to understand what
  layer it belongs to.
- **Express the four-layer architecture in the directory tree.** The
  audit doc names four layers (parsers, readers, builders,
  orchestrators); the package itself flattens all 26 files into one
  directory. A subpackage-per-layer structure makes the dataflow
  visible in `ls`.
- **Keep the change reversible.** Tagging a release before the
  structural reorg gives external consumers a known-good pin and gives
  us a clean revert point if the migration uncovers unexpected
  breakage.

## Non-goals

- **No behavior changes.** Pure rename + deprecation work. Output CSV
  schemas, manifest formats, stats markdown, and CLI flag surfaces
  stay byte-identical.
- **No splitting of `enrich.py`.** It's the largest file (738 lines)
  and a natural seam exists between the JSON parsers and the join
  helpers, but the public API of `enrich_dataframe` is consumed across
  the package. Defer that split to a follow-on refactor with its own
  plan.
- **No splitting of `taxonomy.py`** into `tree.py` +
  `accession_index.py`. Same rationale — defer.
- **No reclassification of `build_annotated_pfam_subsets.py`.** It
  stays a builder. The audit doc may add a "selective builder" note
  in its module docstring during the refactor, but the architectural
  layer doesn't change. See the audit doc for the reasoning.

---

## Phase 1 — deprecate `uniprot_client.py`

**Goal:** signal publicly that the API path is going away, without
breaking any current invocation. Users have ≥ 2 releases to migrate.

### Changes

| File | Change |
|---|---|
| [`src/biom3/dbio/uniprot_client.py`](../src/biom3/dbio/uniprot_client.py) | Add module-level `warnings.warn(..., DeprecationWarning, stacklevel=2)` at import time. Update module docstring to point at `--annotation_cache` and `--uniprot_dat` as the supported paths. |
| [`src/biom3/dbio/build_dataset.py`](../src/biom3/dbio/build_dataset.py) | When the API fallback path is taken (i.e., `--enrich_pfam` without `--annotation_cache` and without `--uniprot_dat`), emit a runtime warning with the same migration pointer. |
| [`docs/building_datasets_with_dbio.md`](building_datasets_with_dbio.md) | Remove API-first examples. Reorder the enrichment-strategy section so cache + `.dat` are the documented paths; API is a "legacy / not recommended" footnote. |
| [`docs/dbio_examples.md`](dbio_examples.md) | Same — replace any `--enrich_pfam` (no cache/dat) examples with cache-first equivalents. |

### Tests

- Add a single test verifying `DeprecationWarning` is emitted when
  `biom3.dbio.uniprot_client` is imported.
- Existing API-path tests (if any in `test_build_dataset.py`) get a
  `pytest.filterwarnings("ignore::DeprecationWarning")` mark so they
  don't pollute the suite output. They keep running because Phase 1
  doesn't remove the code path.

### Commit shape

One commit, one PR. Conventional Commits: `chore(dbio): deprecate
uniprot_client REST API path`. Body explains the rationale and links
to this plan.

### Acceptance

- `pytest tests/dbio_tests/` passes.
- `python -c "from biom3.dbio import uniprot_client"` emits exactly
  one `DeprecationWarning`.
- `biom3_build_dataset --enrich_pfam` (no cache/dat) emits a runtime
  warning naming the migration paths.

---

## Phase 2 — tag a release

**Goal:** clean revert point and a stable pin for external consumers
who haven't migrated yet. No code change beyond the version bump.

### Changes

- Bump version in [`pyproject.toml`](../pyproject.toml) from `0.1.0a5`
  to `0.2.0` (or whatever the next semver-appropriate tag is).
- Update [`docs/versioning.md`](versioning.md) with a release-notes
  entry: TrEMBL cache promoted, API deprecated, structural refactor
  pending.
- Tag the commit (`git tag v0.2.0`).

### Acceptance

- Tag exists on `addison-dev` HEAD after Phase 1 lands.
- `pip install -e .` reports the new version.

---

## Phase 3 — subpackage refactor

**Goal:** reorganize the 26 files into per-layer subpackages with
consistent naming, and provide backward-compat shims for ≥ 2 releases
to allow external consumers to migrate without breakage.

### Target structure

```
src/biom3/dbio/
├── __init__.py
├── __main__.py             ← CLI dispatch (lazy imports updated to new paths)
├── config.py               ← cross-cutting: path resolution
├── caption.py              ← cross-cutting: CaptionSpec + composers
├── stats.py                ← cross-cutting: coverage report
├── enrich.py               ← cross-cutting: annotation enrichment + joins
│
├── parsers/                ← Layer 1: raw-DB format parsers
│   ├── __init__.py
│   ├── swissprot_dat.py    ← (no rename; suffix already expressive)
│   ├── pfam_stockholm.py   ← from pfam_metadata.py
│   ├── expasy_dat.py       ← from expasy.py
│   ├── smart_tsv.py        ← from smart.py
│   ├── brenda_flatfile.py  ← from brenda.py
│   └── uniprot_api.py      ← from uniprot_client.py (still deprecated)
│
├── readers/                ← Layer 2: filtered/indexed CSV readers
│   ├── __init__.py
│   ├── base.py             ← from base.py (DatabaseReader ABC)
│   ├── swissprot_csv.py    ← from swissprot.py
│   ├── pfam_csv.py         ← from pfam.py
│   └── taxonomy.py         ← (no rename; deferred split)
│
├── builders/               ← Layer 3: primary source-CSV builders
│   ├── __init__.py
│   ├── source_swissprot.py ← from build_source_swissprot.py
│   ├── source_pfam.py      ← from build_source_pfam.py
│   ├── source_trembl.py    ← from build_source_trembl.py
│   ├── source_expasy.py    ← from build_source_expasy.py
│   ├── source_smart.py     ← from build_source_smart.py
│   ├── source_brenda.py    ← from build_source_brenda.py
│   └── pfam_subsets.py     ← from build_annotated_pfam_subsets.py
│
├── helpers/                ← Layer 3 helpers (derived artifacts)
│   ├── __init__.py
│   ├── annotation_cache.py ← from build_annotation_cache.py
│   └── csv_to_parquet.py   ← from convert.py
│
└── orchestrators/          ← Layer 4: combine multiple sources
    ├── __init__.py
    └── build_dataset.py    ← (no rename; CLI matches existing biom3_build_dataset)
```

### Naming principles

- **Parsers** named after their input format suffix: `_dat`,
  `_stockholm`, `_tsv`, `_flatfile`, `_api`. Always.
- **Readers** named after their input artifact: `_csv` for CSV/Parquet
  readers.
- **Builders** lose the `build_` prefix because the directory already
  says "builders". `source_` prefix marks primary source-CSV
  producers; bare names mark derived (per-family selective) builders.
- **Helpers** descriptively named (`annotation_cache.py`,
  `csv_to_parquet.py`) — they're not "build a source CSV from a
  database", they're "produce a derived performance artifact".
- **Orchestrators** keep `build_` because user-facing CLI is
  `biom3_build_dataset` and matching reduces cognitive load.

### Per-file mapping table

| Current path | New path | Notes |
|---|---|---|
| `dbio/__init__.py` | `dbio/__init__.py` | Stays empty |
| `dbio/__main__.py` | `dbio/__main__.py` | Update lazy imports to new module paths |
| `dbio/config.py` | `dbio/config.py` | Cross-cutting; stays at top level |
| `dbio/caption.py` | `dbio/caption.py` | Cross-cutting; stays at top level |
| `dbio/stats.py` | `dbio/stats.py` | Cross-cutting; stays at top level |
| `dbio/enrich.py` | `dbio/enrich.py` | Cross-cutting; stays at top level |
| `dbio/base.py` | `dbio/readers/base.py` | Move (DatabaseReader is reader-only ABC) |
| `dbio/swissprot.py` | `dbio/readers/swissprot_csv.py` | Rename + move |
| `dbio/pfam.py` | `dbio/readers/pfam_csv.py` | Rename + move |
| `dbio/taxonomy.py` | `dbio/readers/taxonomy.py` | Move (split deferred) |
| `dbio/swissprot_dat.py` | `dbio/parsers/swissprot_dat.py` | Move |
| `dbio/pfam_metadata.py` | `dbio/parsers/pfam_stockholm.py` | Rename + move |
| `dbio/expasy.py` | `dbio/parsers/expasy_dat.py` | Rename + move |
| `dbio/smart.py` | `dbio/parsers/smart_tsv.py` | Rename + move |
| `dbio/brenda.py` | `dbio/parsers/brenda_flatfile.py` | Rename + move |
| `dbio/uniprot_client.py` | `dbio/parsers/uniprot_api.py` | Rename + move (still deprecated) |
| `dbio/build_source_swissprot.py` | `dbio/builders/source_swissprot.py` | Rename + move |
| `dbio/build_source_pfam.py` | `dbio/builders/source_pfam.py` | Rename + move |
| `dbio/build_source_trembl.py` | `dbio/builders/source_trembl.py` | Rename + move |
| `dbio/build_source_expasy.py` | `dbio/builders/source_expasy.py` | Rename + move |
| `dbio/build_source_smart.py` | `dbio/builders/source_smart.py` | Rename + move |
| `dbio/build_source_brenda.py` | `dbio/builders/source_brenda.py` | Rename + move |
| `dbio/build_annotated_pfam_subsets.py` | `dbio/builders/pfam_subsets.py` | Rename + move |
| `dbio/build_annotation_cache.py` | `dbio/helpers/annotation_cache.py` | Rename + move |
| `dbio/convert.py` | `dbio/helpers/csv_to_parquet.py` | Rename + move |
| `dbio/build_dataset.py` | `dbio/orchestrators/build_dataset.py` | Move (no rename) |

### Backward-compat shims

For each renamed/moved file, create a stub at the **old path** that
re-exports the public API and emits a `DeprecationWarning` at import:

```python
# src/biom3/dbio/swissprot.py (shim — DEPRECATED)
import warnings
warnings.warn(
    "biom3.dbio.swissprot is deprecated; import from "
    "biom3.dbio.readers.swissprot_csv instead. Will be removed in v0.4.0.",
    DeprecationWarning,
    stacklevel=2,
)
from biom3.dbio.readers.swissprot_csv import *  # noqa: F401, F403
```

Apply this pattern to every renamed file. The shims live in `dbio/`
at the old paths until Phase 4 removal. Internal imports inside the
package always go through the new paths — shims exist solely for
external consumers.

### Internal import updates

Every internal `from biom3.dbio.X import Y` statement gets rewritten
to the new path. Inventory of files that need updating (all internal
to the package + tests):

- `src/biom3/dbio/__main__.py` — 9 lazy imports
- `src/biom3/dbio/builders/source_*.py` — each imports from
  `caption`, `stats`, `parsers/*`, `core.run_utils`
- `src/biom3/dbio/builders/pfam_subsets.py` — imports `caption`,
  `stats`, `parsers/pfam_stockholm`, `builders/source_pfam`
  (`PFAM_SPEC`)
- `src/biom3/dbio/helpers/annotation_cache.py` — imports `enrich`,
  `parsers/swissprot_dat`
- `src/biom3/dbio/helpers/csv_to_parquet.py` — no internal dbio
  imports (depends only on pandas/pyarrow)
- `src/biom3/dbio/orchestrators/build_dataset.py` — imports
  `readers/*`, `enrich`, `stats`, `config`, `core.run_utils`
- `src/biom3/dbio/readers/swissprot_csv.py` — imports
  `readers/base`
- `src/biom3/dbio/readers/pfam_csv.py` — imports `readers/base`
- `src/biom3/dbio/readers/taxonomy.py` — no internal dbio imports
- `src/biom3/dbio/enrich.py` — currently imports from `caption`; may
  pick up `parsers/*` paths for join-layer coverage
- `tests/dbio_tests/*.py` — all 18+ test files; rewrite imports to
  match new paths. Tests that hit the deprecation shims should be
  rewritten to use the new paths so the suite isn't drowning in
  warnings.

### Other repo touchpoints

- [`pyproject.toml`](../pyproject.toml) — entry-point definitions
  reference `biom3.dbio.__main__:run_*`, which is unaffected. Internal
  to `__main__.py`, the lazy imports change.
- [`scripts/`](../scripts) — grep for `from biom3.dbio` or `import
  biom3.dbio`. Update any matches.
- [`demos/`](../demos) — same.
- External repos (`BioM3-workflow-demo`, `BioM3-workspace-template`,
  any user notebooks) — these will hit shims and emit deprecation
  warnings. Document the migration in the commit message and in a
  session note so consumers know to update.

### Test reorganization (optional but recommended)

Mirror the test layout to the source layout:

```
tests/dbio_tests/
├── parsers/
│   ├── test_swissprot_dat.py
│   ├── test_pfam_stockholm.py
│   ├── test_expasy.py
│   ├── test_smart.py
│   └── test_brenda.py
├── readers/
│   ├── test_swissprot_csv.py
│   ├── test_pfam_csv.py
│   └── test_taxonomy.py
├── builders/
│   ├── test_source_swissprot.py
│   ├── test_source_pfam.py
│   ├── test_source_trembl.py
│   ├── test_source_expasy.py
│   ├── test_source_smart.py
│   ├── test_source_brenda.py
│   └── test_pfam_subsets.py
├── helpers/
│   ├── test_annotation_cache.py
│   └── test_csv_to_parquet.py
├── orchestrators/
│   └── test_build_dataset.py
├── test_caption.py
├── test_enrich.py
├── test_stats.py
└── test_imports.py
```

This is optional for the refactor PR but worth doing in the same
sweep — splitting it into a follow-on PR risks the layouts drifting.

### Commit shape

One PR, several commits in this order so each is bisectable:

1. `refactor(dbio): create parsers/ readers/ builders/ helpers/
   orchestrators/ subpackages with shims` — moves files, creates
   shims, leaves internal imports pointing at old paths.
2. `refactor(dbio): rewrite internal imports to new module paths` —
   updates `__main__.py`, builders, orchestrators, etc. Shims are
   still in place.
3. `test(dbio): rewrite test imports to new module paths` —
   `tests/dbio_tests/*.py`.
4. `refactor(dbio): mirror test layout to source layout` (optional).
5. `docs(dbio): update audit + building-datasets docs for new
   layout`.

CI must pass after each commit so a future bisect on the post-Phase-3
codebase doesn't trip over a transient broken state.

### Acceptance

- `pytest tests/dbio_tests/` passes (210+ tests).
- `pytest tests/test_imports.py` passes.
- `python -c "import biom3.dbio.swissprot"` emits a
  `DeprecationWarning` and successfully resolves
  `SwissProtReader`.
- `biom3_build_source_swissprot --help`,
  `biom3_build_dataset --help`, etc. all run without error.
- A smoke build of the SH3 dataset (`biom3_build_dataset
  --pfam_ids PF00018 --outdir /tmp/sh3_smoke`) produces
  byte-identical output to a build done on the pre-Phase-3 commit.

---

## Phase 4 — remove deprecation shims (future, ≥ 2 releases later)

Schedule for v0.4.0 (or the second tagged release after Phase 3
lands).

### Changes

- Delete every shim file at the old paths.
- Delete `src/biom3/dbio/parsers/uniprot_api.py` itself, plus the
  API fallback path in `orchestrators/build_dataset.py`.
- Update [`docs/dbio_package_audit.md`](dbio_package_audit.md) to
  remove the deprecated entries.
- Release notes entry naming the removed paths so users have a clear
  migration record.

### Acceptance

- `pytest tests/dbio_tests/` passes.
- `python -c "import biom3.dbio.swissprot"` raises `ModuleNotFoundError`
  (the shim is gone).
- `biom3_build_dataset --enrich_pfam` (no cache/dat) raises an error
  instead of falling through to the API.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| External code (workflow-demo, workspace-template, user notebooks) imports through old paths and breaks | High | Shims in place for ≥ 2 releases; deprecation warning includes target removal version; release-notes entry calls out migration |
| Hidden internal import we miss during the rewrite | Medium | Phase 3 commit (2) does an exhaustive `grep -r "from biom3.dbio"` across `src/`, `tests/`, `scripts/`, `demos/`. CI catches anything missed. |
| Smoke build produces non-identical output post-refactor | Low | No behavior change is intended. Bisectable commit history and the SH3 smoke test in the acceptance criteria catch anything subtle. Worst case: revert to the v0.2.0 tag and re-plan. |
| Shim files accumulate in the repo and confuse readers | Medium | Shim files are < 10 lines each, marked with a clear `DEPRECATED` comment. Phase 4 deletes them on a schedule. |
| Test layout reorganization slips and merges separately, layouts drift | Low–medium | Bundle the test reorg into the same PR as the source refactor (Phase 3 commit 4). |

## Decisions deferred

These came up during planning and are explicitly out of scope:

- **Splitting `enrich.py` into JSON-parser and join-helper modules.**
  Worth doing eventually but breaks a public API surface; deserves
  its own plan.
- **Splitting `taxonomy.py` into `tree.py` + `accession_index.py`.**
  Same reasoning — separate change.
- **Whether to fully remove `uniprot_api.py` in Phase 3** rather than
  carry it through deprecation in parsers/. Leaving it in for now
  keeps Phase 3 a pure rename + reorg with no semantic change. Phase 4
  removes it.
- **Reclassification of `build_annotated_pfam_subsets.py`.** Stays a
  builder. Audit doc may add a "selective builder" docstring during
  the refactor.

## Open questions

- **Release cadence between Phase 3 and Phase 4.** Two patch
  releases? Two minor releases? Default assumption: two minor
  releases with at least one month between them so external consumers
  have time to migrate.
- **Whether to write a separate migration guide.** A
  `docs/dbio_migration_v0.2_to_v0.3.md` may be worth it if external
  consumers (workflow-demo, workspace-template) make non-trivial use
  of internal imports. Decision: write it during Phase 3 if the
  internal-import grep reveals more than ~5 external touchpoints;
  skip it otherwise.

## Related documents

- [dbio_package_audit.md](dbio_package_audit.md) — current package
  layout and architectural rationale.
- [building_datasets_with_dbio.md](building_datasets_with_dbio.md) —
  user-facing CLI reference; will be updated in Phase 1 + Phase 3.
- [training_csv_provenance.md](training_csv_provenance.md) — column
  provenance reference; references current paths.
- [database_linkage.md](database_linkage.md) — cross-DB identifier
  spec; unaffected.
