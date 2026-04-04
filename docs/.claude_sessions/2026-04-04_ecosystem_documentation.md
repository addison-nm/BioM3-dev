# Cross-Repository Ecosystem Documentation

**Date:** 2026-04-04
**Branch:** `addison-main`
**Pre-session state:** `git checkout f30d682` (BioM3-dev)

## Summary

Added unified ecosystem documentation across all three BioM3 repositories (BioM3-dev, BioM3-data-share, BioM3-workflow-demo). This establishes a consistent description of the multi-repo project structure at three levels: README summaries, a detailed shared docs file, and CLAUDE.md context for AI-assisted development. Also created a machine-specific path registry (`.claude/repo_paths.json`, gitignored) so Claude Code sessions can locate sibling repos without hardcoded paths.

## Changes across repos

### All 3 repos
- **`docs/biom3_ecosystem.md`** (new, identical in all): Comprehensive ecosystem doc covering repo table, ASCII dependency diagram, shared data architecture with per-machine paths (DGX Spark, Polaris, Aurora), version compatibility approach (SYNC_LOG.md pattern + known-good commits), machine-specific path config, and cross-repo workflow descriptions.
- **`README.md`** (modified): Added ecosystem table under About section listing all 4 repos (including planned BioM3-workspace-template) with links and one-line descriptions. Links to `docs/biom3_ecosystem.md` for details.
- **`.claude/repo_paths.json`** (new, gitignored): Machine-specific JSON mapping repo names to absolute paths. Includes a null placeholder for the planned BioM3-workspace-template.

### BioM3-dev
- **`CLAUDE.md`** (modified): Added "Ecosystem context" section between Project overview and Repository layout. Lists related repos and references `.claude/repo_paths.json`.

### BioM3-data-share
- **`.gitignore`** (modified): Added `.claude/` entry — was previously missing, unlike the other two repos.
- **`CLAUDE.md`** (modified): Added "Practices" section (`Store session notes in docs/.claude_sessions/`) and "Ecosystem context" section between Structure and Conventions.
- **`SYNC_LOG.md`** (new): Tracks which BioM3-dev versions produced the weights, datasets, and database configs in data-share. Mirrors the pattern already used in BioM3-workflow-demo, but framed around data provenance rather than code dependency.

### BioM3-workflow-demo
- **`CLAUDE.md`** (new file): Created from scratch with sections for Practices, Project overview (8-step pipeline), Ecosystem context (includes SYNC_LOG.md reference), Repository layout, Building and running, and Commit style.

## Commits

| Repository | Commit | Message |
|------------|--------|---------|
| BioM3-data-share | `23493d5` | `docs: add ecosystem documentation and CLAUDE.md updates` |
| BioM3-data-share | `a8f7c72` | `docs: add SYNC_LOG.md for tracking BioM3-dev compatibility` |
| BioM3-dev | `f1f1600` | `docs: add ecosystem documentation across README, CLAUDE.md, and docs` |
| BioM3-dev | `38940c9` | `docs: add session note for ecosystem documentation` |
| BioM3-workflow-demo | `5c50286` | `docs: add ecosystem documentation and CLAUDE.md` |

None of the commits were pushed.

## Design decisions

- **Identical ecosystem doc**: The `docs/biom3_ecosystem.md` file is maintained identically across all repos (marked with an HTML comment at the top). This was chosen over per-repo tailored versions for easier maintenance.
- **SYNC_LOG pattern for versioning**: Rather than creating a formal compatibility matrix at this early stage (v0.0.1), the existing `SYNC_LOG.md` approach in BioM3-workflow-demo is documented as the recommended pattern for tracking cross-repo compatibility. Extended to BioM3-data-share for data provenance tracking. Decided against adding one to BioM3-dev since it's the upstream repo — its git log is the source of truth, and downstream repos' SYNC_LOGs already reference it.
- **`.claude/repo_paths.json` format**: JSON chosen for consistency with the existing config files in `configs/`. Contains a null placeholder for the planned BioM3-workspace-template.
- **`.claude/` gitignore fix**: BioM3-data-share was the only repo missing `.claude/` from its `.gitignore`. Fixed as a prerequisite before creating the path file.

## Future work

- **BioM3-workspace-template**: When this repo is created, all four `docs/biom3_ecosystem.md` files and three README ecosystem sections will need updating to add its GitHub link and remove the "(Planned)" marker.
- **Ecosystem doc sync**: No automation for keeping the identical docs in sync. Manual updates required when the ecosystem doc changes. Could consider a script or CI check in the future.
- **Commits not pushed**: All three repos have unpushed commits.
