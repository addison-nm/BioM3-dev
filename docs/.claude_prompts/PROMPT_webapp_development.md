# Prompt: develop the BioM3 webapp

Use this prompt to bootstrap a session focused on developing the Streamlit
webapp under `src/biom3/app/`. Load context first, then wait for the user's
specific request.

## Mission

You're extending the BioM3 webapp — a multi-page Streamlit app that exposes
the project's visualization and protein-generation tooling. Typical work is
either:

- **Surface existing capability** — wire a `viz/` or `Stage3.animation_tools`
  function that isn't yet in the UI.
- **Improve presentation** — rework how existing pages render data the app
  already produces.
- **Add new pages** — compose existing backends into new workflows.

**Default posture: REUSE over rebuild.** Before writing a new algorithm,
check `src/biom3/viz/` and `src/biom3/Stage3/animation_tools.py` — the most
common mistake is rebuilding something that already has a tested
implementation and just needs a thin Streamlit layer.

## Read these first (in order)

1. **Most recent session note** — `docs/.claude_sessions/` — sort by filename (they're date-prefixed), pick the newest webapp-related one. At the time of writing, that was `2026-04-18_webapp_viz_expansion.md` — the 9-commit round that established most current architecture. The session note's "Follow-ups considered but not done" section is the canonical backlog.
2. **Project conventions** — `CLAUDE.md` at the repo root: layout, worktree workflow, commit style.
3. **App architecture** — `docs/web_app.md` if present; otherwise skim `src/biom3/app/__init__.py` and the pages directory.

## Current architecture snapshot

*This is frozen at the time this prompt was written — verify against the
latest session note or the pages directory before relying on it.*

### Pages (`src/biom3/app/pages/`)

| # | Page | Purpose |
|---|---|---|
| 1 | View Structure | Single PDB viewer (file / paste / fold). Style, color scheme, B-factor/pLDDT toggle. |
| 2 | Align Structures | Superimpose two PDBs. Side-by-side pre-alignment view, then overlay with RMSD. |
| 3 | Highlight Residues | Highlight specified residues. |
| 4 | Color by Values | Color residues by user-provided scalar list. |
| 5 | Unmasking Order | Stage 3 diffusion order on structure + generated AA sequence with AA_COLORS. |
| 6 | BLAST Search | Remote BLAST; sortable hit table, CSV export, colored alignment, fold-a-hit. |
| 7 | Probability Dynamics | Chosen-token-probability heatmap (hide_pad / blank_unmasked). |
| 8 | Generation Animation | GIF/MP4 of diffusion unmasking (brightness / colorbar / logo styles). |
| 9 | BLAST Alignment | Query → BLAST → fetch (RCSB/AlphaFoldDB) or fold (ESMFold) → superimpose. |

### Shared layer

- `src/biom3/app/_data_browser.py` — `browse_file()` hierarchical picker (directory → subfolder → file) with substring filter.
- `src/biom3/app/_helpers.py` — `pick_pdb`, `pick_pt`, `pick_file`, `load_pt`, `render_view`, `render_colored_sequence`.
- `src/biom3/viz/` — `viewer.py`, `alignment.py`, `dynamics.py`, `unmasking.py`, `folding.py`, `sources.py`, `_tokens.py`.
- `src/biom3/Stage3/animation_tools.py` — `generate_sequence_animation`, `gif_to_mp4`, `MetricAnnotation`, `confidence_metric`.

## Patterns to follow

- **Session-state caching.** Cache anything expensive (BLAST ≈30-60s, ESMFold, remote fetches) in `st.session_state`, keyed so results survive reruns and page navigation. See `pages/6_BLAST_Search.py` and `pages/9_BLAST_Alignment.py` for the established pattern.
- **Graceful ImportError guards.** Optional deps (especially ESMFold) get `try/except ImportError` with the install-instruction error (match `pages/1_View_Structure.py`).
- **Colored residue rendering.** Use `_helpers.render_colored_sequence` rather than building HTML inline — it handles wrapping, position numbers, and foreground-luminance picking.
- **Per-residue structure coloring.** `viz.viewer.color_by_values` when you need per-residue control; py3Dmol's built-in schemes otherwise.
- **Bare-mode safety.** `st.stop()` does **not** raise in bare mode (smoke tests). If you write early-exit logic, wrap page bodies in a function and `return`; otherwise imports will fail with `NameError` on later references.

## Operational workflow

### Worktree (for substantive features)

```
git worktree add .claude/worktrees/<feature> -b addison-<feature> addison-dev
```

Small fixes and docs: go directly on `addison-dev`.

**After creating a worktree, check data symlinks.** New worktrees start with
empty `data/databases/` and `data/datasets/` (only `.gitkeep` is tracked; the
symlinks to `/data/data-share/…` are gitignored and machine-specific).

```
ls data/databases/ data/datasets/
```

If empty (or only `.gitkeep`), populate from the appropriate sync script:

- `scripts/sync_databases.sh` — symlinks under `data/databases/`.
- `scripts/sync_weights.sh` — symlinks under `weights/`.
- *(no `sync_datasets.sh` exists yet)* — if the task needs `data/datasets/`
  populated, symlink by hand from `/data/data-share/BioM3-data-share/data/datasets/`
  or ask the user whether to add a sync script.

For most webapp work you won't need `data/datasets/` or `data/databases/`
populated — the pages usually consume `outputs/`, `weights/`, or `tests/_data/`.
Check first; sync only if the specific task reads from `data/`.

### Running the app

The conda env's editable install may point at a sibling worktree rather than
the current tree — check with `conda run -n biom3-env python -c "import biom3; print(biom3.__file__)"`. Two ways to run against the current tree:

```
# A — override path for this invocation (non-destructive)
PYTHONPATH=src conda run -n biom3-env streamlit run src/biom3/app/__init__.py

# B — repoint the editable install at the current tree (one-time, affects sibling worktrees)
conda run -n biom3-env pip install -e .
```

Ask the user before doing (B).

### Testing

- **Syntax + import smoke** for every page (catches stale imports, bare-mode `st.stop()` issues):
  ```
  PYTHONPATH=src conda run -n biom3-env python -c "
  import importlib.util, sys
  from pathlib import Path
  for p in sorted(Path('src/biom3/app/pages').glob('*.py')):
      spec = importlib.util.spec_from_file_location(p.stem, p)
      mod = importlib.util.module_from_spec(spec)
      sys.modules[p.stem] = mod
      try: spec.loader.exec_module(mod); print('OK  ', p.name)
      except SystemExit: print('OK  ', p.name, '(stop)')
      except Exception as e: print('FAIL', p.name, type(e).__name__, str(e)[:200])
  " 2>/dev/null
  ```
- **Viz tests** — `PYTHONPATH=src conda run -n biom3-env pytest tests/viz_tests/ -q`. Any new `viz/` code should have tests here.
- **Browser test** is mandatory for UI changes per the repo's CLAUDE.md. Type-checking and smoke-imports verify code correctness, not feature correctness. If you can't run a browser, say so explicitly rather than claiming success.

### Commits

- Conventional Commits: `feat(app):`, `feat(viz):`, `fix(app):`, `docs:`, `test(viz):`, etc.
- One commit per feature — easier to review, cherry-pick, revert.
- Include the `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer (or whichever model you are).
- Never commit without the user asking.

### Session end

Write a session note to `docs/.claude_sessions/<YYYY-MM-DD>_<slug>.md`
matching the format of `2026-04-18_webapp_viz_expansion.md`:

- Summary (1-2 paragraphs on what was done and why).
- Commits list.
- Files changed (grouped by file, one short paragraph each).
- Design notes (invariants, patterns, gotchas).
- Follow-ups considered but deferred.

This is what makes the next session bootstrap-able via this prompt.

## First action

After loading the above, **confirm to the user that context is loaded and
ask what they want to build**. Do not start coding or run the smoke test
until they've stated a goal — the prompt is for orientation, not autopilot.
