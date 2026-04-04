# 2026-04-04: Visualization Subpackage and Web App

## Summary

Created `biom3.viz`, a new subpackage for interactive 3D protein structure
visualization, and `biom3.app`, a Streamlit multi-page web app that provides
a browser-based interface on top of the visualization library. This session
spanned two conversations (initial code written 2026-04-03, tests validated
and app built 2026-04-04).

## Pre-session state

```bash
git checkout 846dc15   # feat: add confidence_no_pad unmasking order
```

## Changes

### New: `biom3.viz` — Visualization library (`src/biom3/viz/`)

A pure Python library for protein structure visualization using py3Dmol. Works
in Jupyter notebooks, scripts, and as the backend for the web app.

| File | Purpose |
|------|---------|
| `__init__.py` | Public API re-exports |
| `_tokens.py` | Canonical 29-token vocabulary, re-exports `AA_COLORS` from `animation_tools.py` |
| `viewer.py` | py3Dmol rendering: `view_pdb`, `view_overlay`, `highlight_residues`, `color_by_values`, HTML export |
| `alignment.py` | `Bio.PDB.Superimposer` structural alignment + remote BLAST via BioPython |
| `unmasking.py` | Extract Stage 3 unmasking order from diffusion frames or sampling path, color structure by generation order |
| `folding.py` | ESMFold sequence-to-structure (optional, graceful skip if `omegaconf` missing) |

**Key design decisions:**
- py3Dmol chosen as the 3D renderer (user preference over PyMOL or static matplotlib)
- BioPython only for alignment/BLAST (already a dependency)
- ESMFold integration is optional — `folding.py` import is wrapped in try/except
- All viewer functions return `py3Dmol.view` for chaining
- `_read_pdb()` and `_parse_pdb()` distinguish file paths from PDB strings by checking for newlines (avoids `OSError` on long strings passed to `Path.is_file()`)
- Uses `matplotlib.colormaps[name]` instead of deprecated `cm.get_cmap()`

### New: `biom3.app` — Streamlit web app (`src/biom3/app/`)

Multi-page Streamlit app that provides a browser UI for all `biom3.viz`
capabilities.

| File | Purpose |
|------|---------|
| `__init__.py` | Landing page + `main()` entry point for `biom3_app` CLI |
| `_helpers.py` | Shared utilities: `render_view()`, `upload_pdb()` |
| `pages/1_View_Structure.py` | Upload/paste/fold PDB, view with style/color controls |
| `pages/2_Align_Structures.py` | Superimpose two PDBs, show RMSD and overlay |
| `pages/3_Highlight_Residues.py` | Highlight selected residues with custom colors |
| `pages/4_Color_by_Values.py` | Map per-residue floats onto structure via colormap |
| `pages/5_Unmasking_Order.py` | Color by Stage 3 diffusion generation order |
| `pages/6_BLAST_Search.py` | Remote BLAST search with expandable results |

Initially built as a monolithic `viz/app.py`, then refactored into the
multi-page `app/` package for extensibility. New pages are auto-discovered
by Streamlit — just drop a numbered `.py` file in `pages/`.

### New: Test suite (`tests/viz_tests/`)

| File | Coverage |
|------|----------|
| `conftest.py` | Minimal PDB fixtures (5-residue alanine chain, shifted variant) |
| `test_viewer.py` | `_read_pdb`, `view_pdb`, `highlight_residues`, `color_by_values`, HTML export |
| `test_alignment.py` | `superimpose` (identical → RMSD 0, shifted �� RMSD 0 after alignment), BLAST (network-skipped) |
| `test_folding.py` | Import error handling (mocked), model caching, GPU fold (skipped) |
| `test_unmasking.py` | `extract_unmasking_order`, `extract_unmasking_order_from_sampling_path`, normalization |

24 tests pass, 2 expected skips (`@pytest.mark.network` for BLAST,
`@pytest.mark.use_gpu` for ESMFold).

### New: Documentation (`docs/`)

- `docs/structure_visualization.md` — `biom3.viz` library reference (Python API, all functions documented with examples)
- `docs/web_app.md` — `biom3.app` web interface (install, launch, page descriptions, how to add pages, architecture)

### Modified files

| File | Change |
|------|--------|
| `requirements.txt` | Added `py3Dmol` |
| `pyproject.toml` | Added `biom3_app` entry point |
| `tests/conftest.py` | Added `--network` CLI option and `pytest.mark.network` marker |
| `tests/test_imports.py` | Added `test_viz_imports()` |

## New dependencies

| Package | Required by | Install |
|---------|-------------|---------|
| `py3Dmol` | `biom3.viz` | `pip install py3Dmol` |
| `streamlit` | `biom3.app` | `pip install streamlit` |
| `omegaconf` | `biom3.viz.folding` (optional) | `pip install omegaconf` |

## Bugs fixed during implementation

1. **`_read_pdb` / `_parse_pdb` OSError** — `Path(pdb_string).is_file()` throws
   `OSError: File name too long` when the PDB string content is passed instead
   of a path. Fixed by checking `"\n" not in pdb` before calling `is_file()`.

2. **`cm.get_cmap` deprecation** — matplotlib 3.10 dropped `cm.get_cmap()`.
   Replaced with `matplotlib.colormaps[name]`.

## Not implemented / future work

- **Streamlit not yet installed** in biom3-env — user will install separately.
  The app code is written but has not been launched/tested in a browser yet.
- **ESMFold folding** not tested end-to-end (requires GPU + `omegaconf`).
  Import error paths are tested via mocks.
- **Local BLAST** support was discussed but not implemented — only remote BLAST
  via `Bio.Blast.NCBIWWW.qblast` is available. Could add local BLAST via
  `NcbiblastpCommandline` if needed.
- The `viz/app.py` monolith was deleted after refactoring into `app/`. If
  git history is needed, it existed briefly in this session's working tree.
