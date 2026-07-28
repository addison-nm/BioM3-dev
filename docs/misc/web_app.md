# BioM3 Web App

`biom3.app` is a browser-based interface for protein structure analysis, built
with [Streamlit](https://streamlit.io/) on top of the `biom3.viz` library. It
provides interactive 3D viewers, structural alignment, BLAST search, and
diffusion unmasking visualization without writing any code.

## Installation

The web app requires Streamlit in addition to the base `biom3.viz` dependencies:

```bash
pip install py3Dmol streamlit
```

Optional dependencies for additional features:

| Dependency | Install command | Required for |
|------------|-----------------|-------------|
| `omegaconf` | `pip install 'fair-esm[esmfold]' omegaconf` | Fold sequence (ESMFold) on the View Structure page |

All other dependencies (BioPython, matplotlib, numpy, torch) are already part
of the base BioM3 install.

## Launching the app

Two options:

```bash
# Direct launch
streamlit run src/biom3/app/__init__.py

# Via entry point (after pip install -e .)
biom3_app
```

The app opens in your default browser at `http://localhost:8501`. Pages are
listed in the sidebar.

## Pages

| # | Page | Description |
|---|------|-------------|
| 1 | [View Structure](#view-structure) | Browse, upload, paste, or fold a PDB structure and view it interactively |
| 2 | [Align Structures](#align-structures) | Superimpose two structures on C-alpha atoms, view overlay and RMSD |
| 3 | [Highlight Residues](#highlight-residues) | Select residues by number and highlight them with custom colors and styles |
| 4 | [Color by Values](#color-by-values) | Map per-residue float values (pLDDT, conservation, etc.) onto a structure |
| 5 | [Unmasking Order](#unmasking-order) | Color a structure by Stage 3 diffusion generation order |
| 6 | [BLAST Search](#blast-search) | Run remote BLAST against NCBI and browse alignment results |

### View Structure

Browse a PDB from configured data directories, upload a `.pdb` file, paste PDB
text directly, or enter an amino acid sequence to fold with ESMFold. Controls
let you switch between rendering styles (`cartoon`, `stick`, `sphere`, `line`)
and color schemes (`spectrum`, `chain`, `ssJmol`, `residue`).

ESMFold requires a GPU and the optional `omegaconf` dependency. If unavailable,
the other input methods still work.

### Align Structures

Select a reference (fixed) PDB and a mobile (moving) PDB — either by browsing
data directories or uploading files. Click **Superimpose** to align them on
C-alpha atoms. The page displays:

- RMSD in angstroms
- Number of paired CA atoms
- An interactive overlay with the reference in blue and the mobile in red

### Highlight Residues

Select a PDB and enter comma-separated residue numbers (1-based PDB numbering).
Choose a highlight color, highlight style (`stick`, `sphere`, `line`), and
background style. The viewer renders the full structure with the selected
residues emphasized.

### Color by Values

Select a PDB and enter comma-separated per-residue float values (one value per
residue). Choose a matplotlib colormap (`coolwarm`, `bwr`, `viridis`, `plasma`,
`RdYlGn`, `RdBu`). Values are auto-normalized to [0, 1] and mapped onto the
structure.

Useful for visualizing pLDDT confidence scores, conservation, B-factors, or
any other per-residue metric.

### Unmasking Order

Select a PDB structure and a Stage 3 `.pt` file containing either:

- **Animation frames** -- the `animation_frames` dict saved during generation
  (keys are `(prompt_idx, replica_idx)` tuples, values are lists of per-step
  token index arrays)
- **Sampling path** -- the random permutation tensor (`batch_perms`) from
  generation

Both PDB and `.pt` files can be browsed from data directories or uploaded. The
structure is colored on a gradient from blue (earliest unmasked) to red (latest
unmasked), showing the order in which ProteoScribe generated each residue.

### BLAST Search

Enter a protein sequence (single-letter amino acid codes) and configure the
search:

- **Database**: `nr`, `swissprot`, `pdb`, `refseq_protein`
- **Max hits**: 1--50
- **E-value threshold**: adjustable

Click **Run BLAST** to query NCBI remotely (typically 30--60 seconds). Results
are displayed as expandable cards showing hit ID, percent identity, E-value,
score, and the full pairwise alignment.

## Data browser

Every file input in the app offers two modes: **Browse data** (select from
configured directories) or **Upload file**. This lets you work directly with
files on disk without uploading them through the browser.

### Configuration

Browsable directories are defined in `configs/app_data_dirs.json`:

```json
{
  "data_dirs": [
    {"label": "Outputs", "path": "outputs/"},
    {"label": "Weights", "path": "weights/"},
    {"label": "Test Data", "path": "tests/_data/"}
  ]
}
```

Each entry needs a `label` (displayed in the dropdown) and a `path` (directory
to scan, relative to the project root or absolute). Files are listed
recursively. To add a new directory, just add another entry to the array.

## Adding new pages

To add a new analysis page:

1. Create a file in `src/biom3/app/pages/` with a numbered prefix:
   ```
   7_My_Analysis.py
   ```
   The number controls sidebar order. Underscores become spaces in the UI
   (`7_My_Analysis` appears as "My Analysis").

2. Use shared helpers from `biom3.app._helpers`:
   ```python
   import streamlit as st
   from biom3.app._helpers import render_view, pick_pdb

   st.header("My Analysis")
   pdb_data = pick_pdb(key="my_pdb")
   if pdb_data:
       from biom3.viz import view_pdb
       v = view_pdb(pdb_data)
       render_view(v)
   ```

3. Streamlit auto-discovers the file -- no registration or wiring needed.

### Available helpers

| Helper | Description |
|--------|-------------|
| `render_view(view, height=500)` | Embed a `py3Dmol.view` in the Streamlit page as an HTML component |
| `pick_pdb(label, key)` | Browse-or-upload widget for PDB files, returns string content or `None` |
| `pick_pt(label, key)` | Browse-or-upload widget for `.pt` files, returns `Path` or `UploadedFile` or `None` |
| `pick_file(label, extensions, upload_types, key, read_text)` | Generic browse-or-upload widget |
| `load_pt(file_or_path)` | Load a `.pt` file from either a `Path` or Streamlit `UploadedFile` |
| `upload_pdb(label, key)` | Upload-only widget for PDB files (legacy, prefer `pick_pdb`) |

## Architecture

```
src/biom3/
    viz/                    # Python library (no Streamlit dependency)
        viewer.py           #   3D rendering, HTML export
        alignment.py        #   structural alignment, BLAST
        unmasking.py        #   diffusion order extraction
        folding.py          #   ESMFold integration (optional)
        _tokens.py          #   token vocabulary, color constants

    app/                    # Web app (depends on viz + streamlit)
        __init__.py         #   Landing page, main() entry point
        _helpers.py         #   Shared Streamlit utilities
        _data_browser.py    #   Data directory config loader + file browser widget
        pages/              #   One file per page (auto-discovered)
            1_View_Structure.py
            2_Align_Structures.py
            3_Highlight_Residues.py
            4_Color_by_Values.py
            5_Unmasking_Order.py
            6_BLAST_Search.py

configs/
    app_data_dirs.json      # Browsable data directory configuration
```

`viz` is a pure Python library with no web framework dependency -- it works in
notebooks, scripts, and tests. `app` is a thin Streamlit layer that provides a
browser UI on top of `viz`. New pages only need to import from `viz` and
`app._helpers`.

## See also

- `docs/structure_visualization.md` -- `biom3.viz` library reference (Python API)
- `docs/sequence_generation_animation.md` -- GIF animation of the denoising process
- `src/biom3/app/` -- app source code
- `src/biom3/viz/` -- visualization library source code
