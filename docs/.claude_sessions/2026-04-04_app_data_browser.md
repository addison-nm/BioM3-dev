# 2026-04-04: App Data Browser

## Summary

Added a file browser to the BioM3 Streamlit web app, allowing users to browse
files from configured data directories alongside the existing upload widgets.
Refactored all pages to use a unified "Browse data / Upload file" toggle.

## Pre-session state

```bash
git checkout 3e8c433   # feat: add biom3.viz visualization library and biom3.app web interface
```

Note: commit `d310388` (docs: Stage 3 generation strategies) was also present
but is unrelated to this work.

## Changes

### New: Data browser config and module

- `configs/app_data_dirs.json` -- JSON config defining browsable directories.
  Ships with `outputs/`, `weights/`, and `tests/_data/` as defaults.
- `src/biom3/app/_data_browser.py` -- loads config, recursively scans
  directories, provides a `browse_file()` Streamlit widget with directory
  dropdown and file selector.

### Updated: `_helpers.py`

Added unified browse-or-upload widgets:

| Function | Purpose |
|----------|---------|
| `pick_file()` | Generic widget: radio toggle between browse and upload, returns content or Path |
| `pick_pdb()` | Convenience wrapper for PDB files (returns string content) |
| `pick_pt()` | Convenience wrapper for `.pt` files (returns Path or UploadedFile) |
| `load_pt()` | Loads `.pt` from either Path or UploadedFile (handles temp file for uploads) |

The original `upload_pdb()` is retained for backwards compatibility.

### Updated: Pages 1--5

All pages that accept file inputs now use `pick_pdb()` or `pick_pt()` instead
of `upload_pdb()` or raw `st.file_uploader()`. Each file input shows a
"Browse data / Upload file" radio toggle.

Page 6 (BLAST Search) is unchanged -- it only accepts text input.

### Updated: `docs/web_app.md`

- Added "Data browser" section explaining config format and usage
- Updated page descriptions to reflect browse-or-upload capability
- Updated "Adding new pages" example to use `pick_pdb()` instead of `upload_pdb()`
- Updated helpers table with all new functions
- Updated architecture diagram to include `_data_browser.py` and config file

## Files changed

| File | Change |
|------|--------|
| `configs/app_data_dirs.json` | **New** |
| `src/biom3/app/_data_browser.py` | **New** |
| `src/biom3/app/_helpers.py` | **Edit** -- added pick/load helpers |
| `src/biom3/app/pages/1_View_Structure.py` | **Edit** -- use `pick_pdb` |
| `src/biom3/app/pages/2_Align_Structures.py` | **Edit** -- use `pick_pdb` |
| `src/biom3/app/pages/3_Highlight_Residues.py` | **Edit** -- use `pick_pdb` |
| `src/biom3/app/pages/4_Color_by_Values.py` | **Edit** -- use `pick_pdb` |
| `src/biom3/app/pages/5_Unmasking_Order.py` | **Edit** -- use `pick_pdb`/`pick_pt`/`load_pt` |
| `docs/web_app.md` | **Edit** -- data browser docs |

## Tests

All 24 existing viz tests pass (2 expected skips). No new tests added for the
app layer (Streamlit widgets require a running app context to test).

## Notes

- The data browser scans directories recursively and shows all file types. Pages
  pass extension filters to `browse_file()` so only relevant files appear.
- The config uses relative paths by default (`outputs/`, etc.). Absolute paths
  are also supported.
- Streamlit is not yet installed in the biom3-env environment. The app code is
  written but has not been browser-tested yet.
