# Webapp visualization expansion

**Date:** 2026-04-18
**Branch:** `addison-dev-app` (created off `addison-dev`, merged from `worktree-webapp-dropdown-improvements`)

## Summary

Three rounds of improvements to the Streamlit webapp at `src/biom3/app/`, all layered on top of existing `viz/` and `Stage3.animation_tools` code — no new viz algorithms.

1. **Data browser UX** — flat recursive file selectbox replaced with a hierarchical picker (data directory → subfolder scope → file) plus a substring filter, so `outputs/`-style directories with nested per-run structure are usable.
2. **Surface existing viz in the UI** — two new pages wire up code that was already built but never exposed, and several pages gain UX on top of their existing viz calls.
3. **BLAST-driven structure alignment** — new page where a query (PDB or sequence) is BLASTed and each hit can be aligned to the query after fetching a structure from RCSB / AlphaFoldDB or folding via ESMFold.

Net: `src/biom3/app/pages/` grew from 6 pages to 9; one new `viz/` module; one new `_helpers.py` utility.

## Commits on this branch (merged via `951c9fd`)

```
42a09f3 feat(app): hierarchical directory/file picker with substring filter
12ab00a feat(app): add side-by-side pre-alignment view on Align Structures
d8f6b4a feat(app): add Probability Dynamics page
575bad5 feat(app): add Sequence Generation Animation page (GIF + MP4)
221bf4e feat(app): show generated AA sequence on Unmasking page
d1d40a2 feat(app): BLAST hit table, colored alignment, CSV, fold-a-hit
654d8fa feat(app): color-by-B-factor/pLDDT toggle on View Structure
98c4d24 feat(viz): add sources module for RCSB/AlphaFoldDB fetch + hit-ID parsers
6b8a6ad feat(app): add BLAST + Structure Alignment page
```

## Files changed

### `src/biom3/app/_data_browser.py`
`browse_file` now renders three widgets in series: **data directory** → **subfolder scope** → **file**, with a substring filter between scope and file. Scope options are `(root only)`, `(all, recursive)`, and each immediate subdirectory. File paths display relative to the chosen scope (not the root), so subfolder picks get short names. All widget keys are stable so selections persist in `st.session_state` across reruns. Public signature of `browse_file` is unchanged — every existing caller inherits this for free.

### `src/biom3/app/_helpers.py`
New `render_colored_sequence(seq, colors=None, label="", wrap=60, show_positions=True)`: builds monospace HTML with per-character background colors (auto-picks black/white foreground from luminance), wrapped at `wrap` chars with optional position numbers in the gutter. Used by page 5 (generated AA sequence) and page 6 (BLAST alignment block).

### `src/biom3/app/pages/1_View_Structure.py`
New "Color by B-factor / pLDDT" checkbox. Parses CA-atom B-factors via `Bio.PDB.PDBParser` and applies the existing `viz.viewer.color_by_values` with a user-selectable colormap (RdYlGn default for pLDDT). No changes to `viz/viewer.py`.

### `src/biom3/app/pages/2_Align_Structures.py`
Adds a "Pre-alignment view" section that renders the reference and mobile PDBs in two columns (`st.columns(2)`) before the Superimpose button. Useful sanity check on inputs prior to alignment.

### `src/biom3/app/pages/5_Unmasking_Order.py`
After the 3D structure view, extracts the final-frame token indices (when available from animation frames) and renders the generated amino-acid sequence below the structure using `render_colored_sequence` with `AA_COLORS` from `Stage3.animation_tools` (Clustal-style palette). Sampling-path-only mode doesn't carry token identities so the sequence is omitted in that case.

### `src/biom3/app/pages/6_BLAST_Search.py`
- Sortable hit summary rendered as a `pd.DataFrame` at the top via `st.dataframe`, with columns `hit_id`, `percent_identity`, `e_value`, `score`, `align_length`, `identities`, `positives`, `hit_def`.
- `st.download_button` for CSV export of the full hit set (all `BlastResult` fields).
- Per-hit expander now shows a three-row colored alignment: query / midline / subject rendered via `render_colored_sequence`, with per-column background determined by the midline char (`isalpha()` → green match, `+` → amber positive, `-` in either row → gray gap, else light red mismatch).
- "Fold this hit with ESMFold" button on every hit. Strips gaps from `hit_seq`, calls `viz.folding.fold_sequence`, renders the resulting structure inline. Import-guarded so ESMFold-less environments fail gracefully.
- Hits and folded structures are persisted in `st.session_state` keyed by hit_id so selections survive reruns.

### `src/biom3/app/pages/7_Probability_Dynamics.py` *(new)*
Wires `viz.dynamics.plot_probability_dynamics` and `plot_probability_dynamics_from_file` (implemented back in `2026-04-07_pad_gauge_and_dynamics_plot.md`, never exposed). Accepts `.npz` (from `run_ProteoScribe_sample`) or `.pt` dicts containing `probs` / `tokens` / `frames` / `final_frame`. For nested `(prompt_idx, replica_idx)` dicts, adds a selectbox to pick a replica before plotting. Checkboxes for `hide_pad` and `blank_unmasked`.

### `src/biom3/app/pages/8_Generation_Animation.py` *(new)*
Wires `Stage3.animation_tools.generate_sequence_animation` (previously unexposed). Loads a frames `.pt` (with optional `(prompt, replica)` nesting), optionally a separate probabilities `.pt`/`.npz`, and exposes controls for `prob_style` (none / brightness / colorbar / logo), `cols_per_row`, frame `duration`, `title`, and a `confidence_metric` metric-row toggle. Writes the GIF to a tempfile, renders with `st.image`, offers download. "Convert to MP4" appears after a GIF exists; calls `gif_to_mp4` and surfaces the `imageio-ffmpeg`-missing error message if ffmpeg is not available.

### `src/biom3/app/pages/9_BLAST_Alignment.py` *(new)*
The core user-requested feature. Takes a query as uploaded PDB, pasted PDB, or sequence (folded with ESMFold). Runs `blast_sequence` on the query sequence. For each hit:
- Parses `hit_id` / `hit_def` for a PDB accession and/or UniProt accession via the new `viz.sources` parsers.
- Presents a per-hit structure-source selectbox that auto-populates sensible options: `Fetch RCSB (XXXX)` when a PDB ID was parsed, `Fetch AlphaFoldDB (PXXXXX)` when a UniProt accession was parsed, always `Fold with ESMFold (hit sequence)` as fallback.
- "Get structure + align" button fetches/folds, then calls `superimpose(query, hit_structure)`, shows `RMSD` and `Paired CA atoms` metrics, and renders the overlay via `view_overlay` with labels and per-structure colors.
- Hits, fetched structures, and alignment results are cached in `st.session_state`.

### `src/biom3/viz/sources.py` *(new)*
- `fetch_rcsb_pdb(pdb_id, timeout=30.0) -> str` — `urllib` download from RCSB.
- `fetch_alphafold(uniprot_id, timeout=30.0) -> str` — download from AlphaFoldDB (model v4, fragment F1). 404 gives a clear "no model for X" error.
- `parse_pdb_id(hit_id, hit_def="")` — extract 4-char PDB code from `pdb|XXXX`, `XXXX_chain`, and `gi|N|pdb|XXXX` formats.
- `parse_uniprot_id(hit_id, hit_def="")` — extract UniProt accession from `sp|…` / `tr|…` formats, validating against canonical UniProt accession regex.
- `pdb_to_sequence(pdb_str) -> str` — longest peptide sequence via `Bio.PDB.PPBuilder`.
- stdlib + BioPython only; no new project dependencies.

### `tests/viz_tests/test_sources.py` *(new)*
14 unit tests covering parser behavior across common NCBI BLAST ID formats. No network; fetchers are not unit-tested (documented as integration/network-dependent).

## Design notes

- **Reuse over rebuild.** Every visualization function exists already under `viz/` or `Stage3.animation_tools`. Two capabilities (`plot_probability_dynamics`, `generate_sequence_animation`) were fully implemented and tested but had no UI entry point; surfacing them was pure wire-up.
- **Session-state persistence.** Any computation that's expensive (BLAST: ~30-60s; ESMFold: seconds on GPU, minutes on CPU; remote fetches) lives in `st.session_state`, keyed so results survive reruns and don't recompute on every widget interaction. BLAST hits in particular are cached by query rather than by hit_id so a new BLAST run invalidates the prior alignments cleanly.
- **Graceful degradation.** ESMFold is optional; every call site import-guards and surfaces the same install instruction (`pip install 'fair-esm[esmfold]' omegaconf`). `gif_to_mp4` already raises a helpful `RuntimeError` when `imageio-ffmpeg` is missing, so callers just need a `try/except RuntimeError`.
- **Per-hit structure source is user-chosen.** On page 9, the dropdown auto-populates the most likely source based on parsed accessions, but the user always picks. The alternative (auto-fetch-then-fold-fallback) was considered and rejected for this iteration in favor of explicit control.
- **The conda env's editable install is pinned to a sibling worktree.** Running `pytest` or `streamlit run` against the main tree's code requires either a fresh `pip install -e .` or a `PYTHONPATH=src` prefix. Flagged to the user; left unchanged.

## Follow-ups considered but not done

- **Multi-hit stacked overlay on page 9** — single viewer with query + N aligned hits color-coded with a legend. Cheap extension once page 9 exists; can add when useful.
- **Smart structure picker everywhere** — upgrade `pick_pdb` to also offer "Fetch by PDB ID" / "Fetch by UniProt", so pages 1–5 all inherit fetch-by-accession. Not done this round.
- **Chain picker for multi-chain PDBs** — `pdb_to_sequence` currently returns only the longest peptide. Multi-chain workflows would benefit from a chain selector.
- **Length-aware RMSD / TM-score** — page 9's RMSD is raw CA superposition; for distantly related hits that's noisy. A TM-score or length-normalized metric would make the ranking meaningful.
- **"Align all visible hits" batch button on page 9** — currently one click per hit. Matches the user's preference for explicit per-hit control but trivial to add later.
