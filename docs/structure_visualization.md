# Structure Visualization Library

`biom3.viz` is a Python library for interactive 3D protein structure
visualization, structural alignment, and mapping Stage 3 diffusion unmasking
order onto structures. It uses [py3Dmol](https://3dmol.csb.pitt.edu/) for
rendering and works in Jupyter notebooks (inline display), standalone HTML
pages, and as the backend for the BioM3 web app (`biom3.app`).

## Installation

py3Dmol is the only new dependency beyond the base BioM3 install:

```bash
pip install py3Dmol
```

For folding generated sequences with ESMFold (optional):

```bash
pip install 'fair-esm[esmfold]' omegaconf
```

## Quick start

All public functions are available from the top-level `biom3.viz` import:

```python
from biom3.viz import (
    view_pdb, view_overlay, highlight_residues, color_by_values,
    save_html, to_html,
    superimpose, superimpose_and_view, blast_sequence,
    view_unmasking_order, extract_unmasking_order,
)
```

## Viewing structures

### Single structure

```python
from biom3.viz import view_pdb

v = view_pdb("path/to/structure.pdb")
v.show()
```

Accepts a file path or a raw PDB string (e.g. from ESMFold). Optional
parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `style` | `"cartoon"` | py3Dmol style: `"cartoon"`, `"stick"`, `"sphere"`, `"line"` |
| `color_scheme` | `"spectrum"` | py3Dmol color scheme: `"spectrum"`, `"chain"`, `"ssJmol"`, etc. |
| `width` | `800` | Viewer width in pixels |
| `height` | `600` | Viewer height in pixels |

### Overlay multiple structures

```python
from biom3.viz import view_overlay

v = view_overlay(
    ["reference.pdb", "generated.pdb"],
    labels=["Reference", "Generated"],
    colors=["blue", "red"],
)
v.show()
```

Each structure is loaded as a separate model with its own color. Default colors
cycle through blue, red, green, orange, purple, cyan.

## Highlighting residues

### By index

```python
from biom3.viz import view_pdb, highlight_residues

v = view_pdb("structure.pdb")
highlight_residues(v, [10, 25, 42], color="red", style="stick")
v.show()
```

Residue indices use PDB numbering (1-based). You can call `highlight_residues`
multiple times on the same view with different colors to highlight different
groups:

```python
v = view_pdb("structure.pdb")
highlight_residues(v, [10, 11, 12], color="red")     # active site
highlight_residues(v, [50, 51, 52], color="yellow")   # binding region
v.show()
```

### By continuous values (heatmap)

```python
from biom3.viz import view_pdb, color_by_values

v = view_pdb("structure.pdb")
color_by_values(v, per_residue_values, colormap="coolwarm")
v.show()
```

`per_residue_values` is a list of floats (one per residue). Values are
auto-normalized to [0, 1] and mapped through any matplotlib colormap. NaN
values are colored gray. Useful for pLDDT scores, conservation, B-factors, etc.

## Structural alignment

### Superimpose two structures

```python
from biom3.viz import superimpose

result = superimpose("reference.pdb", "generated.pdb")
print(f"RMSD: {result.rmsd:.2f} A over {result.n_atoms} CA atoms")
```

`superimpose()` aligns on C-alpha atoms using `Bio.PDB.Superimposer`, pairing
residues by sequential index (truncated to the shorter chain). Returns an
`AlignmentResult` with:

| Field | Type | Description |
|-------|------|-------------|
| `rmsd` | `float` | Root-mean-square deviation in angstroms |
| `n_atoms` | `int` | Number of paired CA atoms |
| `rotation` | `np.ndarray` | 3x3 rotation matrix |
| `translation` | `np.ndarray` | 3-element translation vector |
| `fixed_pdb` | `str` | PDB string of the reference (unchanged) |
| `moving_pdb` | `str` | PDB string of the mobile structure after transformation |

### Superimpose and view in one step

```python
from biom3.viz import superimpose_and_view

view, result = superimpose_and_view(
    "reference.pdb", "generated.pdb",
    labels=["Reference", "Generated"],
)
print(f"RMSD: {result.rmsd:.2f} A")
view.show()
```

### Custom atom selection

To align on a different atom type (e.g. backbone N):

```python
result = superimpose("ref.pdb", "mobile.pdb", atom_name="N")
```

## BLAST sequence search

```python
from biom3.viz import blast_sequence

hits = blast_sequence("MKTLLILAVL...", max_hits=5)
for h in hits:
    print(f"{h.hit_id}: {h.percent_identity:.1f}% identity, E={h.e_value:.1e}")
```

Runs remote BLAST against NCBI via `Bio.Blast.NCBIWWW.qblast`. Each hit is a
`BlastResult` dataclass with:

| Field | Description |
|-------|-------------|
| `hit_id` | Accession ID |
| `hit_def` | Hit description |
| `e_value` | Expect value |
| `score` | BLAST score |
| `identities` | Number of identical residues |
| `positives` | Number of positive-scoring residues |
| `align_length` | Alignment length |
| `query_seq` | Aligned query sequence |
| `hit_seq` | Aligned hit sequence |
| `midline` | Alignment midline |
| `percent_identity` | Percent identity (identities / align_length * 100) |

Optional parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `database` | `"nr"` | BLAST database |
| `program` | `"blastp"` | BLAST program |
| `e_value` | `10.0` | E-value threshold |
| `max_hits` | `10` | Maximum number of hits |

## Folding generated sequences

Fold amino acid sequences into 3D structures using ESMFold, then pass the
result directly to any viewer function:

```python
from biom3.viz import fold_sequence, view_pdb

pdb_str = fold_sequence("MKTLLILAVL...")
v = view_pdb(pdb_str)
v.show()
```

Fold multiple sequences:

```python
from biom3.viz import fold_sequences

pdb_strings = fold_sequences(["MKTL...", "ACLW...", "GFPK..."])
```

The ESMFold model (~3B parameters) is loaded once and cached for subsequent
calls. Requires a GPU and the optional `omegaconf` dependency. If dependencies
are missing, importing `fold_sequence` raises a clear `ImportError` with
install instructions. The rest of `biom3.viz` works without it.

### End-to-end: generate, fold, align, visualize

```python
import torch
from biom3.viz import fold_sequence, superimpose_and_view, blast_sequence

# Load generated sequences from Stage 3
sequences = torch.load("outputs/generated_sequences.pt")
gen_seq = sequences["replica_0"][0]

# Fold the generated sequence
gen_pdb = fold_sequence(gen_seq)

# Align against a known reference
view, result = superimpose_and_view("reference.pdb", gen_pdb)
print(f"RMSD: {result.rmsd:.2f} A")
view.show()

# Check similarity via BLAST
hits = blast_sequence(gen_seq, max_hits=5)
for h in hits:
    print(f"  {h.hit_id}: {h.percent_identity:.1f}%")
```

## Visualizing unmasking order (Stage 3)

Stage 3 (ProteoScribe) generates sequences by progressively unmasking positions
via absorbing-state diffusion. `biom3.viz` can color a 3D structure by the
order in which each residue was generated: early-unmasked positions appear blue,
late-unmasked positions appear red.

### Using animation frames

The `batch_stage3_generate_sequences()` function in
`run_ProteoScribe_sample.py` returns `animation_frames` -- a dict mapping
`(prompt_idx, replica_idx)` to a list of numpy arrays (one per diffusion step),
each containing the token indices at that step.

```python
from biom3.viz import view_unmasking_order

# frames = animation_frames[(prompt_idx, replica_idx)]
v = view_unmasking_order("structure.pdb", mask_realization_list=frames)
v.show()
```

### Using the sampling path

For random-order unmasking, the sampling path tensor is also available. This is
the permutation that determines which position is unmasked at each step:

```python
from biom3.viz import view_unmasking_order

# sampling_path shape: [seq_len], from batch_perms during generation
v = view_unmasking_order("structure.pdb", sampling_path=sampling_path)
v.show()
```

### Extracting the order for custom use

```python
from biom3.viz import extract_unmasking_order

order = extract_unmasking_order(frames)
# order[i] = diffusion step at which position i was unmasked
# e.g. order = [0, 42, 7, 100, ...] means position 0 was in the first step,
#      position 1 was unmasked at step 42, etc.
```

The returned array has shape `[seq_len]`. Use `unmasking_order_to_normalized()`
to map it to [0, 1] for colormaps, or feed it into `color_by_values()` for
custom rendering.

## HTML export

Any view can be exported as a standalone HTML page that loads the 3Dmol.js
library from CDN. No Python runtime is needed to open the page.

```python
from biom3.viz import view_pdb, save_html, to_html

v = view_pdb("structure.pdb")

# Save to file
save_html(v, "output.html", title="My Structure")

# Or get the HTML string directly
html = to_html(v, title="My Structure")
```

## Chaining and customization

All `view_*` and `highlight_*` functions return the `py3Dmol.view` object, so
calls can be chained. You can also use the full py3Dmol API for further
customization:

```python
v = view_pdb("structure.pdb")
highlight_residues(v, [10, 20], color="red")
highlight_residues(v, [30, 40], color="blue")

# Direct py3Dmol API
v.addLabel("Active Site", {"fontSize": 14, "fontColor": "white"},
           {"resi": 10})
v.setBackgroundColor("0xeeeeee")
v.show()
```

## Module reference

| Module | Contents |
|--------|----------|
| `biom3.viz.viewer` | `view_pdb`, `view_overlay`, `highlight_residues`, `color_by_values`, `to_html`, `save_html` |
| `biom3.viz.alignment` | `superimpose`, `superimpose_and_view`, `blast_sequence`, `AlignmentResult`, `BlastResult` |
| `biom3.viz.unmasking` | `view_unmasking_order`, `extract_unmasking_order`, `extract_unmasking_order_from_sampling_path`, `unmasking_order_to_normalized` |
| `biom3.viz.folding` | `fold_sequence`, `fold_sequences` (optional, requires `omegaconf`) |
| `biom3.viz._tokens` | `TOKENS`, `TOKEN_TO_IDX`, `MASK_IDX`, `STANDARD_AA`, `AA_COLORS` |

## See also

- `docs/web_app.md` -- BioM3 web app (browser-based UI built on this library)
- `docs/sequence_generation_animation.md` -- GIF animation of the denoising process
- `src/biom3/Stage3/animation_tools.py` -- 2D grid animation utilities
- `src/biom3/viz/` -- source code
- `tests/viz_tests/` -- test suite
