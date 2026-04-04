from pathlib import Path

import py3Dmol
import matplotlib
import matplotlib.colors as mcolors


def _read_pdb(pdb: str | Path) -> str:
    """Read PDB content: if it's a file path, read it; otherwise treat as PDB string."""
    if isinstance(pdb, Path):
        return pdb.read_text()
    if "\n" not in pdb:
        p = Path(pdb)
        if p.is_file():
            return p.read_text()
    return str(pdb)


def view_pdb(
    pdb: str | Path,
    style: str = "cartoon",
    color_scheme: str = "spectrum",
    width: int = 800,
    height: int = 600,
) -> py3Dmol.view:
    """Render a single PDB structure in an interactive 3D viewer."""
    data = _read_pdb(pdb)
    view = py3Dmol.view(width=width, height=height)
    view.addModel(data, "pdb")
    view.setStyle({style: {"color": color_scheme}})
    view.zoomTo()
    return view


def view_overlay(
    pdbs: list[str | Path],
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    style: str = "cartoon",
    width: int = 800,
    height: int = 600,
) -> py3Dmol.view:
    """Overlay multiple PDB structures in a single viewer."""
    default_colors = ["blue", "red", "green", "orange", "purple", "cyan"]
    if colors is None:
        colors = default_colors
    view = py3Dmol.view(width=width, height=height)
    for i, pdb in enumerate(pdbs):
        data = _read_pdb(pdb)
        view.addModel(data, "pdb")
        color = colors[i % len(colors)]
        view.setStyle({"model": i}, {style: {"color": color}})
        if labels:
            label = labels[i] if i < len(labels) else f"Model {i}"
            view.addLabel(
                label,
                {"backgroundColor": color, "fontColor": "white", "fontSize": 12},
                {"model": i, "resi": 1},
            )
    view.zoomTo()
    return view


def highlight_residues(
    view: py3Dmol.view,
    residue_indices: list[int],
    color: str = "red",
    style: str = "stick",
    model_idx: int = 0,
) -> py3Dmol.view:
    """Add highlighting to specific residues on an existing view."""
    resi_list = [str(r) for r in residue_indices]
    sel = {"model": model_idx, "resi": resi_list}
    view.addStyle(sel, {style: {"color": color}})
    return view


def color_by_values(
    view: py3Dmol.view,
    values: list[float],
    colormap: str = "bwr",
    model_idx: int = 0,
) -> py3Dmol.view:
    """Color residues by a per-residue float array using a matplotlib colormap.

    NaN values are colored gray.
    """
    import numpy as np

    arr = np.asarray(values, dtype=float)
    valid = ~np.isnan(arr)
    if valid.any():
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        if vmax > vmin:
            normalized = (arr - vmin) / (vmax - vmin)
        else:
            normalized = np.where(valid, 0.5, np.nan)
    else:
        normalized = arr

    cmap = matplotlib.colormaps[colormap]
    for i, val in enumerate(normalized):
        if np.isnan(val):
            hex_color = "#888888"
        else:
            rgba = cmap(val)
            hex_color = mcolors.to_hex(rgba)
        resi = i + 1  # PDB residue numbering is 1-based
        view.setStyle(
            {"model": model_idx, "resi": resi},
            {"cartoon": {"color": hex_color}},
        )
    return view


def to_html(view: py3Dmol.view, title: str | None = None) -> str:
    """Export a py3Dmol view to a standalone HTML string."""
    inner = view._make_html()
    page_title = title or "BioM3 Structure Viewer"
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{page_title}</title>
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <style>body {{ margin: 0; overflow: hidden; }}</style>
</head>
<body>
  {inner}
</body>
</html>"""


def save_html(
    view: py3Dmol.view,
    path: str | Path,
    title: str | None = None,
) -> None:
    """Save a py3Dmol view as a standalone HTML file."""
    html = to_html(view, title=title)
    Path(path).write_text(html)
