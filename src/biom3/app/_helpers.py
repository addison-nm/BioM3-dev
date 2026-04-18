"""Shared Streamlit helpers for BioM3 app pages."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from biom3.viz.viewer import to_html
from biom3.app._data_browser import browse_file

VIEWER_HEIGHT = 500


def render_view(view, height=VIEWER_HEIGHT):
    """Embed a py3Dmol view in the Streamlit page."""
    html = to_html(view)
    components.html(html, height=height, scrolling=False)


def upload_pdb(label="Upload PDB file", key=None):
    """File uploader that returns PDB string content or None."""
    f = st.file_uploader(label, type=["pdb", "ent"], key=key)
    if f is not None:
        return f.read().decode("utf-8")
    return None


def pick_file(
    label: str = "Select file",
    extensions: list[str] | None = None,
    upload_types: list[str] | None = None,
    key: str = "pick",
    read_text: bool = False,
) -> str | Path | None:
    """Unified widget: browse data directories or upload a file.

    Parameters
    ----------
    label : str
        Descriptive label shown to the user.
    extensions : list[str], optional
        Extensions for the data browser filter (e.g. [".pdb", ".ent"]).
    upload_types : list[str], optional
        Extensions for the upload widget (without dots, e.g. ["pdb", "ent"]).
    key : str
        Unique key prefix for Streamlit widgets.
    read_text : bool
        If True, return the file contents as a string (for text files like PDB).
        If False, return the Path object (for binary files like .pt).

    Returns
    -------
    str, Path, or None
        If read_text=True: file contents as str, or None.
        If read_text=False: Path to the file, or None.
    """
    source = st.radio(
        label,
        ["Browse data", "Upload file"],
        horizontal=True,
        key=f"{key}_source",
    )

    if source == "Browse data":
        path = browse_file(
            label="Select file",
            extensions=extensions,
            key=key,
        )
        if path is None:
            return None
        if read_text:
            return path.read_text()
        return path

    else:
        f = st.file_uploader(
            "Upload",
            type=upload_types,
            key=f"{key}_upload",
        )
        if f is None:
            return None
        if read_text:
            return f.read().decode("utf-8")
        # For binary files, return the UploadedFile object directly
        return f


def pick_pdb(label: str = "PDB structure", key: str = "pdb") -> str | None:
    """Convenience wrapper for picking a PDB file (browse or upload).

    Returns PDB content as a string, or None.
    """
    return pick_file(
        label=label,
        extensions=[".pdb", ".ent"],
        upload_types=["pdb", "ent"],
        key=key,
        read_text=True,
    )


def pick_pt(label: str = "PyTorch file", key: str = "pt"):
    """Convenience wrapper for picking a .pt file (browse or upload).

    Returns a Path (if browsed) or UploadedFile (if uploaded), or None.
    """
    return pick_file(
        label=label,
        extensions=[".pt"],
        upload_types=["pt"],
        key=key,
        read_text=False,
    )


def render_colored_sequence(
    seq: str,
    colors: list[tuple[int, int, int]] | None = None,
    label: str = "",
    wrap: int = 60,
    show_positions: bool = True,
) -> str:
    """Build HTML for a monospace sequence row with per-character background colors.

    Parameters
    ----------
    seq : str
        Sequence characters to render.
    colors : list of (r, g, b) tuples or None
        Background color per character. Must match ``len(seq)`` if given.
    label : str
        Row label shown at the start of each wrapped line.
    wrap : int
        Characters per line.
    show_positions : bool
        Whether to prefix each line with the 1-based start position.

    Returns
    -------
    str
        HTML string suitable for ``st.markdown(html, unsafe_allow_html=True)``.
    """
    if colors is not None and len(colors) != len(seq):
        raise ValueError(
            f"colors length {len(colors)} does not match seq length {len(seq)}"
        )

    def _span(ch: str, rgb: tuple[int, int, int] | None) -> str:
        if rgb is None:
            return f'<span>{ch}</span>'
        r, g, b = rgb
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        fg = "#000" if lum > 140 else "#fff"
        return (
            f'<span style="background-color:rgb({r},{g},{b});color:{fg};'
            f'padding:0 2px;">{ch}</span>'
        )

    lines = []
    for start in range(0, len(seq), wrap):
        end = min(start + wrap, len(seq))
        prefix_parts = []
        if label:
            prefix_parts.append(
                f'<span style="color:#888;">{label:>8}</span>'
            )
        if show_positions:
            prefix_parts.append(
                f'<span style="color:#888;">{start + 1:>5}</span>'
            )
        prefix = " ".join(prefix_parts)
        body = "".join(
            _span(seq[i], colors[i] if colors is not None else None)
            for i in range(start, end)
        )
        lines.append(f"{prefix}  {body}")

    return (
        '<div style="font-family:monospace;font-size:13px;line-height:1.6;'
        'white-space:pre;">'
        + "<br>".join(lines)
        + "</div>"
    )


def load_pt(file_or_path):
    """Load a .pt file from either a Path or a Streamlit UploadedFile.

    Returns the deserialized object.
    """
    import tempfile
    import os
    import torch

    if isinstance(file_or_path, Path):
        return torch.load(file_or_path, map_location="cpu", weights_only=False)

    # UploadedFile — write to a temp file first
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp.write(file_or_path.read())
        tmp_path = tmp.name
    try:
        return torch.load(tmp_path, map_location="cpu", weights_only=False)
    finally:
        os.unlink(tmp_path)
