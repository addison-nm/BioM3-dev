"""Data directory browser for the BioM3 Streamlit app.

Reads allowed directories from app settings and provides Streamlit widgets
for browsing and selecting files.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from biom3.app.config import load_settings


def get_data_dirs() -> list[dict]:
    """Return the list of configured data directories.

    Each entry is {"label": str, "path": str}.
    """
    settings = load_settings()
    return settings.get("data_dirs", [])


def list_files(
    directory: str | Path,
    extensions: list[str] | None = None,
    recursive: bool = True,
) -> list[Path]:
    """List files in a directory, optionally filtering by extension.

    Parameters
    ----------
    directory : str or Path
        Directory to scan.
    extensions : list[str], optional
        File extensions to include (e.g. [".pdb", ".ent"]). Include the dot.
        If None, all files are returned.
    recursive : bool
        If True, search subdirectories recursively.
    """
    d = Path(directory)
    if not d.is_dir():
        return []
    if recursive:
        files = sorted(f for f in d.rglob("*") if f.is_file())
    else:
        files = sorted(f for f in d.iterdir() if f.is_file())
    if extensions:
        ext_set = {e.lower() for e in extensions}
        files = [f for f in files if f.suffix.lower() in ext_set]
    return files


_SCOPE_ROOT = "(root only)"
_SCOPE_ALL = "(all, recursive)"


def browse_file(
    label: str = "Browse data",
    extensions: list[str] | None = None,
    key: str | None = None,
) -> Path | None:
    """Streamlit widget to browse configured data directories and select a file.

    Presents a three-step picker: data directory → subfolder scope → file,
    with a substring filter on file names. Selections persist across reruns
    via Streamlit's session_state (keyed by the `key` argument).

    Returns the Path to the selected file, or None if nothing is selected.
    """
    dirs = get_data_dirs()
    if not dirs:
        st.info(
            "No data directories configured. Provide an app settings JSON via "
            "`biom3_app --config <path>` or the `BIOM3_APP_CONFIG` env var."
        )
        return None

    widget_key = key or "default"

    dir_labels = [d["label"] for d in dirs]
    selected_label = st.selectbox(
        "Data directory",
        dir_labels,
        key=f"_browse_dir_{widget_key}",
    )
    root_dir = Path(next(d["path"] for d in dirs if d["label"] == selected_label))

    if not root_dir.is_dir():
        st.warning(f"Directory `{root_dir}` does not exist")
        return None

    subfolders = sorted(p.name for p in root_dir.iterdir() if p.is_dir())
    scope_options = [_SCOPE_ROOT, _SCOPE_ALL, *subfolders]
    scope = st.selectbox(
        "Subfolder",
        scope_options,
        key=f"_browse_scope_{widget_key}",
    )

    if scope == _SCOPE_ROOT:
        search_dir, recursive = root_dir, False
    elif scope == _SCOPE_ALL:
        search_dir, recursive = root_dir, True
    else:
        search_dir, recursive = root_dir / scope, True

    filter_text = st.text_input(
        "Filter",
        key=f"_browse_filter_{widget_key}",
        placeholder="substring match (case-insensitive)",
    ).strip().lower()

    files = list_files(search_dir, extensions=extensions, recursive=recursive)

    entries: list[tuple[str, Path]] = []
    for f in files:
        try:
            name = str(f.relative_to(search_dir))
        except ValueError:
            name = str(f)
        if filter_text and filter_text not in name.lower():
            continue
        entries.append((name, f))

    if not entries:
        if filter_text:
            st.warning(f"No files match filter `{filter_text}` in `{search_dir}`")
        else:
            st.warning(f"No matching files in `{search_dir}`")
        return None

    display_names = [name for name, _ in entries]
    selected_name = st.selectbox(
        label,
        display_names,
        key=f"_browse_file_{widget_key}",
    )

    idx = display_names.index(selected_name)
    return entries[idx][1]
