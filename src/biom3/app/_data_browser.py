"""Data directory browser for the BioM3 Streamlit app.

Reads allowed directories from a JSON config and provides Streamlit widgets
for browsing and selecting files.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "app_data_dirs.json"


@st.cache_data
def _load_config(config_path: str) -> list[dict]:
    p = Path(config_path)
    if not p.is_file():
        return []
    with open(p) as f:
        data = json.load(f)
    return data.get("data_dirs", [])


def get_data_dirs(config_path: str | Path | None = None) -> list[dict]:
    """Return the list of configured data directories.

    Each entry is {"label": str, "path": str}.
    """
    path = str(config_path or DEFAULT_CONFIG_PATH)
    return _load_config(path)


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


def browse_file(
    label: str = "Browse data",
    extensions: list[str] | None = None,
    key: str | None = None,
) -> Path | None:
    """Streamlit widget to browse configured data directories and select a file.

    Returns the Path to the selected file, or None if nothing is selected.
    """
    dirs = get_data_dirs()
    if not dirs:
        st.info("No data directories configured. Edit `configs/app_data_dirs.json` to add directories.")
        return None

    dir_labels = [d["label"] for d in dirs]
    dir_key = f"_browse_dir_{key}" if key else "_browse_dir"
    selected_label = st.selectbox("Data directory", dir_labels, key=dir_key)

    selected_dir = next(d["path"] for d in dirs if d["label"] == selected_label)
    files = list_files(selected_dir, extensions=extensions)

    if not files:
        st.warning(f"No matching files in `{selected_dir}`")
        return None

    base = Path(selected_dir)
    display_names = []
    for f in files:
        try:
            display_names.append(str(f.relative_to(base)))
        except ValueError:
            display_names.append(str(f))

    file_key = f"_browse_file_{key}" if key else "_browse_file"
    selected_name = st.selectbox(label, display_names, key=file_key)

    idx = display_names.index(selected_name)
    return files[idx]
