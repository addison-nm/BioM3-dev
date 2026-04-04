"""Shared Streamlit helpers for BioM3 app pages."""

import streamlit as st
import streamlit.components.v1 as components

from biom3.viz.viewer import to_html

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
