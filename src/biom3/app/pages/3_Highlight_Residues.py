import streamlit as st

from biom3.viz.viewer import view_pdb, highlight_residues
from biom3.app._helpers import render_view, upload_pdb

st.header("Highlight Residues")

pdb_data = upload_pdb()
if pdb_data:
    residues_str = st.text_input(
        "Residue numbers to highlight (comma-separated, 1-based)",
        placeholder="10, 25, 42, 100",
    )
    col1, col2 = st.columns([3, 1])
    with col2:
        color = st.color_picker("Highlight color", "#FF0000")
        hl_style = st.selectbox("Highlight style", ["stick", "sphere", "line"])
        bg_style = st.selectbox("Background style", ["cartoon", "line"])
    with col1:
        v = view_pdb(pdb_data, style=bg_style)
        if residues_str.strip():
            try:
                residues = [int(r.strip()) for r in residues_str.split(",") if r.strip()]
                highlight_residues(v, residues, color=color, style=hl_style)
            except ValueError:
                st.error("Enter residue numbers as comma-separated integers.")
        render_view(v)
