import streamlit as st

from biom3.viz.viewer import view_pdb, color_by_values
from biom3.app._helpers import render_view, pick_pdb

st.header("Color by Per-Residue Values")
st.write("Select a PDB and provide per-residue float values (e.g. pLDDT, conservation scores).")

pdb_data = pick_pdb(key="cbv_pdb")
if pdb_data:
    values_str = st.text_area(
        "Per-residue values (comma-separated floats, one per residue)",
        placeholder="0.95, 0.87, 0.45, 0.92, ...",
        height=100,
    )
    colormap = st.selectbox("Colormap", ["coolwarm", "bwr", "viridis", "plasma", "RdYlGn", "RdBu"])

    if values_str.strip():
        try:
            values = [float(v.strip()) for v in values_str.split(",") if v.strip()]
            v = view_pdb(pdb_data)
            color_by_values(v, values, colormap=colormap)
            render_view(v)
        except ValueError:
            st.error("Enter values as comma-separated floats.")
