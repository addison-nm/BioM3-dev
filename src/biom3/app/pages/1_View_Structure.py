import streamlit as st

from biom3.viz.viewer import view_pdb
from biom3.app._helpers import render_view, pick_pdb

st.header("View Structure")

pdb_source = st.radio(
    "PDB source",
    ["Browse / Upload", "Paste PDB text", "Fold sequence"],
    horizontal=True,
)

pdb_data = None
if pdb_source == "Browse / Upload":
    pdb_data = pick_pdb(key="view_pdb")
elif pdb_source == "Paste PDB text":
    pdb_data = st.text_area("Paste PDB content", height=200, key="paste_pdb")
    if not pdb_data.strip():
        pdb_data = None
elif pdb_source == "Fold sequence":
    seq = st.text_input("Amino acid sequence (single-letter codes)")
    if seq and st.button("Fold with ESMFold"):
        with st.spinner("Folding sequence with ESMFold..."):
            try:
                from biom3.viz.folding import fold_sequence
                pdb_data = fold_sequence(seq)
                st.success("Folding complete")
            except ImportError:
                st.error("ESMFold not available. Install with: `pip install 'fair-esm[esmfold]' omegaconf`")
            except Exception as e:
                st.error(f"Folding failed: {e}")

if pdb_data:
    col1, col2 = st.columns([3, 1])
    with col2:
        style = st.selectbox("Style", ["cartoon", "stick", "sphere", "line"])
        color_scheme = st.selectbox("Color scheme", ["spectrum", "chain", "ssJmol", "residue"])
    with col1:
        v = view_pdb(pdb_data, style=style, color_scheme=color_scheme)
        render_view(v)
