import io

import streamlit as st

from biom3.viz.viewer import view_pdb, color_by_values
from biom3.app._helpers import render_view, pick_pdb


def _extract_ca_bfactors(pdb_str: str) -> list[float]:
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", io.StringIO(pdb_str))
    bfactors: list[float] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    bfactors.append(float(residue["CA"].get_bfactor()))
        break
    return bfactors

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
        bfactor_mode = st.checkbox("Color by B-factor / pLDDT")
        if bfactor_mode:
            bfactor_cmap = st.selectbox(
                "B-factor colormap",
                ["RdYlGn", "viridis", "plasma", "coolwarm", "bwr"],
            )
    with col1:
        v = view_pdb(pdb_data, style=style, color_scheme=color_scheme)
        if bfactor_mode:
            try:
                bfactors = _extract_ca_bfactors(pdb_data)
                if bfactors:
                    color_by_values(v, bfactors, colormap=bfactor_cmap)
                else:
                    st.warning("No CA atoms found in PDB; B-factor coloring skipped.")
            except Exception as e:
                st.warning(f"Could not parse B-factors: {e}")
        render_view(v)
