import streamlit as st

from biom3.viz.viewer import view_pdb, view_overlay
from biom3.viz.alignment import superimpose
from biom3.app._helpers import render_view, pick_pdb

st.header("Align Structures")
st.write("Select two PDB files to superimpose on C-alpha atoms.")

col_up1, col_up2 = st.columns(2)
with col_up1:
    pdb_fixed = pick_pdb("Reference (fixed) PDB", key="fixed")
with col_up2:
    pdb_moving = pick_pdb("Mobile (moving) PDB", key="moving")

if pdb_fixed and pdb_moving:
    st.subheader("Pre-alignment view")
    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        st.caption("Reference")
        render_view(view_pdb(pdb_fixed), height=400)
    with preview_col2:
        st.caption("Mobile")
        render_view(view_pdb(pdb_moving), height=400)

    if st.button("Superimpose"):
        with st.spinner("Aligning structures..."):
            result = superimpose(pdb_fixed, pdb_moving)

        st.subheader("Alignment Statistics")
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("RMSD", f"{result.rmsd:.3f} A")
        col_s2.metric("Paired CA atoms", result.n_atoms)

        v = view_overlay(
            [result.fixed_pdb, result.moving_pdb],
            labels=["Reference", "Mobile"],
            colors=["blue", "red"],
        )
        render_view(v)
