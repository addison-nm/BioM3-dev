import streamlit as st

from biom3.viz.unmasking import view_unmasking_order
from biom3.app._helpers import render_view, pick_pdb, pick_pt, load_pt

st.header("Unmasking Order Visualization")
st.write(
    "Select a PDB structure and Stage 3 diffusion data to color residues "
    "by their generation order (blue = early, red = late)."
)

pdb_data = pick_pdb(key="unmask_pdb")

data_source = st.radio(
    "Unmasking data source",
    ["Animation frames (.pt)", "Sampling path (.pt)"],
    horizontal=True,
)

pt_data = None
if data_source == "Animation frames (.pt)":
    pt_file = pick_pt("Animation frames", key="frames")
    if pt_file is not None:
        pt_data = ("frames", pt_file)
else:
    pt_file = pick_pt("Sampling path", key="spath")
    if pt_file is not None:
        pt_data = ("path", pt_file)

colormap = st.selectbox("Colormap", ["coolwarm", "bwr", "viridis", "plasma", "RdYlGn"])

if pdb_data and pt_data:
    kind, file_or_path = pt_data
    try:
        loaded = load_pt(file_or_path)
        if kind == "frames":
            if isinstance(loaded, dict):
                key = st.selectbox("Select (prompt, replica) key", list(loaded.keys()))
                frames = loaded[key]
            else:
                frames = loaded
            v = view_unmasking_order(pdb_data, mask_realization_list=frames, colormap=colormap)
        else:
            v = view_unmasking_order(pdb_data, sampling_path=loaded, colormap=colormap)

        render_view(v)
    except Exception as e:
        st.error(f"Error: {e}")
