import os
import tempfile

import streamlit as st
import torch

from biom3.viz.unmasking import view_unmasking_order
from biom3.app._helpers import render_view, upload_pdb

st.header("Unmasking Order Visualization")
st.write(
    "Upload a PDB structure and Stage 3 diffusion data to color residues "
    "by their generation order (blue = early, red = late)."
)

pdb_data = upload_pdb()

data_source = st.radio(
    "Unmasking data source",
    ["Upload .pt animation frames", "Upload .pt sampling path"],
    horizontal=True,
)

frames_file = None
path_file = None
if data_source == "Upload .pt animation frames":
    frames_file = st.file_uploader("Animation frames (.pt)", type=["pt"], key="frames")
else:
    path_file = st.file_uploader("Sampling path (.pt)", type=["pt"], key="spath")

colormap = st.selectbox("Colormap", ["coolwarm", "bwr", "viridis", "plasma", "RdYlGn"])

if pdb_data and (frames_file or path_file):
    try:
        if frames_file:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                tmp.write(frames_file.read())
                tmp_path = tmp.name
            frames_data = torch.load(tmp_path, map_location="cpu", weights_only=False)
            os.unlink(tmp_path)
            if isinstance(frames_data, dict):
                key = st.selectbox("Select (prompt, replica) key", list(frames_data.keys()))
                frames = frames_data[key]
            else:
                frames = frames_data
            v = view_unmasking_order(pdb_data, mask_realization_list=frames, colormap=colormap)
        else:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                tmp.write(path_file.read())
                tmp_path = tmp.name
            sp = torch.load(tmp_path, map_location="cpu", weights_only=False)
            os.unlink(tmp_path)
            v = view_unmasking_order(pdb_data, sampling_path=sp, colormap=colormap)

        render_view(v)
    except Exception as e:
        st.error(f"Error: {e}")
