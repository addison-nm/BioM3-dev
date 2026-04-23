import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from biom3.viz.dynamics import (
    plot_probability_dynamics,
    plot_probability_dynamics_from_file,
)
from biom3.viz._tokens import TOKENS
from biom3.app._helpers import pick_file, load_pt

st.header("Probability Dynamics")
st.write(
    "Visualize per-position confidence across Stage 3 diffusion steps. "
    "Load an `.npz` produced by `run_ProteoScribe_sample` or a `.pt` file "
    "containing `probs` (and optionally `frames` / `final_frame` / `tokens`)."
)

source = pick_file(
    label="Probability data",
    extensions=[".npz", ".pt"],
    upload_types=["npz", "pt"],
    key="probdyn",
)

col1, col2 = st.columns(2)
with col1:
    hide_pad = st.checkbox("Hide PAD positions", value=False)
with col2:
    blank_unmasked = st.checkbox("Blank cells after unmasking", value=False)

def _resolve_npz_path(src) -> str:
    if isinstance(src, Path):
        return str(src)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp.write(src.read())
        return tmp.name


if source is not None:
    suffix = getattr(source, "name", str(source)).lower()

    if suffix.endswith(".npz"):
        npz_path = _resolve_npz_path(source)
        try:
            fig = plot_probability_dynamics_from_file(
                npz_path,
                hide_pad=hide_pad,
                blank_unmasked=blank_unmasked,
            )
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error plotting from .npz: {e}")
        finally:
            if not isinstance(source, Path):
                try:
                    os.unlink(npz_path)
                except OSError:
                    pass
    else:
        loaded = None
        try:
            loaded = load_pt(source)
        except Exception as e:
            st.error(f"Failed to load .pt: {e}")

        if loaded is not None:
            payload = loaded
            if isinstance(loaded, dict) and loaded and "probs" not in loaded:
                keys = list(loaded.keys())
                selected_key = st.selectbox("Select (prompt, replica) key", keys)
                payload = loaded[selected_key]

            if not isinstance(payload, dict) or "probs" not in payload:
                st.error(
                    "Expected a dict containing `probs` (shape [steps, seq_len, num_classes]). "
                    "Got: " + (
                        f"keys {list(payload.keys())}" if isinstance(payload, dict)
                        else f"type {type(payload).__name__}"
                    )
                )
            else:
                probs = np.asarray(payload["probs"])
                tokens = list(payload.get("tokens", TOKENS))
                frames = payload.get("frames")
                final_frame = payload.get("final_frame")
                if frames is not None:
                    frames = [np.asarray(f) for f in frames]
                if final_frame is not None:
                    final_frame = np.asarray(final_frame)

                try:
                    fig = plot_probability_dynamics(
                        probs,
                        tokens,
                        frames=frames,
                        final_frame=final_frame,
                        hide_pad=hide_pad,
                        blank_unmasked=blank_unmasked,
                    )
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error plotting: {e}")
