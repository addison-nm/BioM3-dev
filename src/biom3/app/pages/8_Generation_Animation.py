import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from biom3.Stage3.animation_tools import (
    generate_sequence_animation,
    gif_to_mp4,
    confidence_metric,
)
from biom3.viz._tokens import TOKENS
from biom3.app._helpers import pick_file, load_pt

st.header("Sequence Generation Animation")
st.write(
    "Render a GIF (or MP4) showing Stage 3 diffusion unmasking as a styled "
    "grid animation. Load a `.pt` containing per-step token frames "
    "(optionally with probabilities)."
)


def _looks_like_frames(obj) -> bool:
    if isinstance(obj, list) and obj and hasattr(obj[0], "__len__"):
        return True
    if isinstance(obj, np.ndarray) and obj.ndim == 2:
        return True
    return False


def _run_page() -> None:
    frames_src = pick_file(
        label="Frames data (.pt)",
        extensions=[".pt"],
        upload_types=["pt"],
        key="anim_frames",
    )
    if frames_src is None:
        return

    try:
        loaded = load_pt(frames_src)
    except Exception as e:
        st.error(f"Failed to load .pt: {e}")
        return

    payload = loaded
    probs = None
    tokens: list[str] = list(TOKENS)

    if isinstance(loaded, dict):
        if "frames" in loaded:
            payload = loaded
        else:
            keys = list(loaded.keys())
            selected_key = st.selectbox("Select (prompt, replica) key", keys)
            payload = loaded[selected_key]

    if isinstance(payload, dict):
        frames = payload.get("frames")
        probs = payload.get("probs")
        if "tokens" in payload:
            tokens = list(payload["tokens"])
    elif _looks_like_frames(payload):
        frames = payload
    else:
        st.error(
            f"Could not locate frames in loaded object (type={type(payload).__name__}). "
            "Expected a list of per-step token-index arrays or a dict with a 'frames' key."
        )
        return

    if frames is None:
        st.error("No frames available in the selected data.")
        return

    frames = [np.asarray(f, dtype=int) for f in frames]
    if probs is not None:
        probs = np.asarray(probs)

    st.caption(
        f"Loaded {len(frames)} frames, seq_len={len(frames[0])}, "
        f"{'with' if probs is not None else 'without'} probabilities."
    )

    probs_src = pick_file(
        label="Optional: separate probabilities file",
        extensions=[".pt", ".npz"],
        upload_types=["pt", "npz"],
        key="anim_probs",
    )
    if probs_src is not None:
        suffix = getattr(probs_src, "name", str(probs_src)).lower()
        try:
            if suffix.endswith(".npz"):
                if isinstance(probs_src, Path):
                    data = np.load(probs_src, allow_pickle=True)
                else:
                    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
                        tmp.write(probs_src.read())
                        data = np.load(tmp.name, allow_pickle=True)
                probs = np.asarray(data["probs"])
                if "tokens" in data.files:
                    tokens = list(data["tokens"].tolist())
            else:
                ploaded = load_pt(probs_src)
                if isinstance(ploaded, dict) and "probs" in ploaded:
                    probs = np.asarray(ploaded["probs"])
                    if "tokens" in ploaded:
                        tokens = list(ploaded["tokens"])
                else:
                    probs = np.asarray(ploaded)
            st.success(f"Loaded probabilities: shape={tuple(probs.shape)}")
        except Exception as e:
            st.error(f"Failed to load probabilities: {e}")

    st.subheader("Animation options")

    c1, c2, c3 = st.columns(3)
    with c1:
        if probs is None:
            prob_style = st.selectbox(
                "Probability style", ["none"], help="Load probabilities to enable.",
            )
        else:
            prob_style = st.selectbox(
                "Probability style",
                ["none", "brightness", "colorbar", "logo"],
                index=1,
            )
    with c2:
        cols_per_row = st.number_input(
            "Columns per row", min_value=10, max_value=200, value=50, step=5,
        )
    with c3:
        duration = st.number_input(
            "Seconds per frame", min_value=0.02, max_value=2.0, value=0.15, step=0.05,
        )

    title = st.text_input("Title (optional)")
    include_confidence = st.checkbox(
        "Add confidence metric row above cells",
        value=False,
        disabled=probs is None,
    )

    if st.button("Generate GIF", type="primary"):
        metrics = None
        if include_confidence and probs is not None:
            metrics = [confidence_metric(probs)]

        kwargs = dict(
            frames=frames,
            tokens=tokens,
            cols_per_row=int(cols_per_row),
            duration=float(duration),
        )
        if probs is not None and prob_style != "none":
            kwargs["probs"] = probs
            kwargs["prob_style"] = prob_style
        if title.strip():
            kwargs["title"] = title.strip()
        if metrics:
            kwargs["metrics"] = metrics

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            gif_path = tmp.name

        with st.spinner("Rendering frames..."):
            try:
                generate_sequence_animation(output_path=gif_path, **kwargs)
                st.session_state["anim_gif_path"] = gif_path
                st.session_state.pop("anim_mp4_path", None)
            except Exception as e:
                st.error(f"Animation failed: {e}")
                return

    gif_path = st.session_state.get("anim_gif_path")
    if gif_path and Path(gif_path).exists():
        st.image(gif_path)
        with open(gif_path, "rb") as f:
            st.download_button(
                "Download GIF",
                data=f.read(),
                file_name="generation.gif",
                mime="image/gif",
            )

        if st.button("Convert to MP4"):
            with st.spinner("Converting to MP4..."):
                try:
                    mp4_path = gif_to_mp4(gif_path)
                    st.session_state["anim_mp4_path"] = mp4_path
                except RuntimeError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"MP4 conversion failed: {e}")

    mp4_path = st.session_state.get("anim_mp4_path")
    if mp4_path and Path(mp4_path).exists():
        st.video(mp4_path)
        with open(mp4_path, "rb") as f:
            st.download_button(
                "Download MP4",
                data=f.read(),
                file_name="generation.mp4",
                mime="video/mp4",
            )


_run_page()
