from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from biom3.Stage3.animation_tools import _SPECIAL_TOKENS


_MASK_IDX = 0
_PAD_TOKEN = "<PAD>"


def _token_label(tok: str) -> str:
    if tok in _SPECIAL_TOKENS:
        return tok[1]
    if tok == "-":
        return "\u00b7"
    return tok


def plot_probability_dynamics(
    probs: np.ndarray,
    tokens: list[str],
    frames: list[np.ndarray] | None = None,
    final_frame: np.ndarray | None = None,
    output_path: str | None = None,
    figsize: tuple[float, float] = (14, 8),
    hide_pad: bool = False,
    blank_unmasked: bool = False,
) -> matplotlib.figure.Figure:
    """Plot per-position probability dynamics over diffusion steps.

    The heatmap and aggregate line plot show, at each step and position,
    the probability the model assigned to the *chosen* token (the one that
    ultimately appears at that position in the final sequence) — not the
    per-step argmax.  When neither ``frames`` nor ``final_frame`` is
    provided, falls back to per-step top-1 probability.

    Parameters
    ----------
    probs : np.ndarray
        Shape ``[steps, seq_len, num_classes]``.
    tokens : list[str]
        Token vocabulary (index -> string).
    frames : list[np.ndarray], optional
        Per-step token index arrays.  Required for *blank_unmasked*; the
        last element is used as the final frame for chosen-token indexing
        and *hide_pad* if ``final_frame`` is not given.
    final_frame : np.ndarray, optional
        Just the final-step token indices, shape ``[seq_len]``.  Sufficient
        for chosen-token indexing and *hide_pad*; takes precedence over
        ``frames[-1]`` when both are supplied.
    output_path : str, optional
        Save figure to this path when provided.
    figsize : tuple
        Figure size in inches.
    hide_pad : bool
        Drop positions whose final token is ``<PAD>`` from the plot.
        Requires ``frames`` or ``final_frame``.
    blank_unmasked : bool
        Show background colour (white) for steps after a position has been
        unmasked, so only the masked-phase probabilities are visible.
        Requires per-step ``frames``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    steps, seq_len, _ = probs.shape

    has_frames = frames is not None and len(frames) > 0
    if final_frame is not None:
        last_frame = np.asarray(final_frame, dtype=int)
    elif has_frames:
        last_frame = np.asarray(frames[-1], dtype=int)
    else:
        last_frame = None

    if last_frame is not None:
        chosen_probs = np.take_along_axis(
            probs, last_frame[None, :, None], axis=-1,
        ).squeeze(-1)  # [steps, seq_len]
    else:
        chosen_probs = probs.max(axis=-1)

    # -- Build per-position labels and identify PAD positions --
    labels: list[str] = []
    pos_indices = np.arange(seq_len)
    if last_frame is not None:
        for j in range(seq_len):
            labels.append(_token_label(tokens[int(last_frame[j])]))

    # -- hide_pad: remove PAD columns --
    if hide_pad and last_frame is not None:
        keep = np.array([tokens[int(last_frame[j])] != _PAD_TOKEN
                         for j in range(seq_len)])
        chosen_probs = chosen_probs[:, keep]
        pos_indices = pos_indices[keep]
        labels = [labels[j] for j in range(seq_len) if keep[j]]

    plot_len = chosen_probs.shape[1]

    # -- blank_unmasked: mask cells after each position is unmasked --
    if blank_unmasked and has_frames:
        frame_arr = np.array(frames)  # [steps, seq_len]
        if hide_pad:
            frame_arr = frame_arr[:, keep]
        # For each position, find the first step where it leaves MASK_IDX
        unmasked_at = np.full(plot_len, steps, dtype=int)
        for j in range(plot_len):
            hits = np.where(frame_arr[:, j] != _MASK_IDX)[0]
            if len(hits) > 0:
                unmasked_at[j] = hits[0]
        mask = np.zeros_like(chosen_probs, dtype=bool)
        for j in range(plot_len):
            mask[unmasked_at[j]:, j] = True
        chosen_probs = np.ma.masked_array(chosen_probs, mask=mask)

    fig, axes = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]},
    )

    title_label = "chosen-token" if last_frame is not None else "top-1"

    # -- Top panel: confidence heatmap (positions x steps) --
    ax0 = axes[0]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    im = ax0.imshow(
        chosen_probs.T,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax0.set_xlabel("Diffusion step")
    ax0.set_ylabel("Sequence position")
    ax0.set_title(f"P({title_label}) per position over steps")
    fig.colorbar(im, ax=ax0, label="Probability", shrink=0.8)

    if labels:
        tick_step = max(1, plot_len // 40)
        tick_pos = list(range(0, plot_len, tick_step))
        ax0.set_yticks(tick_pos)
        ax0.set_yticklabels(
            [f"{pos_indices[p]} {labels[p]}" for p in tick_pos], fontsize=7,
        )

    # -- Bottom panel: aggregate confidence over time --
    ax1 = axes[1]
    if isinstance(chosen_probs, np.ma.MaskedArray):
        valid_counts = (~chosen_probs.mask).sum(axis=1)
        safe = valid_counts > 0
        mean_conf = np.where(safe, chosen_probs.filled(0).sum(axis=1) / np.maximum(valid_counts, 1), np.nan)
        filled = chosen_probs.filled(np.nan)
        min_conf = np.nanmin(filled, axis=1)
        max_conf_line = np.nanmax(filled, axis=1)
    else:
        mean_conf = chosen_probs.mean(axis=1)
        min_conf = chosen_probs.min(axis=1)
        max_conf_line = chosen_probs.max(axis=1)

    ax1.plot(range(steps), mean_conf, color="steelblue", linewidth=2, label="Mean")
    ax1.fill_between(
        range(steps), min_conf, max_conf_line,
        alpha=0.15, color="steelblue", label="Min\u2013Max range",
    )

    ax1.set_xlabel("Diffusion step")
    ax1.set_ylabel(f"P({title_label})")
    ax1.set_title(f"Aggregate P({title_label}) over diffusion steps")
    ax1.set_xlim(0, steps - 1)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)

    fig.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_probability_dynamics_from_file(
    npz_path: str,
    output_path: str | None = None,
    figsize: tuple[float, float] = (14, 8),
    hide_pad: bool = False,
    blank_unmasked: bool = False,
) -> matplotlib.figure.Figure:
    """Load saved probability data and plot dynamics.

    Parameters
    ----------
    npz_path : str
        Path to ``.npz`` file saved by ``run_ProteoScribe_sample``
        (keys: ``probs`` shape ``[steps, seq_len, num_classes]``,
        ``tokens`` vocabulary array).
    output_path : str, optional
        Save figure to this path when provided.
    figsize : tuple
        Figure size in inches.
    hide_pad : bool
        Drop positions whose final token is ``<PAD>``.
    blank_unmasked : bool
        Show background colour for steps after a position is unmasked.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = np.load(npz_path, allow_pickle=True)
    probs = data["probs"]
    tokens = data["tokens"].tolist()
    final_frame = data["final_frame"] if "final_frame" in data.files else None
    return plot_probability_dynamics(
        probs, tokens, final_frame=final_frame,
        output_path=output_path, figsize=figsize,
        hide_pad=hide_pad, blank_unmasked=blank_unmasked,
    )
