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
    output_path: str | None = None,
    figsize: tuple[float, float] = (14, 8),
    hide_pad: bool = False,
    blank_unmasked: bool = False,
) -> matplotlib.figure.Figure:
    """Plot per-position probability dynamics over diffusion steps.

    Parameters
    ----------
    probs : np.ndarray
        Shape ``[steps, seq_len, num_classes]``.
    tokens : list[str]
        Token vocabulary (index -> string).
    frames : list[np.ndarray], optional
        Token index arrays per step.  Used to annotate y-axis with final
        token identities and required for *hide_pad* / *blank_unmasked*.
    output_path : str, optional
        Save figure to this path when provided.
    figsize : tuple
        Figure size in inches.
    hide_pad : bool
        Drop positions whose final token is ``<PAD>`` from the plot.
        Requires *frames*.
    blank_unmasked : bool
        Show background colour (white) for steps after a position has been
        unmasked, so only the masked-phase probabilities are visible.
        Requires *frames*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    steps, seq_len, _ = probs.shape
    max_probs = probs.max(axis=-1)  # [steps, seq_len]

    has_frames = frames is not None and len(frames) > 0
    last_frame = frames[-1] if has_frames else None

    # -- Build per-position labels and identify PAD positions --
    labels: list[str] = []
    pos_indices = np.arange(seq_len)
    if has_frames:
        for j in range(seq_len):
            labels.append(_token_label(tokens[int(last_frame[j])]))

    # -- hide_pad: remove PAD columns --
    if hide_pad and has_frames:
        keep = np.array([tokens[int(last_frame[j])] != _PAD_TOKEN
                         for j in range(seq_len)])
        max_probs = max_probs[:, keep]
        pos_indices = pos_indices[keep]
        labels = [labels[j] for j in range(seq_len) if keep[j]]

    plot_len = max_probs.shape[1]

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
        mask = np.zeros_like(max_probs, dtype=bool)
        for j in range(plot_len):
            mask[unmasked_at[j]:, j] = True
        max_probs = np.ma.masked_array(max_probs, mask=mask)

    fig, axes = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]},
    )

    # -- Top panel: confidence heatmap (positions x steps) --
    ax0 = axes[0]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    im = ax0.imshow(
        max_probs.T,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax0.set_xlabel("Diffusion step")
    ax0.set_ylabel("Sequence position")
    ax0.set_title("Top-1 predicted probability per position over steps")
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
    if isinstance(max_probs, np.ma.MaskedArray):
        valid_counts = (~max_probs.mask).sum(axis=1)
        safe = valid_counts > 0
        mean_conf = np.where(safe, max_probs.filled(0).sum(axis=1) / np.maximum(valid_counts, 1), np.nan)
        filled = max_probs.filled(np.nan)
        min_conf = np.nanmin(filled, axis=1)
        max_conf_line = np.nanmax(filled, axis=1)
    else:
        mean_conf = max_probs.mean(axis=1)
        min_conf = max_probs.min(axis=1)
        max_conf_line = max_probs.max(axis=1)

    ax1.plot(range(steps), mean_conf, color="steelblue", linewidth=2, label="Mean")
    ax1.fill_between(
        range(steps), min_conf, max_conf_line,
        alpha=0.15, color="steelblue", label="Min\u2013Max range",
    )

    ax1.set_xlabel("Diffusion step")
    ax1.set_ylabel("Top-1 probability")
    ax1.set_title("Aggregate confidence over diffusion steps")
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
    return plot_probability_dynamics(
        probs, tokens, output_path=output_path, figsize=figsize,
        hide_pad=hide_pad, blank_unmasked=blank_unmasked,
    )
