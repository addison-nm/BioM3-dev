from __future__ import annotations

from pathlib import Path

import numpy as np

from biom3.viz._tokens import TOKENS, MASK_IDX, SPECIAL_TOKENS
from biom3.viz import viewer


def extract_unmasking_order(mask_realization_list: list[np.ndarray]) -> np.ndarray:
    """Extract per-residue unmasking step from a list of diffusion frames.

    Compares consecutive frames to find when each position transitions
    from masked (token index 0) to unmasked. Works for both random and
    confidence-based unmasking.

    Parameters
    ----------
    mask_realization_list : list[np.ndarray]
        Output from batch_generate_denoised_sampled(). Each element has
        shape [batch, 1, seq_len] or [1, seq_len] or [seq_len].
        Processes the first sequence (batch index 0).

    Returns
    -------
    np.ndarray
        Shape [seq_len]. Value at position i is the step at which residue i
        was first unmasked. Positions that were never masked get step 0.
        Positions still masked at the end get -1.
    """
    first_frame = np.asarray(mask_realization_list[0]).flatten()
    seq_len = first_frame.shape[0]
    order = np.full(seq_len, -1, dtype=int)

    # Positions already unmasked at step 0
    order[first_frame != MASK_IDX] = 0

    prev = first_frame.copy()
    for step in range(1, len(mask_realization_list)):
        curr = np.asarray(mask_realization_list[step]).flatten()[:seq_len]
        newly_unmasked = (prev == MASK_IDX) & (curr != MASK_IDX)
        order[newly_unmasked] = step
        prev = curr

    return order


def extract_unmasking_order_from_sampling_path(
    sampling_path,
) -> np.ndarray:
    """Extract unmasking order directly from the sampling path tensor.

    For random-order unmasking, sampling_path[i] gives the diffusion step
    at which position i is unmasked.

    Parameters
    ----------
    sampling_path : array-like
        Shape [seq_len] — the random permutation used for one sequence.
    """
    return np.asarray(sampling_path).flatten().astype(int)


def unmasking_order_to_normalized(order: np.ndarray) -> np.ndarray:
    """Normalize unmasking step values to [0, 1] for colormap use.

    0.0 = earliest unmasked (e.g. blue), 1.0 = latest unmasked (e.g. red).
    Special token positions (START, END, PAD, MASK still present) are set to NaN.
    """
    result = order.astype(float).copy()

    # Identify special token positions in the first/last frames
    # Special tokens: START is usually at position 0, END/PAD at the tail
    # We detect them by checking the token identity rather than position
    # Positions that were never unmasked (still -1) are also NaN
    result[result < 0] = np.nan

    valid = ~np.isnan(result)
    if valid.any():
        vmin = np.nanmin(result)
        vmax = np.nanmax(result)
        if vmax > vmin:
            result[valid] = (result[valid] - vmin) / (vmax - vmin)
        else:
            result[valid] = 0.5

    return result


def view_unmasking_order(
    pdb: str | Path,
    mask_realization_list: list[np.ndarray] | None = None,
    sampling_path=None,
    colormap: str = "coolwarm",
    width: int = 800,
    height: int = 600,
) -> "py3Dmol.view":
    """Visualize a structure colored by unmasking order.

    Accepts either mask_realization_list or sampling_path. Exactly one
    must be provided.

    Colors residues on a gradient: early-unmasked = blue, late = red.

    Parameters
    ----------
    pdb : str or Path
        PDB file path or PDB string.
    mask_realization_list : list[np.ndarray], optional
        Diffusion frames from batch_generate_denoised_sampled().
    sampling_path : array-like, optional
        Random permutation tensor from Stage 3 generation.
    colormap : str
        Matplotlib colormap name.
    """
    if (mask_realization_list is None) == (sampling_path is None):
        raise ValueError("Provide exactly one of mask_realization_list or sampling_path")

    if mask_realization_list is not None:
        order = extract_unmasking_order(mask_realization_list)
    else:
        order = extract_unmasking_order_from_sampling_path(sampling_path)

    # Strip special token positions to match PDB residue numbering.
    # The generated sequence has <START> at index 0 and <END>/<PAD> at the tail.
    # PDB structures from ESMFold only contain amino acid residues.
    # We need to find the amino acid subsequence.
    last_frame = None
    if mask_realization_list is not None:
        last_frame = np.asarray(mask_realization_list[-1]).flatten()
    else:
        # Without frames, assume standard layout: START at 0, AAs in middle, END/PAD at tail
        # Use a heuristic: skip index 0 (START) and trim from the end
        last_frame = None

    if last_frame is not None:
        # Keep only positions that are standard amino acids in the final sequence
        aa_mask = np.array([
            TOKENS[int(idx)] not in SPECIAL_TOKENS
            for idx in last_frame
        ])
        aa_order = order[aa_mask]
    else:
        # Without frame data, skip first position (START) and use all remaining
        # until we hit what would be END/PAD. Best effort.
        aa_order = order[1:]  # skip START

    normalized = unmasking_order_to_normalized(aa_order)

    view = viewer.view_pdb(pdb, width=width, height=height)
    viewer.color_by_values(view, normalized.tolist(), colormap=colormap)
    return view
