"""Composed multidomain generation.

K canvases are decoded in parallel by one composed forward per step, so every
domain sees its partners' partially-revealed state through the coupling as it
fills in. Each canvas is fixed-length: ``<START> + n_d generated residues +
<END>``, padded to ``seq_len``. Structural tokens are placed up front and never
sampled, so only residue positions are generated.

**The diffusion clock.** Each canvas starts at ``offset_d = seq_len - n_d`` and
its sampling path holds a permutation of ``[offset_d, offset_d + n_d)``, so the
time index at step ``s`` is ``offset_d + s = seq_len - (n_d - s)`` — the count of
revealed positions. This is the same convention
:func:`~biom3.Stage3.inpaint.build_sampling_path_row` implements for in-painting,
reused rather than re-derived: feeding the raw step counter instead would run the
model far outside the timestep range it was trained on.

Canvases have different lengths, so the shorter ones finish first. A finished
canvas has no path entry matching its clock, and its clock is held at ``seq_len``
rather than running past — it keeps contributing context through the coupling
while its partners finish.
"""

import numpy as np
import torch
import torch.nn.functional as F

from biom3.backend.device import setup_logger
from biom3.Stage3.inpaint import (
    END_ID,
    MASK_ID,
    PAD_ID,
    RUNTIME_TOKENS,
    START_ID,
    build_sampling_path_row,
    build_template_state,
)
from biom3.Stage3.sampling_analysis import _fill_gumbel_buffer, _inference_autocast

logger = setup_logger(__name__)

_PATH_SENTINEL = -1
# Residue ids in the runtime vocabulary, i.e. everything that is not a
# structural token. Restricting sampling to these keeps <START>/<END>/<PAD> out
# of the interior of a generated domain.
_RESIDUE_IDS = [
    i for i, token in enumerate(RUNTIME_TOKENS)
    if not token.startswith("<")
]


def build_domain_canvases(lengths, seq_len, generator=None):
    """Initial states and sampling paths for one assembly's K canvases.

    Args:
        lengths: K residue counts, one per domain
        seq_len: canvas width (``image_size ** 2``)
        generator: optional ``torch.Generator`` for reproducible unmask order

    Returns:
        ``(states [K, seq_len], paths [K, seq_len], offsets [K])``
    """
    states, paths, offsets = [], [], []
    for length in lengths:
        length = int(length)
        if length < 1:
            raise ValueError(f"domain length must be >= 1, got {length}")
        if length + 2 > seq_len:
            raise ValueError(
                f"domain of {length} residues needs {length + 2} canvas slots "
                f"(<START> and <END> included) but the canvas holds {seq_len}"
            )
        state, mask_positions = build_template_state(f"(*:{length})", seq_len)
        states.append(state)
        paths.append(build_sampling_path_row(mask_positions, seq_len,
                                             generator=generator))
        offsets.append(seq_len - int(mask_positions.numel()))
    return (torch.stack(states), torch.stack(paths),
            torch.tensor(offsets, dtype=torch.long))


def _restrict_to_residues(logits):
    """Mask out structural tokens so only amino acids can be sampled."""
    allowed = torch.full((logits.size(-1),), float("-inf"), device=logits.device)
    allowed[torch.tensor(_RESIDUE_IDS, device=logits.device)] = 0.0
    return logits + allowed


@torch.no_grad()
def generate_multidomain(
        model,
        y_c,
        lengths,
        *,
        seq_len,
        device,
        token_strategy="sample",
        restrict_to_residues=True,
        couple=True,
        sample_seeds=None,
        generator=None,
        return_trajectory=False,
    ):
    """Decode B assemblies of K domains each.

    Args:
        model: a :class:`~biom3.Stage3.multidomain.model.MultiDomainProteoScribe`
        y_c: conditioning, ``[B, K, emb_dim]``
        lengths: residue counts, ``[B, K]`` (list of lists is fine)
        seq_len: canvas width
        device: device to run on
        token_strategy: ``"sample"`` (Gumbel-max) or ``"argmax"``
        restrict_to_residues: forbid structural tokens at generated positions
        couple: set False for the zero-coupling ablation — each canvas is then
            decoded by its expert alone, which is the baseline any claim about
            the coupling has to beat
        sample_seeds: one int per assembly, making each row's noise reproducible
            independently of batch packing
        generator: ``torch.Generator`` for the unmask orderings
        return_trajectory: also return the per-step canvas states

    Returns:
        ``(states [B, K, seq_len], trajectory)`` where trajectory is ``None``
        unless requested.
    """
    lengths = torch.as_tensor(lengths, dtype=torch.long)
    if lengths.dim() != 2:
        raise ValueError(f"lengths must be [B, K], got {tuple(lengths.shape)}")
    batch_size, num_domains = lengths.shape
    if num_domains != model.num_domains:
        raise ValueError(
            f"lengths has {num_domains} domains but the model was built for "
            f"{model.num_domains}"
        )
    if y_c.shape[:2] != (batch_size, num_domains):
        raise ValueError(
            f"y_c must be [B, K, D] matching lengths; got {tuple(y_c.shape)}")

    states, paths, offsets = [], [], []
    for b in range(batch_size):
        state, path, offset = build_domain_canvases(
            lengths[b].tolist(), seq_len, generator=generator)
        states.append(state)
        paths.append(path)
        offsets.append(offset)
    state = torch.stack(states).to(device)          # [B, K, L]
    path = torch.stack(paths).to(device)            # [B, K, L]
    clock = torch.stack(offsets).to(device)         # [B, K]

    y_c = y_c.to(device)
    # Non-pad positions, constant across the trajectory: the coupling reads a
    # partner's real region only, and <START>/<END>/residues are all real.
    real_masks = state != PAD_ID
    steps = int(lengths.max().item())
    row_idx = torch.arange(batch_size, device=device)

    gumbel = None
    if token_strategy == "sample":
        gumbel = torch.empty(batch_size * num_domains, seq_len, len(RUNTIME_TOKENS),
                             dtype=torch.float32, device=device)

    trajectory = [] if return_trajectory else None

    with _inference_autocast(device):
        for step in range(steps):
            logits = model(state, clock, y_c, real_masks=real_masks, couple=couple)
            logits = logits.float().permute(0, 1, 3, 2)      # [B, K, L, C]
            if restrict_to_residues:
                logits = _restrict_to_residues(logits)

            if token_strategy == "argmax":
                proposed = logits.argmax(dim=-1)
            else:
                _fill_gumbel_buffer(
                    gumbel,
                    None if sample_seeds is None
                    else [s for s in sample_seeds for _ in range(num_domains)],
                    step=step,
                )
                noise = gumbel.reshape(batch_size, num_domains, seq_len, -1)
                proposed = (logits - noise.log()).argmax(dim=-1)   # [B, K, L]

            for d in range(num_domains):
                matches = path[:, d] == clock[:, d].unsqueeze(-1)   # [B, L]
                active = matches.any(dim=-1)
                if not bool(active.any()):
                    continue
                location = matches.long().argmax(dim=-1)            # [B]
                rows = row_idx[active]
                cols = location[active]
                state[rows, d, cols] = proposed[rows, d, cols]

            # Finished canvases hold at seq_len instead of running past the
            # trained timestep range; they still condition their partners.
            clock = torch.minimum(clock + 1, torch.full_like(clock, seq_len))

            if return_trajectory:
                trajectory.append(state.clone().cpu())

    return state, trajectory


def decode_domain(tokens):
    """Token ids for one canvas -> residue string, stopping at ``<END>``."""
    residues = []
    for token_id in tokens.tolist():
        if token_id in (END_ID, PAD_ID):
            break
        if token_id in (START_ID, MASK_ID):
            continue
        residues.append(RUNTIME_TOKENS[token_id])
    return "".join(residues)


def decode_assemblies(states):
    """``[B, K, seq_len]`` token ids -> ``B`` lists of K residue strings."""
    return [[decode_domain(states[b, d]) for d in range(states.size(1))]
            for b in range(states.size(0))]


def assemble_domains(domain_sequences, linker=""):
    """Join one assembly's per-domain sequences into a single protein.

    Plain concatenation is correct only while the canvases are disjoint, which
    holds for domains cut at their Pfam envelopes. It stops holding the moment
    domain sequences are extended into their shared inter-domain linker: both
    neighbours then carry that region and concatenating repeats it. Callers that
    extend must merge on the overlap instead of calling this.
    """
    return linker.join(domain_sequences)
