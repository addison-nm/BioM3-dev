import os
import numpy as np
import random
import pandas as pd
import math
from dataclasses import dataclass
from tqdm import tqdm
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import biom3.Stage3.preprocess as prep
import biom3.Stage3.cond_diff_transformer_layer as mod
import biom3.Stage3.transformer_training_helper as train_helper

from biom3.backend.device import BACKEND_NAME, _CUDA, _XPU, setup_logger

logger = setup_logger(__name__)


def _device_synchronize(device) -> None:
    """Block the host until all queued kernels on ``device`` complete.

    Cheap when called once per diffusion step (the kernels are serialized
    by data dependency anyway), and makes tqdm reflect actual GPU progress
    instead of Python kernel-launch rate. No-op on CPU.
    """
    dev = torch.device(device)
    if dev.type == _XPU and hasattr(torch, "xpu"):
        try:
            torch.xpu.synchronize(dev)
        except Exception:
            pass
    elif dev.type == _CUDA:
        try:
            torch.cuda.synchronize(dev)
        except Exception:
            pass


def _inference_autocast(device):
    """bf16 autocast on CUDA/XPU; disabled (no-op) on CPU.

    On Blackwell (sm_121a) PyTorch's fp32 GEMM falls back to non-tensor-core
    kernels (magma_sgemmEx / cutlass SIMT), so eager sampling is several
    times slower than it should be. Routing matmul through the bf16 path
    reaches tuned tensor-core kernels and delivers ~3x on the real model.
    Autocast's built-in promotion rules keep softmax/log in fp32 internally.
    """
    device_type = torch.device(device).type
    enabled = device_type in ("cuda", "xpu")
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=enabled)


@dataclass
class _SamplingState:
    """Pre-allocated GPU buffers and per-call state shared by both batched samplers.

    The two ``batch_generate_denoised_sampled*`` entry points have nearly
    identical setup (move inputs to device, allocate ``all_realizations``
    / ``all_time_idx`` / optional ``all_probs``, optional Gumbel noise
    buffer for the 'sample' token strategy) and identical drain logic at
    the end. This dataclass + the helpers below factor that out so each
    sampler's body holds only the per-iteration step that actually
    differs (random-path vs confidence-based location selection).
    """
    temp_y_c: torch.Tensor
    temp_mask_realization: torch.Tensor
    temp_idx: torch.Tensor
    batch_size: int
    seq_len: int
    max_diffusion_step: int
    batch_idx: torch.Tensor
    token_strategy: str
    all_realizations: torch.Tensor
    all_time_idx: torch.Tensor
    all_probs: Optional[torch.Tensor]
    gumbel_buffer: Optional[torch.Tensor]


def _init_sampling_state(
    args,
    extract_digit_samples: torch.Tensor,
    extract_time: torch.Tensor,
    extract_digit_label: torch.Tensor,
    store_probabilities: bool,
) -> _SamplingState:
    """Move inputs to device and pre-allocate the per-step output buffers.

    Mirrors the setup that used to live at the top of each batched
    sampler. Returns a ``_SamplingState`` whose fields the caller mutates
    in place across the diffusion loop.
    """
    batch_size = extract_digit_samples.size(0)
    seq_len = extract_digit_samples.size(1)
    logger.debug("batch_size: %s", batch_size)

    temp_y_c = extract_digit_label.to(args.device)
    temp_mask_realization = extract_digit_samples.unsqueeze(1).long().to(args.device)
    temp_idx = extract_time.unsqueeze(-1).to(args.device)

    max_diffusion_step = args.diffusion_steps
    batch_idx = torch.arange(batch_size, device=args.device)
    token_strategy = getattr(args, "token_strategy", "sample")

    # Pre-allocate GPU tensors to avoid per-iteration CPU transfers
    # (the f778a8d perf optimization — keep one drain at the end).
    all_realizations = torch.empty(
        max_diffusion_step, batch_size, 1, seq_len,
        dtype=temp_mask_realization.dtype, device=args.device,
    )
    all_time_idx = torch.empty(
        max_diffusion_step, batch_size, 1,
        dtype=temp_idx.dtype, device=args.device,
    )

    all_probs: Optional[torch.Tensor] = None
    if store_probabilities:
        all_probs = torch.empty(
            max_diffusion_step, batch_size, seq_len, args.num_classes,
            dtype=torch.float32, device=args.device,
        )

    gumbel_buffer: Optional[torch.Tensor] = None
    if token_strategy == "sample":
        # Reused each step via in-place fill. See docs/gumbel_max_sampling.md.
        gumbel_buffer = torch.empty(
            batch_size, seq_len, args.num_classes,
            dtype=temp_y_c.dtype, device=args.device,
        )

    return _SamplingState(
        temp_y_c=temp_y_c,
        temp_mask_realization=temp_mask_realization,
        temp_idx=temp_idx,
        batch_size=batch_size,
        seq_len=seq_len,
        max_diffusion_step=max_diffusion_step,
        batch_idx=batch_idx,
        token_strategy=token_strategy,
        all_realizations=all_realizations,
        all_time_idx=all_time_idx,
        all_probs=all_probs,
        gumbel_buffer=gumbel_buffer,
    )


def _forward_logits(
    model: nn.Module,
    temp_mask_realization: torch.Tensor,
    temp_idx: torch.Tensor,
    temp_y_c: torch.Tensor,
) -> torch.Tensor:
    """One model forward pass for the diffusion step.

    Returns logits in ``[batch, seq_len, num_classes]`` layout (the model
    natively returns ``[batch, num_classes, seq_len]``; we transpose
    once here so downstream ops work positionally). Promoted to fp32 so
    softmax / Gumbel math stays stable under bf16 autocast.
    """
    return (
        model(x=temp_mask_realization.squeeze(1), t=temp_idx.view(-1), y_c=temp_y_c)
        .transpose(1, 2)
        .float()
    )


def _diffusion_tqdm_desc(args) -> str | None:
    """Return a tqdm `desc` for the diffusion loop, or ``None`` to fall
    back to bare tqdm when the run is single-process.

    Format: ``"diffuse rank=R/W dev=DEVICE host=HOST"`` — emitted only
    when ``args._world_size > 1`` so single-rank runs keep the original
    minimal progress bar.
    """
    world_size = int(getattr(args, "_world_size", 1))
    if world_size <= 1:
        return None
    rank = int(getattr(args, "_rank", 0))
    device = getattr(args, "device", "?")
    import socket
    host = socket.gethostname().split(".")[0]
    return f"diffuse rank={rank}/{world_size} dev={device} host={host}"


def _fill_gumbel_buffer(buffer: torch.Tensor, sample_seeds, step: int = 0) -> None:
    """In-place fill ``buffer`` with Exp(1) noise.

    Without ``sample_seeds`` the call uses the global RNG (legacy path).
    With one seed per row, each row's noise is drawn from a per-row
    generator seeded from ``(seed, step)`` — per-sample reproducibility
    independent of batch packing or world size. The generator is built
    on the buffer's device when supported, falling back to CPU random
    + cross-device copy otherwise.
    """
    if sample_seeds is None:
        buffer.exponential_()
        return
    if len(sample_seeds) != buffer.size(0):
        raise ValueError(
            f"sample_seeds length {len(sample_seeds)} != batch size {buffer.size(0)}"
        )
    device = buffer.device
    for i, seed in enumerate(sample_seeds):
        per_step_seed = (int(seed) ^ (int(step) * 0x9E3779B1)) & 0xFFFFFFFF
        try:
            gen = torch.Generator(device=device).manual_seed(per_step_seed)
            buffer[i].exponential_(generator=gen)
        except (RuntimeError, TypeError):
            cpu_gen = torch.Generator().manual_seed(per_step_seed)
            cpu_noise = torch.empty(buffer.shape[1:], dtype=buffer.dtype)
            cpu_noise.exponential_(generator=cpu_gen)
            buffer[i].copy_(cpu_noise)


def _drain_sampling_state(state: _SamplingState):
    """Convert pre-allocated GPU buffers to the public return shape.

    Single GPU→CPU transfer at the end, per the f778a8d perf
    optimization (vs. four .cpu() syncs per diffusion step).
    """
    L = state.max_diffusion_step
    all_realizations_np = state.all_realizations.cpu().numpy()
    all_time_idx_np = state.all_time_idx.cpu().numpy()
    mask_realization_list = [all_realizations_np[i] for i in range(L)]
    time_idx_list = [all_time_idx_np[i] for i in range(L)]
    probs_np = state.all_probs.cpu().numpy() if state.all_probs is not None else None
    return mask_realization_list, time_idx_list, probs_np


def batch_generate_denoised_sampled(
        args: any,
        model: nn.Module,
        extract_digit_samples: torch.Tensor,
        extract_time: torch.Tensor,
        extract_digit_label: torch.Tensor,
        sampling_path: torch.Tensor,
        store_probabilities: bool = False,
        sample_seeds: Optional[list] = None,
    ) -> tuple[list, list, np.ndarray | None]:
    """Random-path unmasking: at each step, unmask the position selected by
    the per-batch sampling permutation. Token selection is governed by
    ``args.token_strategy`` (``'argmax'`` deterministic, ``'sample'``
    stochastic via Gumbel-max).

    When ``sample_seeds`` is provided (one int per row of the batch), the
    Gumbel noise for each row is drawn from a per-row generator seeded
    from ``(seed, step)`` — per-(prompt, replica) sampling is
    reproducible independently of world size or batch packing.
    """

    # Ensure batch dimension consistency across input tensors
    assert (
        extract_digit_samples.size(0)
        == extract_digit_label.size(0)
        == sampling_path.size(0)
        == extract_time.size(0)
    ), "Mismatched batch dimensions"

    state = _init_sampling_state(
        args, extract_digit_samples, extract_time, extract_digit_label,
        store_probabilities,
    )
    temp_sampling_path = sampling_path.to(args.device)
    # Per-step device_synchronize is only useful for tqdm progress accuracy.
    # It serialises kernel launches and can stall ranks under heavy
    # collectives, so we skip it whenever the run is distributed.
    sync_per_step = int(getattr(args, "_world_size", 1)) <= 1

    with _inference_autocast(args.device):
        for ii in tqdm(range(state.max_diffusion_step), desc=_diffusion_tqdm_desc(args)):
            logits = _forward_logits(
                model, state.temp_mask_realization, state.temp_idx, state.temp_y_c,
            )

            if state.all_probs is not None:
                state.all_probs[ii] = F.softmax(logits, dim=-1)

            # Token selection — softmax is monotonic so argmax(log_softmax + G)
            # == argmax(logits + G). See docs/gumbel_max_sampling.md.
            if state.token_strategy == "argmax":
                next_temp_realization = logits.argmax(dim=-1)
            else:
                _fill_gumbel_buffer(state.gumbel_buffer, sample_seeds, step=ii)
                next_temp_realization = (logits - state.gumbel_buffer.log()).argmax(dim=-1)

            # Update temp_mask_realization per sample (advanced indexing)
            current_location = (temp_sampling_path == state.temp_idx).long().argmax(dim=-1)
            state.temp_mask_realization[state.batch_idx, 0, current_location] = (
                next_temp_realization[state.batch_idx, current_location]
            )

            # Store on GPU
            state.all_realizations[ii] = state.temp_mask_realization
            state.all_time_idx[ii] = state.temp_idx
            state.temp_idx += 1

            # Sync each iteration so tqdm reflects actual GPU progress, not
            # async kernel-launch rate. Without this the loop returns at
            # Python-launch speed and the .cpu() at the end becomes a
            # multi-second hang that looks like a bug. Skipped under
            # distributed runs (see sync_per_step above).
            if sync_per_step:
                _device_synchronize(args.device)

    return _drain_sampling_state(state)


def batch_generate_denoised_sampled_confidence(
        args: any,
        model: nn.Module,
        extract_digit_samples: torch.Tensor,
        extract_time: torch.Tensor,
        extract_digit_label: torch.Tensor,
        store_probabilities: bool = False,
        skip_pad: bool = False,
        sample_seeds: Optional[list] = None,
    ) -> tuple[list, list, np.ndarray | None]:
    """Confidence-based unmasking: at each step, unmask the position where the
    model's max class probability is highest among still-masked positions.

    When ``skip_pad`` is True, positions whose most-likely token is ``<PAD>``
    are deprioritised so that sequence-content positions are unmasked first.
    PAD-dominant positions are still unmasked once no other masked positions
    remain.

    Token selection is controlled by ``args.token_strategy``:
        - ``"argmax"``: deterministic (take the most-likely token)
        - ``"sample"``: stochastic (Gumbel-max trick)
    """
    state = _init_sampling_state(
        args, extract_digit_samples, extract_time, extract_digit_label,
        store_probabilities,
    )
    sync_per_step = int(getattr(args, "_world_size", 1)) <= 1

    with _inference_autocast(args.device):
        for ii in tqdm(range(state.max_diffusion_step), desc=_diffusion_tqdm_desc(args)):
            logits = _forward_logits(
                model, state.temp_mask_realization, state.temp_idx, state.temp_y_c,
            )

            # Confidence path needs normalised probs so max-prob values are
            # comparable across positions (raw logits aren't).
            probs = F.softmax(logits, dim=-1)
            if state.all_probs is not None:
                state.all_probs[ii] = probs

            max_probs, max_classes = probs.max(dim=-1)

            # Mask out already-unmasked positions so they are never selected
            is_masked = (state.temp_mask_realization.squeeze(1) == 0)
            max_probs[~is_masked] = -float("inf")

            # Deprioritise positions whose top prediction is <PAD>
            if skip_pad:
                pad_idx = getattr(args, "pad_token_id", 23)
                is_pad_top = (max_classes == pad_idx) & is_masked
                # Only suppress if there are non-PAD masked positions remaining
                has_non_pad = (is_masked & ~is_pad_top).any(dim=-1, keepdim=True)
                max_probs[is_pad_top & has_non_pad.expand_as(is_pad_top)] = -float("inf")

            # Greedy position selection
            current_location = max_probs.argmax(dim=-1)

            # Token selection — Gumbel-max on raw logits (see random-path comment).
            if state.token_strategy == "argmax":
                chosen_tokens = max_classes[state.batch_idx, current_location]
            else:
                _fill_gumbel_buffer(state.gumbel_buffer, sample_seeds, step=ii)
                sampled = (logits - state.gumbel_buffer.log()).argmax(dim=-1)
                chosen_tokens = sampled[state.batch_idx, current_location]

            state.temp_mask_realization[state.batch_idx, 0, current_location] = chosen_tokens

            # Store on GPU
            state.all_realizations[ii] = state.temp_mask_realization
            state.all_time_idx[ii] = state.temp_idx
            state.temp_idx += 1

            # See note in batch_generate_denoised_sampled — sync per step so
            # tqdm reflects actual GPU progress. Skipped under distributed runs.
            if sync_per_step:
                _device_synchronize(args.device)

    return _drain_sampling_state(state)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy / unused helpers
# ─────────────────────────────────────────────────────────────────────────────
#
# The functions below predate the batched, autocast-aware sampling path
# (``batch_generate_denoised_sampled`` and the ``_confidence`` variant)
# and are not called anywhere in the live codebase as of 2026-04-30.
# Verified via grep across src/, tests/, scripts/.
#
# Kept here, commented out, in case any of them is needed for a future
# single-batch or MNIST-style code path. Each has been style-cleaned
# (4-space indentation, ``Any`` instead of bare ``any``, MNIST-era
# inline comments dropped, return-tuple type hints corrected) so
# uncommenting yields code consistent with the rest of the module.
#
# To revive any of these: uncomment the block, add ``from typing import
# Any`` to the imports at the top of the file, and re-run the relevant
# tests.

# from typing import Any
#
# # generate missing pixels with one shot
# @torch.no_grad()
# def cond_autocomplete_real_samples(
#     model: nn.Module,
#     args: Any,
#     realization: torch.Tensor,
#     y_c: torch.Tensor,
#     idx: torch.Tensor,
# ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor,
#            torch.Tensor, torch.Tensor, torch.Tensor]:
#     model.eval()
#     bs, channel, seq_length = realization.size()
#     # random sampling paths
#     sampled_random_path = train_helper.sample_random_path(bs, seq_length, device=args.device)
#     # mask of already-sampled locations
#     random_path_mask = train_helper.create_mask_at_random_path_index(
#         sampled_random_path, idx, bs, seq_length,
#     )
#     real_tokens, bs, seq_length = train_helper.create_token_labels(args, realization)
#     real_token_masked = train_helper.mask_realizations(real_tokens, random_path_mask)
#     conditional_prob, probs = train_helper.cond_predict_conditional_prob(
#         model, real_token_masked, y_c, idx, args,
#     )
#     log_prob = train_helper.log_prob_of_realization(args, conditional_prob, real_tokens)
#     return (
#         conditional_prob,
#         probs.cpu(),
#         real_token_masked.cpu(),
#         real_tokens.cpu(),
#         log_prob.cpu(),
#         sampled_random_path.cpu(),
#         random_path_mask.cpu(),
#     )
#
#
# def extract_samples_with_labels(
#     dataloader: DataLoader,
#     target_labels: int,
#     total_num: int,
#     pad_included: bool = False,
# ) -> dict:
#     extracted_sampled = {"sample": [], "label": []}
#     for data, labels in dataloader:
#         for i, label in enumerate(labels):
#             if label.item() == target_labels:
#                 if not pad_included:
#                     # account for the absorbing state (i.e. make room)
#                     data[i] += 1
#                 extracted_sampled["sample"].append(data[i])
#                 extracted_sampled["label"].append(label)
#                 if len(extracted_sampled["label"]) == total_num:
#                     return extracted_sampled
#     return extracted_sampled
#
#
# def corrupt_samples(
#     args: Any,
#     realization: torch.Tensor,
#     perc: float,
# ) -> torch.Tensor:
#     """Mask a given percentage of the sample."""
#     bs, channels, seq_length = realization.size()
#     idx = (args.diffusion_steps * torch.Tensor([perc])).to(int).to(args.device)
#     sampled_random_path = train_helper.sample_random_path(bs, seq_length, device=args.device)
#     random_path_mask = train_helper.create_mask_at_random_path_index(
#         sampled_random_path, idx, bs, seq_length,
#     )
#     real_tokens, bs, seq_length = train_helper.create_token_labels(args, realization)
#     real_token_masked = train_helper.mask_realizations(real_tokens, random_path_mask)
#     return real_token_masked, sampled_random_path, idx
#
#
# @torch.no_grad()
# def predict_next_index(
#     model: nn.Module,
#     args: Any,
#     mask_realization: torch.Tensor,
#     y_c: torch.Tensor,
#     idx: torch.Tensor,
# ) -> tuple[Any, torch.Tensor]:
#     """Inpaint missing regions by predicting the next position."""
#     model.eval()
#     bs, channel, seq_length = mask_realization.size()
#     conditional_prob, probs = train_helper.cond_predict_conditional_prob(
#         model, mask_realization.squeeze(1), y_c, idx, args,
#     )
#     return conditional_prob, probs
#
#
# def generate_denoised_sampled(
#     args: Any,
#     model: nn.Module,
#     extract_digit_samples: torch.Tensor,
#     extract_time: torch.Tensor,
#     extract_digit_label: torch.Tensor,
#     sampling_path: torch.Tensor,
# ) -> tuple[list, list]:
#     """Single-batch sampling path. Superseded by ``batch_generate_denoised_sampled``.
#
#     Pre-rewrite reference: per-iteration ``.cpu().numpy()`` transfers,
#     no autocast, no Gumbel buffer. Use the batched version unless you
#     need the original dtype-by-dtype trace for debugging.
#     """
#     mask_realization_list, time_idx_list = [], []
#     temp_y_c = extract_digit_label.to(args.device)
#     temp_mask_realization = extract_digit_samples.unsqueeze(1).long().to(args.device)
#     temp_idx = torch.Tensor([extract_time]).to(args.device).squeeze(0)
#     temp_sampling_path = sampling_path.to(args.device)
#     for ii in tqdm(range(int(temp_idx.item()), args.diffusion_steps)):
#         current_location = temp_sampling_path == temp_idx
#         logger.debug("current_location.shape: %s", current_location.shape)
#         conditional_prob, prob = predict_next_index(
#             model=model,
#             args=args,
#             mask_realization=temp_mask_realization,
#             y_c=temp_y_c,
#             idx=temp_idx,
#         )
#         next_temp_realization = torch.argmax(conditional_prob.sample(), dim=-1)
#         temp_mask_realization[0, current_location] = next_temp_realization[current_location]
#         mask_realization_list.append(temp_mask_realization.cpu().numpy())
#         time_idx_list.append(temp_idx.cpu().numpy())
#         temp_idx += 1
#     return mask_realization_list, time_idx_list
#
#
# def convert_num_to_chars(
#     tokenizer: Any,
#     num_seq: list,
# ) -> list:
#     """Convert a numerical token sequence to a character string.
#
#     Duplicate of ``biom3.Stage3.animation_tools.convert_num_to_char``
#     (singular), which is the version used everywhere else. Prefer
#     the singular form.
#     """
#     char_seq = [tokenizer[num] for num in num_seq]
#     return "".join(char_seq)
