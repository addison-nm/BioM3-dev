import os
import numpy as np
import random
import pandas as pd
import math
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import biom3.Stage3.preprocess as prep

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)
#mport source.sampling as sample_tools
import biom3.Stage3.cond_diff_transformer_layer as mod
import biom3.Stage3.transformer_training_helper as train_helper


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



# generate missing pixels with one shot
@torch.no_grad()
def cond_autocomplete_real_samples(
        model: nn.Module,
        args: any,
        realization: torch.Tensor,
        y_c: torch.Tensor,
        idx: torch.Tensor
    ) -> tuple[
            any,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor
    ]:

        model.eval()
        bs, channel, seq_length = realization.size()
        # get a batch of random sampling paths
        sampled_random_path = train_helper.sample_random_path(bs, seq_length, device=args.device)
        # create a mask that masks the locations where we've already sampled
        random_path_mask = train_helper.create_mask_at_random_path_index(sampled_random_path, idx, bs, seq_length)
        # tokenize realizations 
        real_tokens, bs, seq_length= train_helper.create_token_labels(args, realization)
        #real_tokens = realization.clone().squeeze(1)

        # mask realizations 
        real_token_masked =  train_helper.mask_realizations(real_tokens, random_path_mask)
        # conditional probability
        conditional_prob, probs = train_helper.cond_predict_conditional_prob(model, real_token_masked, y_c, idx, args)
        # evaluate the value of the log probability for the given realization:
        log_prob = train_helper.log_prob_of_realization(args, conditional_prob, real_tokens)
        
        return (
                conditional_prob,
                probs.cpu(),
                real_token_masked.cpu(),
                real_tokens.cpu(),
                log_prob.cpu(),
                sampled_random_path.cpu(),
                random_path_mask.cpu()
        )


# get the label for the corresponding sequence in the dataloader
def extract_samples_with_labels(
        dataloader: DataLoader,
        target_labels: int,
        total_num: int,
        pad_included: bool=False
    ) -> dict:

    extracted_sampled = {
            'sample': [],
            'label': []
    }

    for data, labels in dataloader:
        for i, label in enumerate(labels):

            if label.item() == target_labels:

                if pad_included:
                    pass
                else:
                    data[i] += 1 # account for the absorbing state (i.e. make room)

                extracted_sampled['sample'].append(data[i]) # account for abosrbed state
                extracted_sampled['label'].append(label)
                if len(extracted_sampled['label']) == total_num:
                    return extracted_sampled

    return extracted_sampled


# mask a given percentage of the sample
def corrupt_samples(
        args: any,
        realization: torch.Tensor,
        perc: float
    ) -> torch.Tensor:

    bs, channels, seq_length = realization.size()

    # number of samples to corrupt (i.e. idx)
    idx = (args.diffusion_steps * torch.Tensor([perc])).to(int).to(args.device)
    # get a batch of random sampling paths
    sampled_random_path = train_helper.sample_random_path(bs, seq_length, device=args.device)
    # we create a mask that masks the locations where we've already sampled
    random_path_mask = train_helper.create_mask_at_random_path_index(sampled_random_path, idx, bs, seq_length)
    # tokenize realizations 
    real_tokens, bs, seq_length= train_helper.create_token_labels(args, realization)
    # mask realizations 
    real_token_masked = train_helper.mask_realizations(real_tokens, random_path_mask)

    return (
            real_token_masked,
            sampled_random_path,
            idx
    )

# inpaint missing regions by predicting the next position
@torch.no_grad()
def predict_next_index(
        model: nn.Module,
        args: any,
        mask_realization: torch.Tensor,
        y_c: torch.Tensor,
        idx: torch.Tensor
    ) -> tuple[
            any,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor
    ]:

        model.eval()
        bs, channel, seq_length = mask_realization.size()

        # conditional prob
        conditional_prob, probs = train_helper.cond_predict_conditional_prob(model, mask_realization.squeeze(1), y_c, idx, args)
        
        return (
                conditional_prob,
                probs,
        )




def generate_denoised_sampled(
        args: any,
        model: nn.Module,
        extract_digit_samples: torch.Tensor,
        extract_time: torch.Tensor,
        extract_digit_label: torch.Tensor,
        sampling_path: torch.Tensor
    ) -> tuple[
            list,
            list
    ]:

        mask_realization_list, time_idx_list = [], []

        # prepare data
        temp_y_c = extract_digit_label.to(args.device)
        temp_mask_realization = extract_digit_samples.unsqueeze(1).long().to(args.device)
        temp_idx = torch.Tensor([extract_time]).to(args.device).squeeze(0)
        temp_sampling_path = sampling_path.to(args.device)
        
        for ii in tqdm(range(int(temp_idx.item()), args.diffusion_steps)):
            
            # where we need to sample next
            current_location = temp_sampling_path == temp_idx
            logger.debug("current_location.shape: %s", current_location.shape)

            # make position prediction
            conditional_prob, prob  = predict_next_index(
                    model=model,
                    args=args,
                    mask_realization=temp_mask_realization,
                    y_c=temp_y_c,
                    idx=temp_idx
            )

            # get the label for the next token position
            next_temp_realization = torch.argmax(
                    conditional_prob.sample(), dim=-1
            )

            temp_mask_realization[0, current_location] = next_temp_realization[current_location]
            mask_realization_list.append(temp_mask_realization.cpu().numpy())
            time_idx_list.append(temp_idx.cpu().numpy())
            temp_idx+=1


        return (
                mask_realization_list,
                time_idx_list
        )


def batch_generate_denoised_sampled(
        args: any,
        model: nn.Module,
        extract_digit_samples: torch.Tensor,
        extract_time: torch.Tensor,
        extract_digit_label: torch.Tensor,
        sampling_path: torch.Tensor,
        store_probabilities: bool = False,
    ) -> tuple[list, list, np.ndarray | None]:

    # Ensure batch dimension consistency across input tensors
    assert extract_digit_samples.size(0) == extract_digit_label.size(0) == sampling_path.size(0) == extract_time.size(0), "Mismatched batch dimensions"

    batch_size = extract_digit_samples.size(0)
    seq_len = extract_digit_samples.size(1)
    logger.debug('batch_size: %s', batch_size)

    # Prepare data
    temp_y_c = extract_digit_label.to(args.device)
    temp_mask_realization = extract_digit_samples.unsqueeze(1).long().to(args.device)
    temp_idx = extract_time.unsqueeze(-1).to(args.device)
    temp_sampling_path = sampling_path.to(args.device)

    max_diffusion_step = args.diffusion_steps
    batch_idx = torch.arange(batch_size, device=args.device)
    token_strategy = getattr(args, 'token_strategy', 'sample')

    # Pre-allocate GPU tensors to avoid per-iteration CPU transfers
    all_realizations = torch.empty(
        max_diffusion_step, batch_size, 1, seq_len,
        dtype=temp_mask_realization.dtype, device=args.device
    )
    all_time_idx = torch.empty(
        max_diffusion_step, batch_size, 1,
        dtype=temp_idx.dtype, device=args.device
    )

    if store_probabilities:
        all_probs = torch.empty(
            max_diffusion_step, batch_size, seq_len, args.num_classes,
            dtype=torch.float32, device=args.device
        )

    if token_strategy == 'sample':
        # Pre-allocate Gumbel noise buffer (reused each step via in-place fill).
        # See docs/gumbel_max_sampling.md for derivation.
        gumbel_buffer = torch.empty(
            batch_size, seq_len, args.num_classes,
            dtype=temp_y_c.dtype, device=args.device
        )

    with _inference_autocast(args.device):
        for ii in tqdm(range(max_diffusion_step)):

            # Model forward returns logits as [batch, num_classes, seq_len];
            # transpose once so downstream ops work on [batch, seq_len, num_classes].
            # Promote to fp32 so softmax/Gumbel math stays stable under autocast.
            logits = model(
                x=temp_mask_realization.squeeze(1),
                t=temp_idx.view(-1),
                y_c=temp_y_c,
            ).transpose(1, 2).float()

            if store_probabilities:
                all_probs[ii] = F.softmax(logits, dim=-1)

            # Token selection — softmax is monotonic so argmax(log_softmax + G)
            # == argmax(logits + G). See docs/gumbel_max_sampling.md.
            if token_strategy == 'argmax':
                next_temp_realization = logits.argmax(dim=-1)
            else:
                gumbel_buffer.exponential_()
                next_temp_realization = (logits - gumbel_buffer.log()).argmax(dim=-1)

            # Update temp_mask_realization per sample (advanced indexing)
            current_location = (temp_sampling_path == temp_idx).long().argmax(dim=-1)
            temp_mask_realization[batch_idx, 0, current_location] = next_temp_realization[batch_idx, current_location]

            # Store on GPU
            all_realizations[ii] = temp_mask_realization
            all_time_idx[ii] = temp_idx

            # Increment temp_idx for the next iteration
            temp_idx += 1

    # Single GPU→CPU transfer at the end
    all_realizations_np = all_realizations.cpu().numpy()
    all_time_idx_np = all_time_idx.cpu().numpy()
    mask_realization_list = [all_realizations_np[i] for i in range(max_diffusion_step)]
    time_idx_list = [all_time_idx_np[i] for i in range(max_diffusion_step)]

    probs_np = all_probs.cpu().numpy() if store_probabilities else None

    return mask_realization_list, time_idx_list, probs_np


def batch_generate_denoised_sampled_confidence(
        args: any,
        model: nn.Module,
        extract_digit_samples: torch.Tensor,
        extract_time: torch.Tensor,
        extract_digit_label: torch.Tensor,
        store_probabilities: bool = False,
        skip_pad: bool = False,
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

    batch_size = extract_digit_samples.size(0)
    seq_len = extract_digit_samples.size(1)
    logger.debug('batch_size: %s', batch_size)

    # Prepare data
    temp_y_c = extract_digit_label.to(args.device)
    temp_mask_realization = extract_digit_samples.unsqueeze(1).long().to(args.device)
    temp_idx = extract_time.unsqueeze(-1).to(args.device)

    max_diffusion_step = args.diffusion_steps
    batch_idx = torch.arange(batch_size, device=args.device)

    token_strategy = getattr(args, 'token_strategy', 'sample')

    # Pre-allocate GPU tensors
    all_realizations = torch.empty(
        max_diffusion_step, batch_size, 1, seq_len,
        dtype=temp_mask_realization.dtype, device=args.device
    )
    all_time_idx = torch.empty(
        max_diffusion_step, batch_size, 1,
        dtype=temp_idx.dtype, device=args.device
    )

    if store_probabilities:
        all_probs = torch.empty(
            max_diffusion_step, batch_size, seq_len, args.num_classes,
            dtype=torch.float32, device=args.device
        )

    if token_strategy == 'sample':
        gumbel_buffer = torch.empty(
            batch_size, seq_len, args.num_classes,
            dtype=temp_y_c.dtype, device=args.device
        )

    with _inference_autocast(args.device):
        for ii in tqdm(range(max_diffusion_step)):

            # Model forward returns logits as [batch, num_classes, seq_len];
            # transpose once so downstream ops work on [batch, seq_len, num_classes].
            # Promote to fp32 so softmax/Gumbel math stays stable under autocast.
            logits = model(
                x=temp_mask_realization.squeeze(1),
                t=temp_idx.view(-1),
                y_c=temp_y_c,
            ).transpose(1, 2).float()

            # Confidence path needs normalised probs so max-prob values are
            # comparable across positions (raw logits aren't).
            probs = F.softmax(logits, dim=-1)
            if store_probabilities:
                all_probs[ii] = probs

            max_probs, max_classes = probs.max(dim=-1)

            # Mask out already-unmasked positions so they are never selected
            is_masked = (temp_mask_realization.squeeze(1) == 0)
            max_probs[~is_masked] = -float('inf')

            # Deprioritise positions whose top prediction is <PAD>
            if skip_pad:
                pad_idx = getattr(args, 'pad_token_id', 23)
                is_pad_top = (max_classes == pad_idx) & is_masked
                # Only suppress if there are non-PAD masked positions remaining
                has_non_pad = (is_masked & ~is_pad_top).any(dim=-1, keepdim=True)
                max_probs[is_pad_top & has_non_pad.expand_as(is_pad_top)] = -float('inf')

            # Greedy position selection
            current_location = max_probs.argmax(dim=-1)

            # Token selection — Gumbel-max on raw logits (see random-path comment).
            if token_strategy == 'argmax':
                chosen_tokens = max_classes[batch_idx, current_location]
            else:
                gumbel_buffer.exponential_()
                sampled = (logits - gumbel_buffer.log()).argmax(dim=-1)
                chosen_tokens = sampled[batch_idx, current_location]

            temp_mask_realization[batch_idx, 0, current_location] = chosen_tokens

            # Store on GPU
            all_realizations[ii] = temp_mask_realization
            all_time_idx[ii] = temp_idx

            temp_idx += 1

    # Single GPU→CPU transfer at the end
    all_realizations_np = all_realizations.cpu().numpy()
    all_time_idx_np = all_time_idx.cpu().numpy()
    mask_realization_list = [all_realizations_np[i] for i in range(max_diffusion_step)]
    time_idx_list = [all_time_idx_np[i] for i in range(max_diffusion_step)]

    probs_np = all_probs.cpu().numpy() if store_probabilities else None

    return mask_realization_list, time_idx_list, probs_np


# convert sequence with numerical variables into character letters
def convert_num_to_chars(
        tokenizer: any,
        num_seq: list
    ) -> list:

    char_seq = [tokenizer[num] for num in num_seq]
    return "".join(char_seq)
