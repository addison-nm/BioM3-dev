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

import Stage3_source.preprocess as prep
#mport source.sampling as sample_tools
import Stage3_source.cond_diff_transformer_layer as mod
import Stage3_source.transformer_training_helper as train_helper



# generate missing pixels with one shot
@torch.no_grad()
def cond_autocomplete_real_samples(
        model: nn.Module,
        args: any,
        realization: torch.Tensor,
        y_c: torch.Tensor,
        idx: torch.Tensor
    ) -> (
            any,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor
    ):

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
    ) -> (
            any,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor
    ):

        model.eval()
        bs, channel, seq_length = mask_realization.size()

        # conditional prob
        conditional_prob, probs = train_helper.cond_predict_conditional_prob(model, mask_realization.squeeze(1), y_c, idx, args)
        
        return (
                conditional_prob,
                probs.cpu(),
        )




def generate_denoised_sampled(
        args: any,
        model: nn.Module,
        extract_digit_samples: torch.Tensor,
        extract_time: torch.Tensor,
        extract_digit_label: torch.Tensor,
        sampling_path: torch.Tensor
    ) -> (
            list,
            list
    ):

        mask_realization_list, time_idx_list = [], []

        # prepare data
        temp_y_c = extract_digit_label.to(args.device)
        temp_mask_realization = extract_digit_samples.unsqueeze(1).long().to(args.device)
        temp_idx = torch.Tensor([extract_time]).to(args.device).squeeze(0)
        temp_sampling_path = sampling_path.to(args.device)
        
        for ii in tqdm(range(int(temp_idx.item()), args.diffusion_steps)):
            
            # where we need to sample next
            current_location = temp_sampling_path == temp_idx
            print(current_location.shape)

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
        sampling_path: torch.Tensor
    ) -> (list, list):

    # Ensure batch dimension consistency across input tensors
    assert extract_digit_samples.size(0) == extract_digit_label.size(0) == sampling_path.size(0) == extract_time.size(0), "Mismatched batch dimensions"

    batch_size = extract_digit_samples.size(0)
    mask_realization_list, time_idx_list = [], []
    print('batch_size:', batch_size)

    # Prepare data
    temp_y_c = extract_digit_label.to(args.device)
    temp_mask_realization = extract_digit_samples.unsqueeze(1).long().to(args.device)
    temp_idx = extract_time.unsqueeze(-1).to(args.device)  # Adding an extra dimension for batch processing
    temp_sampling_path = sampling_path.to(args.device)

    max_diffusion_step = args.diffusion_steps
    for ii in tqdm(range(max_diffusion_step)):
        
        # Broadcast ii to match the batch size
        current_ii = torch.full((batch_size,), ii, dtype=torch.long, device=args.device)

        # Make position prediction
        conditional_prob, prob = predict_next_index(
                model=model,
                args=args,
                mask_realization=temp_mask_realization,
                y_c=temp_y_c,
                idx=temp_idx
        )
    

        # Get the label for the next token position
        next_temp_realization = torch.argmax(conditional_prob.sample(), dim=-1)

        # Update temp_mask_realization for each item in the batch
        current_location = temp_sampling_path == temp_idx  # Adding an extra dimension for comparison
        current_location = torch.argmax(current_location.detach().cpu()*1, dim=-1)
        temp_mask_realization[:, 0, current_location] = next_temp_realization[:,current_location]
        
        # Append results for each item in the batch
        mask_realization_list.append(temp_mask_realization.cpu().numpy())
        time_idx_list.append(temp_idx.cpu().numpy())

        # Increment temp_idx for the next iteration
        temp_idx += 1

    return mask_realization_list, time_idx_list



# convert sequence with numerical variables into character letters
def convert_num_to_chars(
        tokenizer: any,
        num_seq: list
    ) -> list:

    char_seq = [tokenizer[num] for num in num_seq]
    return "".join(char_seq)
