"""
Description:
    Metrics to compute protein diffusion model performance.
    This module provides functions for evaluating model accuracy, perplexity,
    and sequence quality at different stages of the diffusion process.
"""

# ----- Bioinformatics Libraries -----
import Bio
from Bio.Align import substitution_matrices  # For protein sequence scoring matrices

# ----- Numerical and Visualization Libraries -----
import numpy as np
import matplotlib.pyplot as plt  # For visualization of metrics

# ----- PyTorch for Model Evaluation -----
import torch

# ----- Text Processing -----
import re  # For regular expressions in sequence processing

# ----- Custom Modules -----
import biom3.Stage3.animation_tools as ani_tools  # For visualization of generation process



class blosum_soft_accuracy:
    """
    Evaluate protein sequence predictions using BLOSUM62-based soft accuracy metrics.
    
    This class implements biologically-informed accuracy metrics for comparing
    predicted and reference protein sequences. Instead of using simple exact match
    (hard accuracy), it employs the BLOSUM62 substitution matrix to calculate
    "soft accuracy" that accounts for biochemically similar amino acid substitutions.
    
    The BLOSUM62 matrix captures the likelihood of amino acid substitutions in 
    evolutionarily related proteins, allowing the metric to give partial credit
    when the model predicts amino acids that are biochemically similar to the
    reference. This better reflects biological reality where certain substitutions
    (e.g., between hydrophobic amino acids) are more acceptable than others.
    
    The class handles special tokens (START, END, PAD) with standard exact matching,
    and applies the soft accuracy only to standard amino acid positions.
    """

    def __init__(self):
        """
        Initialize the soft accuracy evaluator with the BLOSUM62 substitution matrix.
        """
        self.blosum62 = substitution_matrices.load("BLOSUM62")
        self.alphabet = self.blosum62.alphabet

    def blosum_acc(
            self,
            aa1: str,
            aa2: str
        ) -> np.single:
        """
        Calculate soft accuracy between two amino acids using BLOSUM62 scores.
        
        This method computes a similarity score between the predicted amino acid (aa1)
        and the reference amino acid (aa2) by:
        1. Converting BLOSUM62 substitution scores to probabilities via softmax
        2. Computing dot product with one-hot encoding of the reference amino acid
        3. Normalizing by the maximum probability for scale consistency
        
        Args:
            aa1: Predicted amino acid (single letter code)
            aa2: Reference amino acid (single letter code)
            
        Returns:
            float: Soft accuracy score between 0 and 1
        """
        row = self.blosum62.alphabet.index(aa1)
        col = self.blosum62.alphabet.index(aa2)
        substitution_scores = self.blosum62[row, :].values()

        # Apply the softmax function to the substitution scores to get a prob dist.
        probs = np.exp(substitution_scores)/np.sum(np.exp(substitution_scores))

        # compute the soft acc. as the dot product of the prob dist. with a one-hot encoding
        # of the amino acid ...
        correct_aa = aa2
        correct_index = self.alphabet.index(correct_aa)
        one_hot = np.zeros_like(probs)
        one_hot[correct_index] = 1

        # normalize acc.
        soft_acc = np.dot(probs, one_hot) / np.max(probs)

        return soft_acc

    def split_seq(self, seq: str) -> list:
        """
        Split a protein sequence string into individual tokens.
        
        This method uses regular expressions to parse a protein sequence,
        handling both standard amino acids and special tokens like <START>,
        <END>, <PAD>, and gap characters (-).
        
        Args:
            seq: Protein sequence string to split
            
        Returns:
            list: Sequence split into individual tokens
        """
        split_seq = re.findall(r'<START>|<END>|<PAD>|[A-Z]|-|\*', seq)

        # remove empty strings and whitespace-only elements
        split_seq = [char for char in split_seq if char and char.strip()]
        return split_seq

    def compute_soft_accuracy(
            self,
            seq1_list: list,
            seq2_list: list
        ) -> float:
        """
        Compute average soft accuracy between two lists of protein sequences.
        
        This method calculates both soft accuracy (using BLOSUM62) for standard
        amino acids and hard accuracy (exact match) for special tokens, then
        combines them to produce a final accuracy score for each sequence pair.
        
        Args:
            seq1_list: List of predicted protein sequences
            seq2_list: List of reference protein sequences
            
        Returns:
            float: Average soft accuracy across all sequence pairs
            
        Raises:
            Warning: If sequence lengths don't match (attempts to proceed anyway)
        """
        # make sure batch size matches
        if len(seq1_list) == len(seq2_list):
            self.batch_size = len(seq1_list)
        else:
            print("Please make sequence batch size equivalent...")

        # make sure sequence length matches
        if len(seq1_list[0]) == len(seq2_list[0]):
            self.L = len(seq1_list[0])
        else:
            #print("Please make sequence length match...")
            pass

        avg_soft_acc_per_batch = 0
        # loop over the batch of sequence
        for seq1, seq2 in zip(seq1_list, seq2_list):

            # split sequence into individual tokens
            seq1 = self.split_seq(seq1)
            seq2 = self.split_seq(seq2)
            # set number of positions
            self.L = len(seq2)
            self.L_h = 0  # Count of positions requiring hard accuracy
            self.L_s = 0  # Count of positions requiring soft accuracy
            avg_soft_acc_per_seq = 0
            avg_hard_acc_per_seq = 0

            # loop over the amino acid positions
            for aa1, aa2 in zip(seq1, seq2):
                # Standard amino acids use soft accuracy
                if (aa1 not in ['-', '<START>', '<END>', '<PAD>']) and (aa2 not in ['-', '<START>', '<END>', '<PAD>']):
                    self.L_s += 1
                    soft_acc = self.blosum_acc(aa1=aa1, aa2=aa2)
                    avg_soft_acc_per_seq += soft_acc
                # Special tokens use hard (exact match) accuracy
                else:
                    self.L_h += 1
                    acc = 1*(aa1==aa2)  # 1 if match, 0 otherwise
                    avg_hard_acc_per_seq += acc

            # compute accuracy for soft positions
            try:
                avg_soft_acc_per_seq *= 1/self.L_s
            except ZeroDivisionError:
                #print("L_s cannot be zero. Setting avg_soft_acc_per_seq to zero.")
                avg_soft_acc_per_seq = 0

            # compute accuracy for hard positions
            try:
                avg_hard_acc_per_seq *= 1/self.L_h
            except ZeroDivisionError:
                #print("L_h cannot be zero. Setting avg_hard_acc_per_seq to zero.")
                avg_hard_acc_per_seq = 0

            # compute the average accuracy between soft and hard
            if self.L_s == 0:
                avg_soft_acc_per_batch += avg_hard_acc_per_seq
            elif self.L_h == 0:
                avg_soft_acc_per_batch += avg_soft_acc_per_seq
            else:
                avg_soft_acc_per_batch += (avg_soft_acc_per_seq + avg_hard_acc_per_seq)/2

        avg_soft_acc_per_batch *= 1/self.batch_size
        return avg_soft_acc_per_batch


def compute_ppl(probs: torch.Tensor) -> float:
    """
    Calculate perplexity from token probability distributions.
    
    Perplexity is a measure of how "surprised" a model is by the data - 
    lower values indicate better predictions. It's calculated as the 
    exponential of the entropy of the probability distribution.
    
    Mathematically: perplexity = exp(entropy) = exp(-sum(p_i * log(p_i)))
    
    Args:
        probs: Tensor of shape [batch_size, sequence_length, class_labels]
              containing probability distributions over tokens at each position
              
    Returns:
        float: Average perplexity across all positions in all sequences
    """
    batch_size, sequence_length, class_labels = probs.shape
    # flatten batch and sequence dimensions into a single dimension
    flattened_probs = probs.reshape(batch_size * sequence_length, class_labels)
    # calc. perplexity for each sequence independently
    ppl = []
    for i in range(batch_size * sequence_length):
        sequence_probs = flattened_probs[i]
        # compute ppl per seq
        sequence_ppl = torch.exp(-torch.sum(
            sequence_probs * torch.log(sequence_probs)
            )
        )
        ppl.append(sequence_ppl.item())
    ppl = torch.tensor(ppl).view(batch_size, sequence_length) # ppl per sequence in a given batch
    avg_ppl = ppl.mean().item() # average ppl per batch
    return avg_ppl

def batch_compute_ppl(probs_list: list) -> float:
    """
    Calculate average perplexity across a list of probability distributions.
    
    This utility function applies the perplexity calculation to multiple
    probability tensors and returns their average. It's useful for evaluating
    perplexity across multiple batches or for different parts of sequences.
    
    Args:
        probs_list: List of tensors, each of shape [class_labels, sequence_length]
                   containing probability distributions
                   
    Returns:
        float: Average perplexity across all tensors in the list
    """
    batch_prob = sum([
        compute_ppl(probs=probs.unsqueeze(0).permute(0,2,1)) for probs in probs_list
    ]) / len(probs_list)
    return batch_prob

def compute_hard_acc(
        seq1: str,
        seq2: str
    ) -> float:
    """
    Calculate hard (exact match) accuracy between two protein sequences.
    
    Hard accuracy measures the fraction of positions where the predicted 
    sequence (seq1) exactly matches the reference sequence (seq2), ignoring
    gap positions ('-') in the reference.
    
    Args:
        seq1: Predicted protein sequence
        seq2: Reference protein sequence
        
    Returns:
        float: Fraction of non-gap positions that match exactly (0.0 to 1.0)
        
    Note:
        If the reference sequence consists entirely of gaps, the function
        returns 1.0 (perfect accuracy) by convention.
    """
    hard_acc = sum([aa1 == aa2 for (aa1 ,aa2) in zip(seq1, seq2) if aa2 != '-'])
    valid_length = len([aa2 for aa2 in seq2 if aa2 != '-'])
    if valid_length == 0:
        return 1.0
    hard_acc /= valid_length
    return hard_acc

def batch_hard_acc(seq1_list: list, seq2_list: list) -> float:
    """
    Calculate average hard accuracy across multiple sequence pairs.
    
    This function applies the hard accuracy calculation to multiple
    sequence pairs and returns their average, useful for evaluating
    model performance on a batch of sequences.
    
    Args:
        seq1_list: List of predicted protein sequences
        seq2_list: List of reference protein sequences
        
    Returns:
        float: Average hard accuracy across all sequence pairs
    """
    hard_acc = sum([
        compute_hard_acc(seq1=seq1, seq2=seq2) for (seq1,seq2) in zip(seq1_list, seq2_list)
    ]) / len(seq2_list)
    return hard_acc

def time_split_on_seq(
    seq: torch.Tensor,
    sample_seq_path: torch.Tensor,
    idx: torch.Tensor
    ) -> (
        list,
        list,
        list
    ):
    """
    Split sequences or probability tensors based on diffusion timestep.
    
    This function divides the input tensor into three parts based on the 
    sampling path and current timestep:
    - Current: Positions currently being sampled (sample_seq_path == idx)
    - Previous: Positions already sampled (sample_seq_path < idx)
    - Future: Positions yet to be sampled (sample_seq_path > idx)
    
    The function handles both:
    1. Token tensors of shape [batch_size, sequence_length]
    2. Probability tensors of shape [batch_size, class_labels, sequence_length]
    
    Args:
        seq: Input tensor containing tokens or probabilities
        sample_seq_path: Tensor of shape [batch_size, sequence_length] containing
                        the random path indices for each position
        idx: Tensor of shape [batch_size] containing current timestep indices
        
    Returns:
        tuple: (
            current_seq: List of tensors for positions at current timestep
            prev_seq: List of tensors for positions before current timestep
            fut_seq: List of tensors for positions after current timestep
        )
    """
    if len(seq.shape) != 2:
        # Handle probability tensors with shape [batch_size, class_labels, sequence_length]
        batch_size, class_labels, _ = seq.shape
        # collect list
        current_seq, prev_seq, fut_seq = [], [], []
        for ii in range(batch_size):
            current_stack_probs, prev_stack_probs, fut_stack_probs = [], [], []
            for jj in range(class_labels):
                # current probs
                current_stack_probs.append(
                    seq[ii,jj][
                        (sample_seq_path.cpu()[ii] == idx.cpu()[ii])
                    ]
                )
                # prev probs
                prev_stack_probs.append(
                    seq[ii,jj][
                        (sample_seq_path.cpu()[ii] < idx.cpu()[ii])
                    ]
                )
                # future probs
                fut_stack_probs.append(
                    seq[ii,jj][
                        (sample_seq_path.cpu()[ii] > idx.cpu()[ii])
                    ]
                )
            current_seq.append(torch.stack(current_stack_probs))
            prev_seq.append(torch.stack(prev_stack_probs))
            fut_seq.append(torch.stack(fut_stack_probs))
    else:
        # Handle token tensors with shape [batch_size, sequence_length]
        # Split the sequences based on time indices
        current_seq = [seq[ii][sample_seq_path[ii] == idx[ii]] for ii in range(seq.shape[0])]
        prev_seq = [seq[ii][sample_seq_path[ii] < idx[ii]] for ii in range(seq.shape[0])]
        fut_seq = [seq[ii][sample_seq_path[ii] > idx[ii]] for ii in range(seq.shape[0])]
    
    return (
        current_seq,
        prev_seq,
        fut_seq
    )


@torch.no_grad()
def compute_acc_given_time_pos(
    real_tokens: torch.Tensor,
    sample_seq: torch.Tensor,
    sample_path: torch.Tensor,
    idx: torch.Tensor
    ) -> (
    float,
    float,
    float,
    float,
    float,
    float
    ):
    """
    Compute accuracy metrics at different positions in the diffusion sampling process.
    
    This function evaluates model performance by calculating accuracy metrics for:
    - Previous positions: Tokens already sampled in previous timesteps (t < idx)
    - Current position: Token being sampled at the current timestep (t = idx)
    - Future positions: Tokens that will be sampled in future timesteps (t > idx)
    
    For each position category, both hard accuracy (exact match) and soft accuracy 
    (BLOSUM62-based similarity) are calculated. The soft accuracy gives partial credit 
    for predicting biochemically similar amino acids, which is important for protein
    sequence evaluation.
    
    Note: The @torch.no_grad() decorator ensures no gradients are tracked during
    evaluation, improving memory efficiency.
    
    Args:
        real_tokens: Ground truth token indices with shape [batch_size, seq_length]
        sample_seq: Model-sampled token indices with shape [batch_size, seq_length]
        sample_path: Random sampling path indices with shape [batch_size, seq_length]
        idx: Current timestep indices with shape [batch_size]
        
    Returns:
        tuple: (
            prev_batch_hard_acc: Hard accuracy for previously sampled positions
            prev_batch_soft_acc: Soft accuracy for previously sampled positions
            fut_batch_hard_acc: Hard accuracy for future positions
            fut_batch_soft_acc: Soft accuracy for future positions
            current_hard_acc: Hard accuracy for current position
            current_soft_acc: Soft accuracy for current position
        )
    """

    # tokenizer
    tokens = ['*', '<START>', 'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','<END>','-']
    #tokens = ['<START>', 'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','<END>','<PAD>']
    tokens = tokens + ['X', 'U', 'Z', 'B', 'O']


    # split real tokens based on time indices
    current_real_tokens, prev_real_tokens, fut_real_tokens = time_split_on_seq(
        seq=real_tokens.cpu(),
        sample_seq_path=sample_path.cpu(),
        idx=idx.cpu()
    )

    # split sampled tokens based on time indices
    current_sample_tokens, prev_sample_tokens, fut_sample_tokens = time_split_on_seq(
        seq=sample_seq.cpu(),
        sample_seq_path=sample_path.cpu(),
        idx=idx.cpu()
    )

    # convert real sequences to characters
    current_real_chars = [ani_tools.convert_num_to_char(tokens,seq_tokens) for seq_tokens in current_real_tokens]
    prev_real_chars = [ani_tools.convert_num_to_char(tokens,seq_tokens) for seq_tokens in prev_real_tokens]
    fut_real_chars = [ani_tools.convert_num_to_char(tokens,seq_tokens) for seq_tokens in fut_real_tokens]

    # convert sample sequences to characters
    current_sample_chars = [ani_tools.convert_num_to_char(tokens,seq_tokens) for seq_tokens in current_sample_tokens]
    prev_sample_chars = [ani_tools.convert_num_to_char(tokens,seq_tokens) for seq_tokens in prev_sample_tokens]
    fut_sample_chars = [ani_tools.convert_num_to_char(tokens,seq_tokens) for seq_tokens in fut_sample_tokens]


    # drop empty entries in list (happens if t=0 or t=D)
    # prev string sequences
    prev_sample_chars = [item for item in prev_sample_chars if item]
    prev_real_chars = [item for item in prev_real_chars if item]
    # fut string sequences
    fut_real_chars = [item for item in fut_real_chars if item]
    fut_sample_chars = [item for item in fut_sample_chars if item]

    # class object to copmute blosum62 soft acc.
    soft_acc_tool = blosum_soft_accuracy()

    # split real sequence
    prev_real_split_chars = [
        soft_acc_tool.split_seq(sample) for sample in prev_real_chars
    ]
    fut_real_split_chars = [
        soft_acc_tool.split_seq(sample) for sample in fut_real_chars
    ]

    # split sample sequence
    prev_sample_split_chars = [
        soft_acc_tool.split_seq(sample) for sample in prev_sample_chars
    ]
    fut_sample_split_chars = [
        soft_acc_tool.split_seq(sample) for sample in fut_sample_chars
    ]

    # compute hard and soft accuracy
    ' soft accuracy: '
    # positions < t ( aa positions)
    #prev_batch_soft_acc = soft_acc_tool.compute_soft_accuracy(
    #    seq1_list=prev_sample_chars,
    #    seq2_list=prev_real_chars
    #)

    # positions > t ( aa positions)
    #fut_batch_soft_acc = soft_acc_tool.compute_soft_accuracy(
    #    seq1_list=fut_sample_chars,
    #    seq2_list=fut_real_chars
    #)

    # positions = t (aa positions)
    #current_soft_acc = soft_acc_tool.compute_soft_accuracy(
    #seq1_list=current_sample_chars,
    #seq2_list=current_real_chars
    #)

    # Soft accuracy calculations are commented out in the original code
    prev_batch_soft_acc, fut_batch_soft_acc, current_soft_acc = 0, 0, 0

    ' hard accuracy: '
    # positions < t ( aa positions)
    prev_batch_hard_acc = batch_hard_acc(
        seq1_list=prev_sample_split_chars,
        seq2_list=prev_real_split_chars
    )

    # positions > t ( aa positions)
    fut_batch_hard_acc = batch_hard_acc(
        seq1_list=fut_sample_split_chars,
        seq2_list=fut_real_split_chars
    )

    # positions = t (aa positions)
    current_hard_acc = compute_hard_acc(
        seq1=current_sample_chars,
        seq2=current_real_chars
    )


    return (
    prev_batch_hard_acc,
    prev_batch_soft_acc,
    fut_batch_hard_acc,
    fut_batch_soft_acc,
    current_hard_acc,
    current_soft_acc
    )


@torch.no_grad()
def compute_ppl_given_time_pos(
        probs: torch.Tensor,
        sample_path: torch.Tensor,
        idx: torch.Tensor
    ) -> (
            float,
            float,
            float
    ):
    """
    Compute perplexity at different positions in the diffusion sampling process.
    
    Perplexity measures the model's uncertainty in its predictions - lower
    values indicate higher confidence. This function evaluates perplexity for:
    - Current positions: Tokens being sampled at the current timestep (t = idx)
    - Previous positions: Tokens already sampled in previous timesteps (t < idx)
    - Future positions: Tokens that will be sampled in future timesteps (t > idx)
    
    By comparing perplexity across these position categories, we can analyze
    how the model's confidence changes throughout the diffusion process.
    
    Note: The @torch.no_grad() decorator ensures no gradients are tracked during
    evaluation, improving memory efficiency.
    
    Args:
        probs: Model probability outputs with shape [batch_size, num_tokens, seq_length]
        sample_path: Random sampling path indices with shape [batch_size, seq_length]
        idx: Current timestep indices with shape [batch_size]
        
    Returns:
        tuple: (
            current_ppl: Perplexity at current positions
            prev_ppl: Perplexity at previously sampled positions
            fut_ppl: Perplexity at future positions
        )
    """
    current_probs, prev_probs, fut_probs = time_split_on_seq(
            probs.cpu(),
            sample_seq_path=sample_path.cpu(),
            idx=idx.cpu()
    )
    # ppl at the current time position (aa_i = t)
    # current_ppl = compute_ppl(probs=torch.stack(current_probs).permute(0,2,1))
    current_ppl = batch_compute_ppl(probs_list=current_probs)
    # ppl at the prev and fut time positions (aa_i < t and aa_i > t)
    prev_ppl = batch_compute_ppl(probs_list=prev_probs)
    fut_ppl = batch_compute_ppl(probs_list=fut_probs)
    return (
            current_ppl,
            prev_ppl,
            fut_ppl
    )

