#!/usr/bin/env bash
#=============================================================================
#
# FILE: seqgen_wrapper.sh
#
# USAGE: seqgen_wrapper.sh infpath weights_fpath outdir config prefix
#
# DESCRIPTION: Wraps the ProteoScribe entrypoint command and executes it on the
#   given input embeddings using the given model weights and config file.
#   Weights can be a raw state dict (.bin, .pth, .pt), a Lightning checkpoint
#   (.ckpt), or a sharded DeepSpeed checkpoint directory.
#
# EXAMPLE: sh seqgen_wrapper.sh \
#   data/test_Facilitator_embeddings.pt \
#   weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
#   outputs/sequence_generation \
#   configs/stage3_config_ProteoScribe_sample.json \
#   <dataset_name>
#=============================================================================


set -euo pipefail

infpath=$1  # input embeddings
weights_fpath=$2  # model weights
outdir=$3
config=$4  # configs/stage3_config_ProteoScribe_sample.json
prefix=$5

# TODO: Check args
# TODO: Consider randomness implications and seeding

# Create directory for outputs
mkdir -p ${outdir}

# Run ProteoScribe
biom3_ProteoScribe_sample \
    -i ${infpath} \
    -c ${config} \
    -m ${weights_fpath} \
    -o ${outdir}/${prefix}.ProteoScribe_output.pt
