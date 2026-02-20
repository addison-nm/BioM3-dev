#!/usr/bin/env bash
#=============================================================================
#
# FILE: finetune_singlenode.sh
#
# USAGE: finetune_singlenode.sh config_dir config_name \
#       NRANKS NGPU_PER_RANK wandb_api_key version_name epochs \
#       resume_from_checkpoint pretrained_checkpoint finetune_last_n_blocks
#
# DESCRIPTION: Wrapper for running the stage3_finetuning.sh script.
#
#=============================================================================

if [ "$#" -ne 10 ]; then
    echo "Usage: $0 config_dir config_name NRANKS NGPU_PER_RANK wandb_api_key version_name epochs resume_from_checkpoint pretrained_checkpoint finetune_last_n_blocks"
    exit 1
fi

config_dir=$1
config_name=$2
NRANKS=$3
NGPU_PER_RANK=$4
wandb_api_key=$5
version_name=$6
epochs=$7
resume_from_checkpoint=$8
pretrained_checkpoint=$9
finetune_last_n_blocks=$10

if [ "$NRANKS" -ne 1 ]; then
    echo "NRANKS should be specified as 1 for single node run."
    exit 1
fi

# Compute total number of devices across all nodes
NGPUS="$((${NRANKS}*${NGPU_PER_RANK}))"

echo "NRANKS: ${NRANKS}, NGPU_PER_RANK: ${NGPU_PER_RANK}, NGPUS: ${NGPUS}"


./scripts/stage3_finetuning.sh \
    $wandb_api_key $config_dir $config_name \
    $version_name \
    $NRANKS \
    $NGPU_PER_RANK \
    $epochs \
    $resume_from_checkpoint \
    $pretrained_checkpoint \
    $finetune_last_n_block
