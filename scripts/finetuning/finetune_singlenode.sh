#!/usr/bin/env bash
#=============================================================================
#
# FILE: finetune_singlenode.sh
#
# USAGE: finetune_singlenode.sh config_dir config_name \
#       NRANKS NGPU_PER_RANK device wandb_api_key run_id epochs \
#       resume_from_checkpoint \
#       pretrained_weights finetune_last_n_blocks finetune_last_n_layers
#
# DESCRIPTION: Wrapper for running the stage3_finetuning.sh script.
#
#=============================================================================

if [ "$#" -ne 12 ]; then
    echo "Usage: $0 config_dir config_name NRANKS NGPU_PER_RANK device wandb_api_key run_id epochs resume_from_checkpoint pretrained_weights finetune_last_n_blocks finetune_last_n_layers"
    exit 1
fi

config_dir=$1
config_name=$2
NRANKS=$3
NGPU_PER_RANK=$4
device=$5
wandb_api_key=$6
run_id=$7
epochs=$8
resume_from_checkpoint=$9
pretrained_weights=${10}
finetune_last_n_blocks=${11}
finetune_last_n_layers=${12}

if [ "$NRANKS" -ne 1 ]; then
    echo "NRANKS should be specified as 1 for single node run."
    exit 1
fi

# Compute total number of devices across all nodes
NGPUS="$((${NRANKS}*${NGPU_PER_RANK}))"

echo "NRANKS: ${NRANKS}, NGPU_PER_RANK: ${NGPU_PER_RANK}, NGPUS: ${NGPUS} ($device)"

./scripts/stage3_finetuning.sh \
    $wandb_api_key $config_dir $config_name \
    $run_id \
    $NRANKS \
    $NGPU_PER_RANK \
    $device \
    $epochs \
    $resume_from_checkpoint \
    $pretrained_weights \
    $finetune_last_n_blocks \
    $finetune_last_n_layers
