#!/usr/bin/env bash
#=============================================================================
#
# FILE: pretrain_singlenode.sh
#
# USAGE: pretrain_singlenode.sh config_dir config_name \
#       NRANKS NGPU_PER_RANK device wandb_api_key run_id epochs \
#       resume_from_checkpoint [pretrained_weights]
#
# DESCRIPTION: Wrapper for running the stage3_pretraining.sh script.
#
#=============================================================================

if [ "$#" -eq 9 ]; then
  pretrained_weights=UNSPECIFIED
elif [ "$#" -eq 10 ]; then
  pretrained_weights=${10}
else
  echo "Usage: $0 config_dir config_name NRANKS NGPU_PER_RANK device wandb_api_key run_id epochs resume_from_checkpoint [pretrained_weights]"
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

if [ "$NRANKS" -ne 1 ]; then
    echo "NRANKS should be specified as 1 for single node run."
    exit 1
fi

# Compute total number of devices across all nodes
NGPUS="$((${NRANKS}*${NGPU_PER_RANK}))"

echo "NRANKS: ${NRANKS}, NGPU_PER_RANK: ${NGPU_PER_RANK}, NGPUS: ${NGPUS}"

./scripts/stage3_pretraining.sh \
    $wandb_api_key $config_dir $config_name \
    $run_id \
    $NRANKS \
    $NGPU_PER_RANK \
    $device \
    $epochs \
    $resume_from_checkpoint \
    $pretrained_weights
