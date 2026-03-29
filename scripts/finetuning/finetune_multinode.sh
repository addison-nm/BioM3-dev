#!/usr/bin/env bash
#=============================================================================
#
# FILE: finetune_multinode.sh
#
# USAGE: finetune_multinode.sh config_dir config_name \
#       NRANKS NGPU_PER_RANK device wandb_api_key version_name epochs \
#       resume_from_checkpoint \
#       pretrained_weights finetune_last_n_blocks finetune_last_n_layers
#
# DESCRIPTION: Wrapper for running the stage3_finetuning.sh script in a
#   distributed setting.
#
#=============================================================================

if [ "$#" -ne 12 ]; then
    echo "Usage: $0 config_dir config_name NRANKS NGPU_PER_RANK device wandb_api_key version_name epochs resume_from_checkpoint pretrained_weights finetune_last_n_blocks finetune_last_n_layers"
    exit 1
fi

config_dir=$1
config_name=$2
NRANKS=$3
NGPU_PER_RANK=$4
device=$5
wandb_api_key=$6
version_name=$7
epochs=$8
resume_from_checkpoint=$9
pretrained_weights=${10}
finetune_last_n_blocks=${11}
finetune_last_n_layers=${12}

# Compute total number of devices across all nodes
NGPUS="$((${NRANKS}*${NGPU_PER_RANK}))"

echo "NRANKS: ${NRANKS}, NGPU_PER_RANK: ${NGPU_PER_RANK}, NGPUS: ${NGPUS} ($device)"

mpiexec \
    --verbose \
    --envall \
    -n "${NGPUS}" \
    --ppn "${NGPU_PER_RANK}" \
    --hostfile="${PBS_NODEFILE}" \
    ./scripts/stage3_finetuning.sh \
        $wandb_api_key $config_dir $config_name \
        $version_name \
        $NRANKS \
        $NGPU_PER_RANK \
        $device \
        $epochs \
        $resume_from_checkpoint \
        $pretrained_weights \
        $finetune_last_n_blocks \
        $finetune_last_n_layers
