#!/usr/bin/env bash
#=============================================================================
#
# FILE: finetune_multinode.sh
#
# USAGE: finetune_multinode.sh config_dir config_name \
#       NRANKS NGPU_PER_RANK wandb_api_key version_name epochs \
#       resume_from_checkpoint pretrained_checkpoint finetune_last_n_blocks
#
# DESCRIPTION: Wrapper for running the stage3_finetuning.sh script in a 
#   distributed setting.
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

# Compute total number of devices across all nodes
NGPUS="$((${NRANKS}*${NGPU_PER_RANK}))"

echo "NRANKS: ${NRANKS}, NGPU_PER_RANK: ${NGPU_PER_RANK}, NGPUS: ${NGPUS}"

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
        $epochs \
        $resume_from_checkpoint \
        $pretrained_checkpoint \
        $finetune_last_n_block
