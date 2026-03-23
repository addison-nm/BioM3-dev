#!/usr/bin/env bash
#=============================================================================
#
# FILE: pretrain_multinode.sh
#
# USAGE: pretrain_multinode.sh config_dir config_name \
#       NRANKS NGPU_PER_RANK device wandb_api_key version_name epochs \
#       resume_from_checkpoint [pretrained_weights]
#
# DESCRIPTION: Wrapper for running the stage3_pretraining.sh script in a 
#   distributed setting.
#
#=============================================================================

if [ "$#" -eq 9 ]; then
  pretrained_weights=UNSPECIFIED
elif [ "$#" -eq 10 ]; then
  pretrained_weights=${10}
else
  echo "Usage: $0 config_dir config_name NRANKS NGPU_PER_RANK device wandb_api_key version_name epochs resume_from_checkpoint [pretrained_weights]"
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

# Compute total number of devices across all nodes
NGPUS="$((${NRANKS}*${NGPU_PER_RANK}))"

echo "NRANKS: ${NRANKS}, NGPU_PER_RANK: ${NGPU_PER_RANK}, NGPUS: ${NGPUS} ($device)"

mpiexec \
    --verbose \
    --envall \
    -n "${NGPUS}" \
    --ppn "${NGPU_PER_RANK}" \
    --hostfile="${PBS_NODEFILE}" \
    ./scripts/stage3_pretraining.sh \
        $wandb_api_key $config_dir $config_name \
        $version_name \
        $NRANKS \
        $NGPU_PER_RANK \
        $device \
        $epochs \
        $resume_from_checkpoint \
        $pretrained_weights
