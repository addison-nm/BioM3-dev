#!/usr/bin/env bash

if [ "$#" -ne 8 ]; then
  echo "Usage: $0 args config_dir config_name NRANKS NGPU_PER_RANK wandb_api_key version_name epochs resume_from_checkpoint"
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

# Compute total number of devices across all nodes
NGPUS="$((${NRANKS}*${NGPU_PER_RANK}))"

echo "NRANKS: ${NRANKS}, NGPU_PER_RANK: ${NGPU_PER_RANK}, NGPUS: ${NGPUS}"

mpiexec \
  --verbose \
  --envall \
  -n "${NGPUS}" \
  --ppn "${NGPU_PER_RANK}" \
  --hostfile="${PBS_NODEFILE}" \
  ./scripts/PL_train_transformer_production.sh \
    $wandb_api_key $config_dir $config_name \
    $version_name \
    $NRANKS \
    $NGPU_PER_RANK \
    $epochs \
    $resume_from_checkpoint
