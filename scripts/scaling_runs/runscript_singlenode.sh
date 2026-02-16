#!/usr/bin/env bash

NRANKS=$1
NGPU_PER_RANK=$2

if [ "$NRANKS" -ne 1 ]; then
    echo "Usage: $0 NRANKS=1 NGPU_PER_RANK"
    exit 1
fi


NGPUS="$((${NRANKS}*${NGPU_PER_RANK}))"
echo "NRANKS: ${NRANKS}, NGPU_PER_RANK: ${NGPU_PER_RANK}, NGPUS: ${NGPUS}"

datetime=$(date +%Y%m%d_%H%M%S)

wandb_api_key="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"
configdir=./arglists
config_name=config_${NRANKS}node_${NGPU_PER_RANK}gpu

version_name="scaletest_V${datetime}_nohydra_${NRANKS}node_${NGPU_PER_RANK}gpu"
epoch=2 # number of epochs
resume_from_checkpoint=None

./scripts/PL_train_transformer.sh \
    $wandb_api_key $configdir $config_name \
    $version_name \
    $epoch \
    $resume_from_checkpoint
