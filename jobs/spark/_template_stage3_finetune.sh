#!/bin/bash
#=============================================================================
# Template: Stage 3 finetuning (DGX Spark)
#=============================================================================

set -euo pipefail

projdir=$(cd "$(dirname "$0")/../.." && pwd)
cd ${projdir}

source environment.sh

# Configurations to edit
config_dir=./arglists           # Directory containing config files
config_name=<CONFIG_FILE_NAME>  # Config file name. Note: exclude .sh extension
epochs=5                        # Number of epochs to finetune
resume_from_checkpoint=None     # None to start finetuning fresh
pretrained_weights=<WEIGHTS_PATH>  # Path to pretrained weights (.bin)
finetune_last_n_blocks=1        # Number of last transformer blocks to unfreeze
finetune_last_n_layers=1        # Number of last layers per block to unfreeze

# Constant configurations
num_nodes=1                     # single node
num_devices=1                   # single GPU on Spark
wandb=True                      # enable Weights&Biases logging
wandb_api_key=$WANDB_API_KEY    # define W&B key prior to run, e.g. via .bashrc
device=cuda                     # device available (cuda)

# Construct the version name
datetime=$(date +%Y%m%d_%H%M%S)
version_name=${config_name/config_/}_n${num_nodes}_d${num_devices}_e${epochs}_V${datetime}

# Direct output to log file
mkdir -p logs/run_logs/finetuning
log_fpath=logs/run_logs/finetuning/${version_name}.o

./scripts/finetuning/finetune_singlenode.sh \
    ${config_dir} \
    ${config_name} \
    ${num_nodes} \
    ${num_devices} \
    ${device} \
    ${wandb_api_key} \
    ${version_name} \
    ${epochs} \
    ${resume_from_checkpoint} \
    ${pretrained_weights} \
    ${finetune_last_n_blocks} \
    ${finetune_last_n_layers} \
> ${log_fpath} 2>&1
