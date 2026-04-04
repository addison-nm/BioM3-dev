#!/bin/bash
#=============================================================================
# Template: Stage 3 pretraining from scratch (DGX Spark)
#=============================================================================

set -euo pipefail

projdir=$(cd "$(dirname "$0")/../.." && pwd)
cd ${projdir}

source environment.sh

# Configurations to edit
config_dir=./arglists           # Directory containing config files
config_name=<CONFIG_FILE_NAME>  # Config file name. Note: exclude .sh extension
epochs=5                        # Number of epochs to train
resume_from_checkpoint=None     # None to train from scratch

# Constant configurations
num_nodes=1                     # single node
num_devices=1                   # single GPU on Spark
wandb=True                      # enable Weights&Biases logging
wandb_api_key=$WANDB_API_KEY    # define W&B key prior to run, e.g. via .bashrc
device=cuda                     # device available (cuda)

# Construct the run ID
datetime=$(date +%Y%m%d_%H%M%S)
run_id=${config_name/config_/}_n${num_nodes}_d${num_devices}_e${epochs}_V${datetime}

# Direct output to log file
mkdir -p logs/run_logs/pretraining
log_fpath=logs/run_logs/pretraining/${run_id}.o

./scripts/pretraining/pretrain_singlenode.sh \
    ${config_dir} \
    ${config_name} \
    ${num_nodes} \
    ${num_devices} \
    ${device} \
    ${wandb_api_key} \
    ${run_id} \
    ${epochs} \
    ${resume_from_checkpoint} \
> ${log_fpath} 2>&1
