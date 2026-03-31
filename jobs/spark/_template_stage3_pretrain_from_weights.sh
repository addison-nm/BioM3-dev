#!/bin/bash
#=============================================================================
# Template: Stage 3 pretraining from pretrained weights (DGX Spark)
#
# Use this when starting a new training phase (e.g. Phase 2 / Pfam) from
# a raw state_dict file rather than a Lightning checkpoint.
#=============================================================================

set -euo pipefail

projdir=$(cd "$(dirname "$0")/../.." && pwd)
cd ${projdir}

source environment.sh

# Configurations to edit
config_dir=./arglists           # Directory containing config files
config_name=<CONFIG_FILE_NAME>  # Config file name. Note: exclude .sh extension
epochs=5                        # Ignored for Phase 2 training (uses max_steps)
resume_from_checkpoint=None     # None to start fresh with pretrained weights

# Path to pretrained state_dict file.
# Typically logs/history/Stage3_history/checkpoints/<NAME>/state_dict.best.pth
pretrained_weights=<STATE_DICT_PATH>

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
mkdir -p logs/run_logs/pretraining
log_fpath=logs/run_logs/pretraining/${version_name}.o

./scripts/pretraining/pretrain_singlenode.sh \
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
> ${log_fpath} 2>&1
