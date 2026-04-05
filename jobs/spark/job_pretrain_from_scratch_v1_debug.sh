#!/bin/bash
#=============================================================================
# Job: Stage 3 pretrain from scratch — debug (DGX Spark)
# Equivalent to Aurora n2/n8 debug jobs (3 epochs, single node)
#=============================================================================

set -euo pipefail

projdir=$(cd "$(dirname "$0")/../.." && pwd)
cd ${projdir}

# Configurations to edit
config_path=./configs/stage3_training/pretrain_scratch_v1.json  # JSON config file
epochs=3                        # Number of epochs to train
resume_from_checkpoint=None     # None to train from scratch

# Constant configurations
num_nodes=1                     # single node
num_devices=1                   # single GPU on Spark
wandb_api_key=${WANDB_API_KEY:-}    # define W&B key prior to run, e.g. via .bashrc
device=cuda                     # device available (cuda)

# Construct the run ID
datetime=$(date +%Y%m%d_%H%M%S)
config_name=$(basename "${config_path}" .json)
run_id=${config_name}_n${num_nodes}_d${num_devices}_e${epochs}_V${datetime}

# Direct output to log file
mkdir -p logs
log_fpath=logs/${run_id}.o

source environment.sh
./scripts/stage3_train_singlenode.sh \
    ${config_path} \
    ${num_devices} \
    ${device} \
    "${wandb_api_key}" \
    ${run_id} \
    --epochs ${epochs} \
    --resume_from_checkpoint ${resume_from_checkpoint} \
    --wandb True \
> ${log_fpath} 2>&1
