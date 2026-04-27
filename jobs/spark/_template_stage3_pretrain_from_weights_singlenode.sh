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

# Configurations to edit
config_path=./configs/stage3_training/<CONFIG_NAME>.json  # JSON config file
epochs=5                        # Ignored for Phase 2 training (uses max_steps)
use_wandb=True                  # set to False to disable wandb (requires WANDB_API_KEY exported when True)
resume_from_checkpoint=None     # None to start fresh with pretrained weights

# Path to pretrained state_dict file.
# Typically logs/history/Stage3_history/checkpoints/<NAME>/state_dict.best.pth
pretrained_weights=<STATE_DICT_PATH>

# Constant configurations
num_nodes=1                     # single node
num_devices=1                   # single GPU on Spark
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
    ${run_id} \
    --epochs ${epochs} \
    --resume_from_checkpoint ${resume_from_checkpoint} \
    --pretrained_weights ${pretrained_weights} \
    --wandb ${use_wandb} \
> ${log_fpath} 2>&1
