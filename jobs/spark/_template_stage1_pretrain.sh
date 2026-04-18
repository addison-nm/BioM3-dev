#!/bin/bash
#=============================================================================
# Template: Stage 1 PenCL pretraining (DGX Spark)
#=============================================================================

set -euo pipefail

projdir=$(cd "$(dirname "$0")/../.." && pwd)
cd ${projdir}

# Configurations to edit
config_path=./configs/stage1_training/pretrain_scratch_v1.json   # or pretrain_pfam_v1.json
epochs=5
resume_from_checkpoint=None

# Constant configurations
num_nodes=1
num_devices=1
wandb_api_key=${WANDB_API_KEY:-}
device=cuda

# Construct the run ID
datetime=$(date +%Y%m%d_%H%M%S)
config_name=$(basename "${config_path}" .json)
run_id=${config_name}_n${num_nodes}_d${num_devices}_e${epochs}_V${datetime}

mkdir -p logs
log_fpath=logs/${run_id}.o

source environment.sh
./scripts/stage1_train_singlenode.sh \
    ${config_path} \
    ${num_devices} \
    ${device} \
    "${wandb_api_key}" \
    ${run_id} \
    --epochs ${epochs} \
    --resume_from_checkpoint ${resume_from_checkpoint} \
    --wandb True \
> ${log_fpath} 2>&1
