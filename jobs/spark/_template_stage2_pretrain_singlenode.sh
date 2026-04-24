#!/bin/bash
#=============================================================================
# Template: Stage 2 Facilitator pretraining (DGX Spark)
#=============================================================================

set -euo pipefail

projdir=$(cd "$(dirname "$0")/../.." && pwd)
cd ${projdir}

# Configurations to edit
config_path=./configs/stage2_training/pretrain_scratch_v1.json
epochs=20
resume_from_checkpoint=None

# Constant configurations
num_nodes=1
num_devices=1
device=cuda

# Construct the run ID
datetime=$(date +%Y%m%d_%H%M%S)
config_name=$(basename "${config_path}" .json)
run_id=${config_name}_n${num_nodes}_d${num_devices}_e${epochs}_V${datetime}

mkdir -p logs
log_fpath=logs/${run_id}.o

source environment.sh
./scripts/stage2_train_singlenode.sh \
    ${config_path} \
    ${num_devices} \
    ${device} \
    ${run_id} \
    --epochs ${epochs} \
    --resume_from_checkpoint ${resume_from_checkpoint} \
    --wandb True \
> ${log_fpath} 2>&1
