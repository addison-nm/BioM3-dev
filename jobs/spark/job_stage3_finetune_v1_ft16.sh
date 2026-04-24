#!/bin/bash
#=============================================================================
# Job: Stage 3 finetune — 1 block / 16 layers (DGX Spark)
#=============================================================================

set -euo pipefail

projdir=$(cd "$(dirname "$0")/../.." && pwd)
cd ${projdir}

# Configurations to edit
config_path=./configs/stage3_training/finetune_v1.json  # JSON config file
epochs=100                      # Number of epochs to finetune
resume_from_checkpoint=None     # None to start finetuning fresh
pretrained_weights="./weights/ProteoScribe/ProteoScribe_epoch200.pth"  # Path to pretrained weights (.bin)
finetune_last_n_blocks=1        # Number of last transformer blocks to unfreeze
finetune_last_n_layers=16       # Number of last layers per block to unfreeze
primary_data_path="outputs/embeddings/FINAL_SH3_all_dataset_with_prompts/FINAL_SH3_all_dataset_with_prompts.compiled_emb.hdf5"

# Constant configurations
num_nodes=1                     # single node
num_devices=1                   # single GPU on Spark
device=cuda                     # device available (cuda)

# Construct the run ID
datetime=$(date +%Y%m%d_%H%M%S)
config_name=$(basename "${config_path}" .json)
run_id=${config_name}_ft${finetune_last_n_blocks}-${finetune_last_n_layers}_n${num_nodes}_d${num_devices}_e${epochs}_V${datetime}

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
    --finetune True \
    --pretrained_weights ${pretrained_weights} \
    --finetune_last_n_blocks ${finetune_last_n_blocks} \
    --finetune_last_n_layers ${finetune_last_n_layers} \
    --primary_data_path ${primary_data_path} \
    --wandb True \
> ${log_fpath} 2>&1
