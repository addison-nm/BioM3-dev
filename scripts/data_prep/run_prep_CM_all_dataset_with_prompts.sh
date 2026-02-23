#!/usr/bin/env bash

infpath="data/FINAL_CM_all_dataset_with_prompts.csv"
outdir="outputs/finetuning/CM"
config1="configs/stage1_config_PenCL_inference.json"
config2="configs/stage2_config_Facilitator_sample.json"
prefix="CM"

./scripts/data_prep/prepare_sequence_data.sh \
    ${infpath} \
    ${outdir} \
    ${config1} \
    ${config2} \
    ${prefix}
