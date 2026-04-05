#!/usr/bin/env bash

infpath="data/FINAL_SH3_all_dataset_with_prompts.csv"
outdir="demos/outputs/SH3"
pencl_weights="weights/PenCL/PenCL_V09152023_last.ckpt"
facilitator_weights="weights/Facilitator/Facilitator_MMD15.ckpt/last.ckpt"
config1="configs/inference/stage1_PenCL.json"
config2="configs/inference/stage2_Facilitator.json"
prefix="SH3"

./scripts/embedding_pipeline.sh \
    ${infpath} \
    ${outdir} \
    ${pencl_weights} \
    ${facilitator_weights} \
    ${config1} \
    ${config2} \
    ${prefix}
