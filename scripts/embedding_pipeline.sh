#!/bin/bash
#=============================================================================
#
# FILE: embedding_pipeline.sh
#
# USAGE: embedding_pipeline.sh infpath outdir pencl_weights facilitator_weights config1 config2 prefix
#
# DESCRIPTION: Runs a data processing pipeline to embed sequence-text pairs and
#   save these in hdf5 format for Stage 3 ProteoScribe pretraining/finetuning.
#
# EXAMPLE: sh embedding_pipeline.sh \
#   data/dataset_with_prompts.csv \
#   outputs/embeddings \
#   weights/PenCL/PenCL_V09152023_last.ckpt \
#   weights/Facilitator/Facilitator_MMD15.ckpt/last.ckpt \
#   configs/stage1_config_PenCL_inference.json \
#   configs/stage2_config_Facilitator_sample.json \
#   <dataset_name>
#=============================================================================

set -euo pipefail

infpath=$1
outdir=$2
PENCL_WEIGHTS=$3
FACILITATOR_WEIGHTS=$4
config1=$5  # configs/stage1_config_PenCL_inference.json
config2=$6  # configs/stage2_config_Facilitator_sample.json
prefix=$7

dataset_key=MMD_data

# Create directory for outputs
mkdir -p ${outdir}

# Run PenCL for Stage 1 embeddings
biom3_PenCL_inference \
    -i ${infpath} \
    -c ${config1} \
    -m ${PENCL_WEIGHTS} \
    -o ${outdir}/${prefix}.PenCL_emb.pt \
    --batch_size 256

# Run Facilitator for Stage 2 embeddings
biom3_Facilitator_sample \
    -i ${outdir}/${prefix}.PenCL_emb.pt \
    -c ${config2} \
    -m ${FACILITATOR_WEIGHTS} \
    -o ${outdir}/${prefix}.Facilitator_emb.pt \
    --device cpu \
    --mmd_sample_limit 1000

# Compile Stage 1 and 2 data into an hdf5 dataset ready for finetuning
python scripts/data_prep/compile_stage2_data_to_hdf5.py \
    -o ${outdir}/${prefix}.compiled_emb.hdf5 \
    --dataset_key ${dataset_key} \
    --facilitator_embeddings ${outdir}/${prefix}.Facilitator_emb.pt
