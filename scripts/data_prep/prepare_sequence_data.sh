#!/bin/bash
#=============================================================================
#
# FILE: prepare_sequence_data.sh
#
# USAGE: prepare_sequence_data.sh infpath outdir
#
# DESCRIPTION: Runs a data processing pipeline to embed sequence-text pairs and
#   save these in hdf5 format for Stage 3 ProteoScribe training.
#
# EXAMPLE: sh prepare_sequence_data.sh <infpath> <outdir>
#=============================================================================

infpath=$1
outdir=$2
config1=$3  # configs/stage1_config_PenCL_inference.json
config2=$4  # configs/stage1_config_PenCL_inference.json
prefix=$5

# Check args

PENCL_WEIGHTS=weights/PenCL/BioM3_PenCL_epoch20.bin
FACILITATOR_WEIGHTS=weights/Facilitator/BioM3_Facilitator_epoch20.bin

# Create directory for outputs
mkdir -p ${outdir}

# Run PenCL for Stage 1 embeddings
biom3_run_PenCL_inference \
    -i ${infpath} \
    -c ${config1} \
    -m ${PENCL_WEIGHTS} \
    -o ${outdir}/${prefix}.PenCL_emb.pt

# Run Facilitator for Stage 2 embeddings
biom3_run_Facilitator_sample \
    -i ${outdir}/${prefix}.PenCL_emb.pt \
    -c ${config2} \
    -m ${FACILITATOR_WEIGHTS} \
    -o ${outdir}/${prefix}.Facilitator_emb.pt
