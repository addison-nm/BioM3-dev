#!/usr/bin/env bash

set -euo pipefail

FACILITATOR_EMB="outputs/finetuning/SH3/SH3.Facilitator_emb.pt"
WEIGHTS="weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.renamed.bin"
CONFIG="configs/inference/stage3_ProteoScribe_sample.json"
OUTDIR="outputs/sequence_generation/SH3"

mkdir -p "${OUTDIR}"

biom3_ProteoScribe_sample \
    --input_path  "$FACILITATOR_EMB" \
    --config_path "$CONFIG" \
    --model_path  "$WEIGHTS" \
    --output_path "${OUTDIR}/SH3.ProteoScribe_output.pt"
