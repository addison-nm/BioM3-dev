#!/usr/bin/env bash

set -euo pipefail

WEIGHTS="weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.renamed.bin"
CONFIG="configs/inference/stage3_ProteoScribe_sample.json"
OUTDIR="outputs/sequence_generation/SH3_prompts"

mkdir -p "${OUTDIR}"

for i in {1..5}; do
    biom3_ProteoScribe_sample \
        --input_path  "outputs/finetuning/SH3_prompts/prompt_${i}/SH3_prompt_${i}.Facilitator_emb.pt" \
        --config_path "$CONFIG" \
        --model_path  "$WEIGHTS" \
        --output_path "${OUTDIR}/SH3_prompt_${i}.ProteoScribe_output.pt"
done
