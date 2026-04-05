#!/usr/bin/env bash


for i in {1..5}; do
    infpath="outputs/finetuning/SH3_prompts/prompt_${i}/SH3_prompt_${i}.Facilitator_emb.pt"
    weights_fpath=weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.renamed.bin
    outdir="outputs/sequence_generation/SH3_prompts"
    config="configs/inference/stage3_ProteoScribe_sample.json"
    prefix="SH3_prompt_${i}"

    ./scripts/seqgen_wrapper.sh \
        ${infpath} \
        ${weights_fpath} \
        ${outdir} \
        ${config} \
        ${prefix}
done
