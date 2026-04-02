#!/usr/bin/env bash

infpath="outputs/finetuning/SH3/SH3.Facilitator_emb.pt"
weights_fpath=weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.renamed.bin
outdir="outputs/sequence_generation/SH3"
config="configs/stage3_config_ProteoScribe_sample.json"
prefix="SH3"

./scripts/seqgen_wrapper.sh \
    ${infpath} \
    ${weights_fpath} \
    ${outdir} \
    ${config} \
    ${prefix}
