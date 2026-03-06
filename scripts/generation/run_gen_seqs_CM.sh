#!/usr/bin/env bash

infpath="outputs/test_Facilitator_embeddings.pt"
weights_fpath=weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin
outdir="outputs/sequence_generation/CM"
config="configs/stage3_config_ProteoScribe_sample.json"
prefix="CM"

./scripts/seqgen_wrapper.sh \
    ${infpath} \
    ${weights_fpath} \
    ${outdir} \
    ${config} \
    ${prefix}
