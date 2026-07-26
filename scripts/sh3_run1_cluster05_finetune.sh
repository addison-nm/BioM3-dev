#!/usr/bin/env bash
#
# Reproducible SH3 finetune: run1_base weights + 50%-identity cluster-aware split.
#
#   1. Embed    SH3 CSV --(run1_base PenCL + Facilitator)--> compiled HDF5
#   2. Split    cluster sequences at 50% identity; pack whole clusters 80/10/10
#               (no cluster spans splits -> no train/val/test leakage)
#   3. Finetune ProteoScribe (last block) from run1_base, holding out the test split
#
# Assumes the `biom3-env` conda environment is active and `mmseqs` is on PATH.
# Override device/threshold inline, e.g.  DEVICE=cuda MIN_SEQ_ID=0.5 bash <script>
#
set -euo pipefail

# --- paths -------------------------------------------------------------------
SH3_CSV=data/datasets/SH3/FINAL_SH3_all_dataset_with_prompts.csv
EMB_DIR=outputs/run1_base/SH3_embeddings
PREFIX=SH3_run1_base
HDF5=${EMB_DIR}/${PREFIX}.compiled_emb.hdf5
MANIFEST=${EMB_DIR}/${PREFIX}.split_manifest.json

WEIGHT_SET=configs/weights/run1_base.json
PROTEOSCRIBE_INIT=weights/ProteoScribe/run1_base_proteoscribe.bin
FINETUNE_CONFIG=configs/stage3_training/finetune_v1.json
RUN_ID=sh3_run1_clust05_v1

DEVICE="${DEVICE:-cuda}"
MIN_SEQ_ID="${MIN_SEQ_ID:-0.5}"

# .ckpt init weights are loaded via torch.load without weights_only.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

# --- 1. Embeddings (Stage 1 PenCL + Stage 2 Facilitator -> HDF5) -------------
if [ ! -f "${HDF5}" ]; then
    biom3_embedding_pipeline \
        -i "${SH3_CSV}" \
        -o "${EMB_DIR}" \
        --prefix "${PREFIX}" \
        --weight_set "${WEIGHT_SET}" \
        --pencl_config configs/inference/stage1_PenCL.json \
        --facilitator_config configs/inference/stage2_Facilitator.json \
        --device "${DEVICE}"
else
    echo "[1/3] Embeddings already present: ${HDF5} (skipping Stage 1+2)"
fi

# --- 2. Cluster-aware train/val/test split -----------------------------------
biom3_cluster_split \
    --primary_data_path "${HDF5}" \
    --facilitator MMD \
    --train_frac 0.7 --val_frac 0.2 --test_frac 0.1 \
    --min_seq_id "${MIN_SEQ_ID}" --coverage 0.8 \
    --diffusion_steps 1024 \
    -o "${MANIFEST}"

# --- 3. Finetune ProteoScribe on the curated split ---------------------------
biom3_train_stage3 \
    --config_path "${FINETUNE_CONFIG}" \
    --pretrained_weights "${PROTEOSCRIBE_INIT}" \
    --primary_data_path "${HDF5}" \
    --split_manifest_path "${MANIFEST}" \
    --devices_per_node 1 \
    --num_nodes 1 \
    --distributed_strategy ddp \
    --limit_val_batches 1.0 \
    --early_stopping_metric val_loss \
    --run_id "${RUN_ID}" \
    --device "${DEVICE}"

echo "Done. Checkpoints/artifacts under outputs/Stage3/finetuning/{checkpoints,runs}/${RUN_ID}/"
