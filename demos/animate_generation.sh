#!/usr/bin/env bash
# animate_generation.sh — Demonstrate sequence generation animation
#
# Shows how to use --animate_prompts / --animate_replicas / --animation_dir
# to produce per-step GIF animations of the ProteoScribe diffusion process.
# Each frame in the output GIF corresponds to one denoising step, with masked
# positions shown as '-' and gradually replaced by amino acid characters.
#
# Prerequisites:
#   pip install -e .
#   source environment.sh
#   # Stage 1 → Stage 2 outputs already produced (or use an existing .pt file)
#   # ProteoScribe weights available at WEIGHTS_PATH below
#
# Usage:
#   bash demos/animate_generation.sh
#
# If you don't have SH3 Facilitator embeddings, substitute any .pt file
# containing a 'z_c' key for FACILITATOR_EMB below.

set -euo pipefail

FACILITATOR_EMB="outputs/finetuning/SH3/SH3.Facilitator_emb.pt"
WEIGHTS="weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.renamed.bin"
CONFIG="configs/stage3_config_ProteoScribe_sample.json"
OUTDIR="demos/outputs/animate_generation"

echo "============================================================"
echo "  Demo: Sequence Generation Animation"
echo "============================================================"
echo ""

# --- Example 1: Animate a single prompt (default behaviour) ---
# Omitting --animate_replicas uses the default of 1, which selects replica 0.
# The GIF is written to <output_dir>/animations/prompt_0_replica_0.gif.
echo "[1] Animate prompt 0, replica 0 (default replica count)..."
biom3_ProteoScribe_sample \
    --input_path  "$FACILITATOR_EMB" \
    --json_path   "$CONFIG" \
    --model_path  "$WEIGHTS" \
    --output_path "${OUTDIR}/ex1_sequences.pt" \
    --animate_prompts 0

echo ""

# --- Example 2: Animate a specific subset of prompts ---
# Passes indices 0, 1, and 2; animates replica 0 for each.
# GIFs: prompt_0_replica_0.gif, prompt_1_replica_0.gif, prompt_2_replica_0.gif
echo "[2] Animate prompts 0, 1, 2 — replica 0..."
biom3_ProteoScribe_sample \
    --input_path  "$FACILITATOR_EMB" \
    --json_path   "$CONFIG" \
    --model_path  "$WEIGHTS" \
    --output_path "${OUTDIR}/ex2_sequences.pt" \
    --animate_prompts 0 1 2

echo ""

# --- Example 3: Animate all prompts, first 3 replicas ---
# --animate_replicas N selects range(0, N), so 3 → replicas 0, 1, 2.
# --animate_prompts all animates every prompt in the embedding file.
echo "[3] Animate all prompts, replicas 0–2..."
biom3_ProteoScribe_sample \
    --input_path  "$FACILITATOR_EMB" \
    --json_path   "$CONFIG" \
    --model_path  "$WEIGHTS" \
    --output_path "${OUTDIR}/ex3_sequences.pt" \
    --animate_prompts all \
    --animate_replicas 3

echo ""

# --- Example 4: Custom animation output directory ---
# Animations are written to the path given by --animation_dir instead of
# the default <output_dir>/animations/.
echo "[4] Animate prompt 0, all replicas — custom animation dir..."
biom3_ProteoScribe_sample \
    --input_path   "$FACILITATOR_EMB" \
    --json_path    "$CONFIG" \
    --model_path   "$WEIGHTS" \
    --output_path  "${OUTDIR}/ex4_sequences.pt" \
    --animate_prompts 0 \
    --animate_replicas all \
    --animation_dir "${OUTDIR}/ex4_animations"

echo ""
echo "============================================================"
echo "  Done. Outputs in ${OUTDIR}/"
echo ""
echo "  Each example produces:"
echo "    *_sequences.pt              — generated sequences (all prompts × replicas)"
echo "    animations/                 — GIFs named prompt_<P>_replica_<R>.gif"
echo "                                  (examples 1–3 use default animations/ subdir)"
echo "    ex4_animations/             — custom dir used by example 4"
echo "    *.log, *manifest.json       — logging and reproducibility metadata"
echo "============================================================"
