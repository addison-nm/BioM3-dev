#!/usr/bin/env bash
#=============================================================================
#
# FILE: grpo_train_singlenode.sh
#
# USAGE: grpo_train_singlenode.sh CONFIG_PATH RUN_ID DEVICE \
#        [additional --key value overrides]
#
# DESCRIPTION: Single-GPU wrapper for biom3_grpo_train. GRPO does not use
#   mpiexec or DeepSpeed (single-GPU by design — see Phase 4 in
#   docs/.claude_prompts/PROMPT_grpo_integration.md), so this is a thin
#   shim around the entry point that sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
#   and forwards extra args to the trainer.
#
#   On Aurora, pin to a specific tile by exporting ZE_AFFINITY_MASK before
#   invocation (e.g. ZE_AFFINITY_MASK=0 to use tile 0 of GPU 0).
#
#   Requires: source environment.sh first.
#
#=============================================================================
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 CONFIG_PATH RUN_ID DEVICE [--key value ...]"
    echo "  CONFIG_PATH  e.g. configs/grpo/example_grpo.json"
    echo "  RUN_ID       unique identifier; outputs land at OUTPUT_ROOT/RUN_ID"
    echo "  DEVICE       cuda | xpu | cpu (forwarded as --device)"
    exit 1
fi

config_path=$1
run_id=$2
device=$3
shift 3

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

exec biom3_grpo_train \
    --config_path "${config_path}" \
    --run_id "${run_id}" \
    --device "${device}" \
    "$@"
