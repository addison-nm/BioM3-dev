#!/usr/bin/env bash
#=============================================================================
#
# FILE: scripts/cloud/lib.sh
#
# Shared helpers for the provider-agnostic cloud job scripts
# (scripts/cloud/{test,finetune,generate}.sh). These jobs are env-var driven
# and baked into the BioM3 image (via `COPY scripts/`), so the same script runs
# unchanged whether launched by a Mithril/AWS SkyPilot task, a bare
# `docker run`, or `docker/run.sh`.
#
# Source this from a job script:  source "$(dirname "$0")/lib.sh"
#
#=============================================================================

# log MESSAGE...   -> timestamped line on stderr (keeps stdout clean for data).
log() { echo "[cloud] $*" >&2; }

# die MESSAGE...   -> log and exit non-zero.
die() { echo "[cloud] ERROR: $*" >&2; exit 1; }

# require_input VAR1 VAR2 ...  -> die unless at least one named var is non-empty.
# Used to assert "an input was provided" when several env vars are alternatives.
require_one() {
    local v
    for v in "$@"; do
        [[ -n "${!v:-}" ]] && return 0
    done
    die "set one of: ${*/#/\$}"
}

# cloud_run_id CONFIG_PATH NGPU EPOCHS [EXTRA_TAG]
#   Mirrors the ALCF HPC run_id convention
#   (jobs/aurora/job_pretrain_from_scratch_v1_n2.pbs):
#     {config_name}[_EXTRA]_n1_d{NGPU}_e{EPOCHS}_V{YYYYMMDD_HHMMSS}
#   Single-node only here (cloud jobs are single-instance), so nodes is fixed
#   at 1. Only used when RUN_ID is unset.
cloud_run_id() {
    local config_path="$1" ngpu="$2" epochs="$3" extra="${4:-}"
    local config_name datetime
    config_name="$(basename "${config_path}" .json)"
    datetime="$(date +%Y%m%d_%H%M%S)"
    [[ -n "${extra}" ]] && extra="_${extra}"
    echo "${config_name}${extra}_n1_d${ngpu}_e${epochs}_V${datetime}"
}

# weight_from_set WEIGHT_SET_JSON KEY   -> print the path stored under KEY, or
# empty if the file/key is absent. KEY is one of pencl_weights /
# facilitator_weights / proteoscribe_weights (see configs/weights/*.json and
# biom3.core.weight_sets.WEIGHT_KEYS). Uses the image's venv python.
weight_from_set() {
    local path="$1" key="$2"
    [[ -z "${path}" || ! -f "${path}" ]] && { echo ""; return 0; }
    python - "${path}" "${key}" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as fh:
        print(json.load(fh).get(sys.argv[2], "") or "")
except Exception:
    print("")
PY
}
