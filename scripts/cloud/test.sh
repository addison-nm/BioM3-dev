#!/usr/bin/env bash
#=============================================================================
#
# FILE: scripts/cloud/test.sh
#
# Cloud job: run the BioM3 test suite inside the container. The test-suite
# counterpart of finetune.sh / generate.sh, so "run the tests" is a first-class
# job in the same {job} x {provider} x {data} matrix (see docs/setup/cloud_jobs.md).
#
# This is the generalized, env-driven form of mithril/run_tests.task.yaml's
# inline `pytest` command. The container ENTRYPOINT syncs weights from
# BIOM3_WEIGHTS_URI (when set) before pytest runs, so weight-gated tests
# execute instead of skipping.
#
# ENV (all optional):
#   PYTEST_ARGS   args passed to pytest (default: "tests/ --use_gpu").
#                 e.g. "tests/ --quick"  or  "tests/stage3_tests -x".
#
#=============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/cloud/lib.sh
source "${SCRIPT_DIR}/lib.sh"

PYTEST_ARGS="${PYTEST_ARGS:-tests/ --use_gpu}"

log "pytest ${PYTEST_ARGS}"
# Word-split PYTEST_ARGS intentionally (it is a flag string, not a path).
# shellcheck disable=SC2086
exec pytest ${PYTEST_ARGS}
