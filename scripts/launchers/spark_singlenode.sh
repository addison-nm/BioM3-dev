#!/usr/bin/env bash
#=============================================================================
#
# FILE: spark_singlenode.sh
#
# USAGE: spark_singlenode.sh ENTRYPOINT [args...]
#
# DESCRIPTION: DGX Spark single-node launcher. The Spark has a single GPU
#   so distributed launchers and CPU binding are unnecessary. We exec the
#   entry point directly.
#
#=============================================================================
set -euo pipefail
exec "$@"
