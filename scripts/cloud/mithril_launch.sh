#!/usr/bin/env bash
# Launch a BioM3 cloud/*.yaml on Mithril via the bundled sky CLI (`mithril sky
# launch`), with a UNIQUE cluster name and AWS credentials passed as redacted secrets.
#
# The image is PUBLIC on GHCR (ghcr.io/natural-machine/biom3), so there is NO registry
# login or token — Mithril pulls it anonymously. AWS credentials are still required
# for the container's S3 weight-sync. Real infra identifiers (e.g. the weights bucket)
# live in the gitignored configs/jobs/local.env, auto-loaded below via --env-file when
# present (copy configs/jobs/local.env.example to create it).
#
# Why `mithril sky launch` (not the `mithril launch` wrapper): only the bundled sky CLI
# supports --secret (redacted) and fills `secrets: KEY: null`.
# Why a unique cluster name: Mithril retains bid names, so reuse fails with a
# misleading "ResourcesUnavailableError".
#
# Usage: scripts/cloud/mithril_launch.sh <task.yaml> [cluster-prefix] [extra sky args...]
#   scripts/cloud/mithril_launch.sh cloud/test.mithril.yaml    biom3-test
#   scripts/cloud/mithril_launch.sh cloud/run.mithril.yaml biom3-run --env CMD="nvidia-smi"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TASK="${1:?usage: mithril_launch.sh <task.yaml> [cluster-prefix] [extra sky args...]}"
PREFIX="${2:-biom3}"
shift || true
shift 2>/dev/null || true

# Clear any stale static creds inherited from the caller's shell: env creds win over
# the SSO profile in AWS's chain, so an old exported trio shadows a fresh `aws sso login`.
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "[launch] ERROR: AWS credentials are invalid/expired. Run 'aws sso login'" \
         "(or refresh your profile) and retry." >&2
    exit 1
fi
eval "$(aws configure export-credentials --format env)"        # harvest fresh from the profile

# Auto-load gitignored local infra defaults (real weights bucket, etc.) when present.
LOCAL_ENV="${REPO_ROOT}/configs/jobs/local.env"
ENVFILE=()
[ -f "${LOCAL_ENV}" ] && ENVFILE=(--env-file "${LOCAL_ENV}")

CLUSTER="${PREFIX}-$(date +%y%m%d-%H%M%S)"
echo "[launch] cluster=${CLUSTER}  task=${TASK}"

exec mithril sky launch "${TASK}" -c "${CLUSTER}" -y --down \
  "${ENVFILE[@]}" \
  --secret AWS_ACCESS_KEY_ID \
  --secret AWS_SECRET_ACCESS_KEY \
  --secret AWS_SESSION_TOKEN \
  "$@"
