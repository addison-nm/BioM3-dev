#!/usr/bin/env bash
#=============================================================================
#
# FILE: docker/push.sh
#
# Register (tag + push) an existing local BioM3 image to AWS ECR, so cloud
# instances pull it instead of rebuilding (~10-40 min build -> seconds). The
# SkyPilot tasks under cloud/ reference the ECR ref via
# `resources.image_id: docker:<ECR_REF>`.
#
# This is the build-elsewhere-then-push path. To build AND push in one step on
# a linux/amd64 builder, use build.sh directly:
#   docker/build.sh --awscli --platform linux/amd64 \
#       --tag 955510722784.dkr.ecr.us-east-2.amazonaws.com/biom3:gpu --push
#
# USAGE:
#   docker/push.sh [--local-tag T] [--repo R] [--region X] [--create-repo]
#
#   --local-tag T   local image to push (default: biom3:gpu)
#   --repo R        ECR repo ref WITHOUT the tag
#                   (default: 955510722784.dkr.ecr.us-east-2.amazonaws.com/biom3)
#   --region X      AWS region (default: us-east-2)
#   --create-repo   create the ECR repository first (one-time, idempotent)
#
# Requires: docker, awscli, and valid AWS creds in the environment
# (e.g. `eval "$(aws configure export-credentials --format env)"`).
#
#=============================================================================
set -euo pipefail

LOCAL_TAG="biom3:gpu"
REPO="955510722784.dkr.ecr.us-east-2.amazonaws.com/biom3"
REGION="us-east-2"
CREATE_REPO=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local-tag)   LOCAL_TAG="$2"; shift 2 ;;
        --repo)        REPO="$2"; shift 2 ;;
        --region)      REGION="$2"; shift 2 ;;
        --create-repo) CREATE_REPO=1; shift ;;
        -h|--help)     sed -n '3,33p' "$0"; exit 0 ;;
        *)             echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# The image tag (e.g. ":gpu") rides along from the local tag.
TAG="${LOCAL_TAG##*:}"
REMOTE="${REPO}:${TAG}"
REGISTRY="${REPO%%/*}"          # <acct>.dkr.ecr.<region>.amazonaws.com
REPO_NAME="${REPO#*/}"          # everything after the registry host

command -v aws >/dev/null 2>&1 || { echo "ERROR: awscli not found." >&2; exit 1; }

if [[ "${CREATE_REPO}" -eq 1 ]]; then
    echo "+ aws ecr create-repository --repository-name ${REPO_NAME}" >&2
    aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}" \
        >/dev/null 2>&1 || echo "  (repository already exists; continuing)" >&2
fi

echo "+ docker login ${REGISTRY}" >&2
aws ecr get-login-password --region "${REGION}" \
    | docker login --username AWS --password-stdin "${REGISTRY}"

echo "+ docker tag ${LOCAL_TAG} ${REMOTE}" >&2
docker tag "${LOCAL_TAG}" "${REMOTE}"

echo "+ docker push ${REMOTE}" >&2
docker push "${REMOTE}"

echo "Pushed ${REMOTE}" >&2
echo "Use in cloud/*.yaml:  image_id: docker:${REMOTE}" >&2
