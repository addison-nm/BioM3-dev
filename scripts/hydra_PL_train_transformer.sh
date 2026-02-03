#!/usr/bin/env sh

version=$1
epochs=$2
resume=$3

python scripts/PL_train_stage3.py \
	version_name=$version \
	epochs=$epochs \
	resume_from_checkpoint=$resume
