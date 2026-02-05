#!/usr/bin/env sh

WANDB_API_KEY=$1
config_dir=$2
config_name=$3

version=$4
epochs=$5
resume=$6

export WANDB_API_KEY=$WANDB_API_KEY
wandb login

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
python scripts/PL_train_stage3.py -cd $config_dir --config-name $config_name \
	version_name=$version \
	epochs=$epochs \
	resume_from_checkpoint=$resume
