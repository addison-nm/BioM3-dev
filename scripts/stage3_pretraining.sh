#!/usr/bin/env bash
#=============================================================================
#
# FILE: stage3_pretraining.sh
#
# USAGE: stage3_pretraining.sh \
#		WANDB_API_KEY config_dir config_name \
# 		version_name num_nodes gpu_devices epochs resume_from_checkpoint
#
# DESCRIPTION: Wraps the Stage 3 training python script PL_train_stage3.py.
#	Reads command line arguments from a specified config file. Arguments for 
#	for this script specify additional configurations, including a version name,
#	number of nodes and devices per node, number of training epochs, and whether 
#	to resume training from a previous checkpoint.
#
#=============================================================================

WANDB_API_KEY=$1
config_dir=$2
config_name=$3

# Source args from config file
source ${config_dir}/${config_name}.sh

# Override config file with given args
version_name=$4
num_nodes=$5
gpu_devices=$6
epochs=$7
resume_from_checkpoint=$8

# Weights&Biases API key
export WANDB_API_KEY=$WANDB_API_KEY

# Run training script
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
python ./scripts/PL_train_stage3.py \
	--output_hist_folder ${output_hist_folder} \
	--output_folder ${output_folder} \
	--save_hist_path ${save_hist_path} \
	--model_option ${model_option} \
	--swissprot_data_root ${swissprot_data_root} \
	--pfam_data_root ${pfam_data_root} \
	--diffusion_steps ${diffusion_steps} \
	--seed ${seed} \
	--batch-size ${batch_size} \
	--warmup-steps ${warmup_steps} \
	--image-size ${image_size} \
	--lr ${lr} \
	--weight-decay ${weight_decay} \
	--ema_inv_gamma ${ema_inv_gamma} \
	--ema_max_value ${ema_max_value} \
	--precision ${precision} \
	--device ${device} \
	--model_option ${model_option} \
	--transformer_dim ${transformer_dim} \
	--transformer_heads ${transformer_heads} \
	--num_classes ${num_classes} \
	--task ${task} \
	--num_y_class_labels ${num_y_class_labels} \
	--enter_eval ${enter_eval} \
	--transformer_depth ${transformer_depth} \
	--choose_optim ${choose_optim} \
	--epochs ${epochs} \
	--acc_grad_batches ${acc_grad_batches} \
	--gpu_devices ${gpu_devices} \
	--num_nodes ${num_nodes} \
	--version_name ${version_name} \
	--scheduler_gamma ${scheduler_gamma} \
	--text_emb_dim ${text_emb_dim} \
	--sequence_keyname ${sequence_keyname} \
	--facilitator ${facilitator} \
	--tb_logger_path ${tb_logger_path} \
	--tb_logger_folder ${tb_logger_folder} \
	--resume_from_checkpoint ${resume_from_checkpoint} \
	--valid_size ${valid_size} \
	--max_steps ${max_steps} \
	--log_every_n_steps ${log_every_n_steps} \
	--val_check_interval ${val_check_interval} \
	--limit_val_batches ${limit_val_batches} \
	--start_pfam_trainer ${start_pfam_trainer} \
	--num_workers ${num_workers} \
	--wandb_entity thenaturalmachine \
	--wandb_project "BioM3-dev" \
	--wandb_name ${version_name} \
