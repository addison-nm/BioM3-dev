#!/usr/bin/env bash

WANDB_API_KEY=$1
config_dir=$2
config_name=$3

# Export args from config file
source ${config_dir}/${config_name}.sh

# Override config file with given args
version_name=$4
num_nodes=$5
gpu_devices=$6
epochs=$7
resume_from_checkpoint=$8

# echo "num_nodes: ${num_nodes} gpu_devices: ${gpu_devices}"

# Login to W&B
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

	# --resume_from_checkpoint_state_dict_path ${resume_from_checkpoint_state_dict_path} \
	#--total-steps ${total_steps} \
	#--output_model_folder ${output_model_folder} \
	#--save_model_path ${save_model_path} \
	#--model_param_df_path ${model_param_df_path} \
	#--checkpoint_path ${checkpoint_path} \


# projdir="/grand/NLDesignProtein/ahowe/BioM3-dev"

# export DIR="$(dirname "$(pwd)")"

# export output_hist_folder="${projdir}/logs/history/test_stage3/" # csv folder output
# export tb_logger_path="${projdir}/logs/history/"
# export tb_logger_folder='Stage3_history'
# export version_name=${1} #Test_stage3_V07182024'

# export output_folder=None
# export save_hist_path=None #'./logs' # path belonging to the csv folder that contains the training results
# export model_option='transformer' # set up transformer training 


# export swissprot_data_root="${projdir}/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5"
# export pfam_data_root="${projdir}/data/Stage2_MMD_pfam_embedding_subset_1000.hdf5"

# # training infor
# export diffusion_steps=1024 #1024 # make sure time trajectory is the length of the data structure
# export seed=42 # for reproducibility
# export batch_size=16 # hp for training
# export warmup_steps=500 # learning rate scheduler
# export image_size=32 #32 # sequence_length = image_size*image_size (artifact from mnist)
# export learning_rate=1e-4 # hp for training
# export weight_decay=1e-6 # hp for training
# export ema_inv_gamma=1.0 # (not needed with pl for now)
# export ema_power=0.75 # (not needed with pl for now)
# export ema_max_value=0.95 # (not needed with pl for now)
# export precision='32' # (not needed with pl for now)
# export device='cuda' # Use GPU
# export num_classes=29 #24 #29 # 20 aa labels, 1 deletion padded token, 1 start token, 1 end token
# export num_y_class_labels=6 # conditional variables
# export task='proteins' # (artifact from mnist)
# export enter_eval=1000 # when to enter eval performance step
# export choose_optim='AdamW'  #options --> 'AdamW', 'AdaFactor', 'Adam', 'DeepSpeedCPUAdam'
# export epochs=${2} # number of epochs (final configured model should run over 100 epochs)
# export acc_grad_batches=1 # number of gradients to accumulate
# export gpu_devices=${8} #4 # number of gpus to use
# export num_nodes=${9} # number of nodes to use
# export scheduler_gamma='coswarmup' #'coswarmup'
# export text_emb_dim=512 #512 #12 #256
# export sequence_keyname='sequence' #'sequence' # 'seq' | 'sequence'
# export facilitator='MMD' 
# export valid_size=0.2

# # parameters only related for training on pfam database...
# export max_steps=${4} #50000 #100000 # note: 452749 iterations for global batch size 80; not relevant without pfam dataset
# export val_check_interval=${5} #1000 # when to start evaluating on validation set; not relevant without pfam dataset
# export limit_val_batches=${6} #0.001 # total percentage of the validation to evaluate on...; not relevant without pfam dataset
# export log_every_n_steps=${5} #1000
# export start_pfam_trainer=${7} #'false' # 'False' if we are training swissprot or continuning training swissprot+pfam | 'True' if transition from swissprot to pfam+swissprot.

# # Resume from checkpoint
# export resume_from_checkpoint=${3}
# export resume_from_checkpoint_state_dict_path=${3}

# # model hyperparameters (small)
# # 1.4B --> dim=2048, heads=16, depth=16
# # 350M --> dim=1024, heads=16, depth=16
# # ~700M --> dim=1024, heads=16, depth=32
# export transformer_dim=1024 #512
# export transformer_heads=16
# export transformer_depth=32 #16

