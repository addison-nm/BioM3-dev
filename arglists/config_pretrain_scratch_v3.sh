#=============================================================================
# Configuration file for pretraining model. Increased number of transformer
# blocks from 1 (as used in paper) to 16. Uses learning rate of 1e-2.
#=============================================================================

export swissprot_data_root="./data/Stage2_MMD_swissprot_embedding_last_ckpt_all.hdf5"
export pfam_data_root="None"
export sequence_keyname="sequence"
export num_classes=29
export num_y_class_labels=6

export wandb_entity="thenaturalmachine"
export wandb_project="BioM3"
export wandb_logging_dir="./logs"
export wandb_tags="pretrain_from_scratch"

export choose_optim="AdamW"
export lr=1e-2
export weight_decay=1e-6
export scheduler_gamma="coswarmup"

export seed=42
export epochs=100
export valid_size=0.2
export enter_eval=1000
export resume_from_checkpoint=None
export batch_size=32
export output_folder=None
export model_option=transformer
export diffusion_steps=1024
export warmup_steps=500
export image_size=32
export ema_inv_gamma=1.0
export ema_power=0.75
export ema_max_value=0.95
export task=proteins
export max_steps=3000000

export device=cuda
export precision=bf16
export gpu_devices=4
export num_nodes=1
export acc_grad_batches=1

export val_check_interval=20
export limit_val_batches=0.05

export log_every_n_steps=100
export start_pfam_trainer=False
export num_workers=1

# Flow params
export num_steps=1
export actnorm=False
export perm_channel=none
export perm_length=reverse

export input_dp_rate=0.0

# Transformer params
export transformer_dim=512
export transformer_heads=16
export transformer_depth=16
export transformer_blocks=16
export transformer_dropout=0.1
export transformer_reversible=False
export transformer_local_heads=8
export transformer_local_size=128

export text_emb_dim=512
export facilitator=MMD

export version_name=None
export output_hist_folder=./logs/history/test_stage3
export tb_logger_path=./logs/history
export tb_logger_folder=Stage3_history
export save_hist_path=None

export traindata_len=None
