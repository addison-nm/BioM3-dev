
export swissprot_data_root="./data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5"
export pfam_data_root="./data/Stage2_MMD_pfam_embedding_subset_1000.hdf5"
export sequence_keyname="sequence"
export num_classes=29
export num_y_class_labels=6

export wandb_entity="thenaturalmachine"
export wandb_project="BioM3"
export wandb_logging_dir="./logs"
export wandb_tags="debug"

export choose_optim="AdamW"
export lr=1e-4
export weight_decay=1e-6
export scheduler_gamma="coswarmup"

export seed=42
export epochs=5
export valid_size=0.2
export enter_eval=200
export resume_from_checkpoint=None
export batch_size=8
export output_folder=None
export model_option=transformer
export diffusion_steps=1024
export warmup_steps=100
export image_size=32
export ema_inv_gamma=1.0
export ema_power=0.75
export ema_max_value=0.95
export task=proteins
export max_steps=150

export device=cuda
export precision=32
export gpu_devices=2
export num_nodes=1
export acc_grad_batches=1

# For 2 GPU
export val_check_interval=50
export limit_val_batches=0.1

export log_every_n_steps=10
export start_pfam_trainer=False
export num_workers=4

# Flow params
export num_steps=1
export actnorm=False
export perm_channel=none
export perm_length=reverse

export input_dp_rate=0.0

# Transformer params
export transformer_dim=1024
export transformer_heads=16
export transformer_depth=32
export transformer_blocks=1
export transformer_dropout=0.1
export transformer_reversible=False
export transformer_local_heads=8
export transformer_local_size=128

export text_emb_dim=512
export facilitator=MMD

export version_name=config_1node_1gpu
export output_hist_folder=./logs/history/test_stage3
export tb_logger_path=./logs/history
export tb_logger_folder=Stage3_history
export save_hist_path=None

export traindata_len=None
