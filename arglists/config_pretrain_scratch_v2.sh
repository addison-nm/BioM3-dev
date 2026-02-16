
export swissprot_data_root="./data/Stage2_MMD_swissprot_embedding_last_ckpt_all.hdf5"
export pfam_data_root="None"
export sequence_keyname="sequence"              # VERIFIED HF
export num_classes=29                           # VERIFIED HF
export num_y_class_labels=6                     # VERIFIED HF

export wandb_entity="thenaturalmachine"
export wandb_project="BioM3"
export wandb_logging_dir="./logs"
export wandb_tags="pretrain_from_scratch"

export choose_optim="AdamW"
export lr=1e-4
export weight_decay=1e-6
export scheduler_gamma="coswarmup"

export seed=42
export epochs=100
export valid_size=0.2                               # PAPER SAYS 0.2, HF SAYS 0.1
export enter_eval=1000                          # VERIFIED HF
export resume_from_checkpoint=None              # TRAINING FROM SCRATCH
export batch_size=32                                # PAPER SAYS 32, HF SAYS 16
export output_folder=None                           # CONSIDER
export model_option=transformer                 # VERIFIED
export diffusion_steps=1024                     # VERIFIED HF + PAPER
export warmup_steps=500                         # VERIFIED HF + PAPER
export image_size=32                            # VERIFIED
export ema_inv_gamma=1.0                            # CHECK (VERIFIED HF)
export ema_power=0.75                               # CHECK (VERIFIED HF)
export ema_max_value=0.95                           # CHECK (VERIFIED HF)
export task=proteins                            # VERIFIED
export max_steps=3000000                            # COMPUTE OR CHECK

export device=cuda                              # CONFIDENT
export precision=bf16                               # NEED TO TEST
export gpu_devices=4                            # OVERRIDE
export num_nodes=1                              # OVERRIDE
export acc_grad_batches=1                           # CONSIDER

export val_check_interval=20                       # COMPUTE OR CHECK
export limit_val_batches=0.05                       # COMPUTE OR CHECK

export log_every_n_steps=100                       # CHECK
export start_pfam_trainer=False                 # CONFIDENT
export num_workers=1                                # NEED TO TEST

# Flow params
export num_steps=1
export actnorm=False
export perm_channel=none
export perm_length=reverse

export input_dp_rate=0.0                        # CHECK

# Transformer params
export transformer_dim=512                      # VERIFIED HF
export transformer_heads=16                     # VERIFIED HF
export transformer_depth=16                     # VERIFIED HF
export transformer_blocks=16                    # VERIFIED HF
export transformer_dropout=0.1                  # VERIFIED HF
export transformer_reversible=False             # VERIFIED HF
export transformer_local_heads=8                # VERIFIED HF
export transformer_local_size=128               # VERIFIED HF

export text_emb_dim=512                         # VERIFIED HF
export facilitator=MMD                          # VERIFIED HF

export version_name=config_1node_1gpu
export output_hist_folder=./logs/history/test_stage3
export tb_logger_path=./logs/history
export tb_logger_folder=Stage3_history
export save_hist_path=None

export traindata_len=None
