#!/bin/bash

export version_name='V20250520_test_model'
#export model_checkpoint_dir='/lus/grand/projects/mm_protein/Project_TextDiffEDP/history/history/Stage3_history/FINAL_Stage3_small_ProtARDM_mmd_phase1' 

export resume_from_checkpoint='None' #'/lus/grand/projects/mm_protein/Project_TextDiffEDP/history/history/Stage3_history/checkpoints/V20240805_final_phase2/last.ckpt'


#'/lus/grand/projects/mm_protein/Project_TextDiffEDP/history/history/Stage3_history/checkpoints/V20240805_final_phase2/last_epoch300.ckpt' # 'None'
export epoch=375 # number of epochs
export max_steps=3500000 # number of steps
export val_check_interval=200000 # checkpoint interval
export limit_val_batches=0.25 # validation size
export start_pfam_trainer='false' # starting checkpoint: 'true'; continue checkpoint: 'false'



#Run the python script
sh PL_train_transformer.sh \
	$version_name \
	$epoch \
	$resume_from_checkpoint \
	$max_steps \
	$val_check_interval \
	$limit_val_batches \
	$start_pfam_trainer \

