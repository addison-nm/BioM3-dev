
# module use /soft/modulefiles
# module load conda
# conda activate /grand/NLDesignProtein/biom3_env

NGPUS=1
NNODES=1

datetime=$(date +%Y%m%d_%H%M%S)
# NRANKS=8 # Number of MPI ranks to spawn per node

export version_name="V${datetime}_${NNODES}_nodes"
export resume_from_checkpoint='None'

export epoch=2 # number of epochs
export max_steps=100000 # number of steps
export val_check_interval=100 # checkpoint interval
export limit_val_batches=0.05 #0.0001 #0.25 # validation size # TN: kept to a small value in scaling study 
export log_every_n_steps=1
export start_pfam_trainer='true' # starting checkpoint: 'true'; continue checkpoint: 'false'

export gpu_devices=${NGPUS}
export num_nodes=${NNODES}

./scripts/PL_train_transformer.sh \
    $version_name \
    $epoch \
    $resume_from_checkpoint \
    $max_steps \
    $val_check_interval \
    $limit_val_batches \
    $start_pfam_trainer \
    $gpu_devices \
    $num_nodes > my_log_file.o 2>&1
