#!/bin/bash

export DATASET='AWA2'
export MODE='validation' #'test'
export CODE_DIR="."
export DATA_DIR="./datasets/AWA2P" # required (preprocessed AWA2 directory!)

# set params
source ${CODE_DIR}/scripts/AWA2_hps.sh

#train ALE
python ${CODE_DIR}/train.py \
			--dataset=$DATASET \
			--mode=$MODE \
			--data_dir=$DATA_DIR \
			--optim_type=$OPTIM_TYPE \
			--lr=$LR \
			--wd=$WD \
			--lr_decay=$LR_DECAY \
			--n_epoch=$N_EPOCH \
			--batch_size=$BATCH_SIZE