#!/bin/bash

export DATASET='AWA1'
export MODE='validation' #'test'
export CODE_DIR="."
export DATA_DIR="./datasets/AWA1P" # required (preprocessed AWA1 directory!)

# set params
source ${CODE_DIR}/scripts/AWA1_hps.sh

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