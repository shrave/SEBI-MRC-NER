#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 
# Author: Xiaoy Li 
# On Two 16G P100


EXP_ID=22_1
FOLDER_PATH=/home2/shravya.k/SEBI-MRC-NER 
CONFIG_PATH=${FOLDER_PATH}/config/en_bert_base_uncased.json
DATA_PATH=/home2/shravya.k/ace2004
BERT_PATH=/home2/shravya.k/uncased_L-12_H-768_A-12
EXPORT_DIR=/scratch/shravya.k/ace2004


max_seq_length=150
learning_rate=4e-05
start_loss_ratio=1.0
end_loss_ratio=1.0
span_loss_ratio=0.9
dropout=0.1
train_batch_size=8
dev_batch_size=8
test_batch_size=8
max_train_expoch=10
warmup_proportion=-1
gradient_accumulation_step=1
checkpoint=600
seed=2333
n_gpu=1
data_sign=ace2004
entity_sign=nested
model_sign=ace2004
output_path=${EXPORT_DIR}/${data_sign}/${model_sign}-${data_sign}-${EXP_ID}-${max_seq_length}-${learning_rate}-${train_batch_size}-${dropout}


mkdir -p ${output_path}
export PYTHONPATH=${FOLDER_PATH}


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ${FOLDER_PATH}/run/train_bert_mrc.py \
--data_dir ${DATA_PATH} \
--n_gpu ${n_gpu} \
--dropout ${dropout} \
--entity_sign ${entity_sign} \
--data_sign ${data_sign} \
--bert_model ${BERT_PATH} \
--config_path ${CONFIG_PATH} \
--output_dir ${output_path} \
--checkpoint ${checkpoint} \
--max_seq_length ${max_seq_length} \
--train_batch_size ${train_batch_size} \
--dev_batch_size ${dev_batch_size} \
--test_batch_size ${test_batch_size} \
--learning_rate ${learning_rate} \
--weight_start ${start_loss_ratio} \
--weight_end ${end_loss_ratio} \
--weight_span ${span_loss_ratio} \
--num_train_epochs ${max_train_expoch} \
--seed ${seed} \
--warmup_proportion ${warmup_proportion} \
--gradient_accumulation_steps ${gradient_accumulation_step} 




