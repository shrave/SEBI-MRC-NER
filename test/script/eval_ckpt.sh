#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# 

REPO_PATH=/home2/shravya.k/SEBI-MRC-NER
export PYTHONPATH=${REPO_PATH}

config_path=${REPO_PATH}/config/en_bert_base_uncased.json
data_path=/home2/shravya.k/sebi_normal
bert_path=/home2/shravya.k/uncased_L-12_H-768_A-12
saved_model=/home2/shravya.k/mrc_checkpoints/mrc-ner/sebi/mrc-ner-sebi-2021.4.1-22_1-180-8e-04-4-0.2/bert_finetune_model_0_3.bin
max_seq_length=150
test_batch_size=2
data_sign=sebi
entity_sign=nested
n_gpu=1
seed=2333



CUDA_VISIBLE_DEVICES=0,1 python3 ${REPO_PATH}/run/evaluate_mrc_ner.py \
--config_path ${config_path} \
--data_dir ${data_path} \
--bert_model ${bert_path} \
--saved_model ${saved_model} \
--max_seq_length ${max_seq_length} \
--test_batch_size ${test_batch_size} \
--data_sign ${data_sign} \
--entity_sign ${entity_sign} \
--n_gpu ${n_gpu} \
--seed ${seed}
