export PYTHONPATH="$PWD"
DATA_DIR="/home2/shravya.k/weakly_labelled"
BERT_DIR="/home2/shravya.k/uncased_L-12_H-768_A-12"

BERT_DROPOUT=0.1
MRC_DROPOUT=0.3
LR=3e-5
SPAN_WEIGHT=0.1
WARMUP=0
MAXLEN=128
MAXNORM=1.0

OUTPUT_DIR="/scratch/shravya.k/train_logs/sebi/sebi_20210330reproduce_lr${LR}_drop${MRC_DROPOUT}_norm${MAXNORM}_bsz32_hard_span_weight${SPAN_WEIGHT}_warmup${WARMUP}_maxlen${MAXLEN}_newtrunc_debug"
mkdir -p $OUTPUT_DIR
python /home2/shravya.k/mrc-for-flat-nested-ner/trainer.py \
--data_dir $DATA_DIR \
--bert_config_dir $BERT_DIR \
--max_length $MAXLEN \
--batch_size 8 \
--gpus="0,1,2,3" \
--distributed_backend=ddp \
--precision=16 \
--progress_bar_refresh_rate 1 \
--lr $LR \
--val_check_interval 0.5 \
--accumulate_grad_batches 2 \
--default_root_dir $OUTPUT_DIR \
--mrc_dropout $MRC_DROPOUT \
--bert_dropout $BERT_DROPOUT \
--max_epochs 20 \
--span_loss_candidates "pred_and_gold" \
--weight_span $SPAN_WEIGHT \
--warmup_steps $WARMUP \
--max_length $MAXLEN \
--gradient_clip_val $MAXNORM
