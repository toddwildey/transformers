#!/bin/bash

# Train gpt2_infinity model under ../transformers
source .env/bin/activate

DATASET_NAME="wikitext"
DATASET_CONFIG_NAME="wikitext-103-raw-v1"
OUTPUT_DIR="../models/gpt2_infinity/focused/checkpoints"

./clean_old_checkpoints.sh "$OUTPUT_DIR" 2>&1 > /dev/null &
CLEAN_CHECKPOINTS_PID="$?"
disown %1

function cleanup() {
    kill "$CLEAN_CHECKPOINTS_PID"
    exit
}

trap cleanup EXIT
trap cleanup SIGHUP
trap cleanup SIGINT
trap cleanup SIGQUIT
trap cleanup SIGABRT
trap cleanup SIGKILL
trap cleanup SIGALRM
trap cleanup SIGTERM

while true; do
    python ./examples/pytorch/language-modeling/run_clm.py \
        --model_name_or_path=gpt2 \
        --model_type=gpt2_infinity \
        --config_name=infinite_memory_transformer_sticky_mem \
        --per_device_eval_batch_size=1 \
        --per_device_train_batch_size=1 \
        --dataset_name "$DATASET_NAME" \
        --dataset_config_name "$DATASET_CONFIG_NAME" \
        --do_train \
        --save_steps=1000 \
        --block_size=512 \
        --output_dir="$OUTPUT_DIR" \
        2>&1
done
