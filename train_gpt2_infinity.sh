#!/bin/bash

# Train gpt2_infinity model under ../transformers
source .env/bin/activate

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL_BASE_NAME="${1:-gpt2}"
DATASET_NAME="${DATASET_NAME:-"wikimedia/wikipedia"}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-"20231101.en"}"
OUTPUT_DIR="../models/${MODEL_BASE_NAME}_infinity/focused/checkpoints"

# while true; do
    python ./examples/pytorch/language-modeling/run_clm.py \
        --model_name_or_path "${MODEL_BASE_NAME}" \
        --model_type=gpt2_infinity \
        --config_name "infinite_memory_transformer_sticky_mem/${MODEL_BASE_NAME}_infinity_config.json" \
        --per_device_eval_batch_size=1 \
        --per_device_train_batch_size=1 \
        --dataset_name "$DATASET_NAME" \
        --dataset_config_name "$DATASET_CONFIG_NAME" \
        --do_train \
        --save_steps=1000 \
        --save_total_limit=2 \
        --block_size=512 \
        --output_dir="$OUTPUT_DIR"
        2>&1
# done
