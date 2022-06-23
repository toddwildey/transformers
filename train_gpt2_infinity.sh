#!/bin/bash

# Train gpt2_infinity model under ../transformers
source .env/bin/activate

DATASET_NAME="wikitext"
DATASET_CONFIG_NAME="wikitext-2-raw-v1"
OUTPUT_DIR="../models/gpt2_infinity/checkpoints"

rm -Rf "$OUTPUT_DIR"

python ./examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path=gpt2 \
    --model_type=gpt2_infinity \
    --config_name=infinite_memory_transformer_sticky_mem \
    --per_device_eval_batch_size=1 \
    --per_device_train_batch_size=1 \
    --dataset_name "$DATASET_NAME" \
    --dataset_config_name "$DATASET_CONFIG_NAME" \
    --do_train \
    --block_size=512 \
    --output_dir="$OUTPUT_DIR" \
    2>&1 | tee results

