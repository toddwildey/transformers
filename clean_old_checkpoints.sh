#!/bin/bash

OUTPUT_DIR="$1"

while [[ true ]]; do
    sleep 60

    CHECKPOINT_DIRS=$(find "$OUTPUT_DIR" -maxdepth 1 -mindepth 1 -type "d" -exec basename {} \; | grep "checkpoint")
    CHECKPOINT_NUMBERS=$(echo "$CHECKPOINT_DIRS" | sed -E 's/checkpoint-([[:digit:]]+)/\1/')
    CHECKPOINT_NUMBERS_SORTED=$(echo "$CHECKPOINT_NUMBERS" | sort -nr)
    LAST_CHECKPOINT_NUMBER=$(echo "$CHECKPOINT_NUMBERS_SORTED" | head -n1)
    LAST_CHECKPOINT="checkpoint-$LAST_CHECKPOINT_NUMBER"
    OLD_CHECKPOINTS=$(echo "$CHECKPOINT_DIRS" | grep -v "$LAST_CHECKPOINT")

    if [[ ! -z "$OLD_CHECKPOINTS" ]]; then
        echo "$OLD_CHECKPOINTS" | xargs -I{} rm -Rf "$OUTPUT_DIR/{}"
    fi
done
