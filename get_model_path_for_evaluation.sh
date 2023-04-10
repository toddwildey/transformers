#!/bin/bash

MODEL_PATH_DIR="$1"
DIR_NAME_PREFIX="${2:-"checkpoint"}"

CHECKPOINT_DIRS=$(find "$MODEL_PATH_DIR" -maxdepth 1 -mindepth 1 -type "d" -exec basename {} \; | grep "$DIR_NAME_PREFIX")
CHECKPOINT_NUMBERS=$(echo "$CHECKPOINT_DIRS" | sed -E "s/$DIR_NAME_PREFIX-([[:digit:]]+)/\\1/")
CHECKPOINT_NUMBERS_SORTED=$(echo "$CHECKPOINT_NUMBERS" | sort -nr)
LAST_CHECKPOINT_NUMBER=$(echo "$CHECKPOINT_NUMBERS_SORTED" | head -n1)

echo "$DIR_NAME_PREFIX-$LAST_CHECKPOINT_NUMBER"
