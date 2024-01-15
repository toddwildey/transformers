#!/bin/bash

BASE_GPT2_CONFIG_JSON_PATH="$1"

jq --argjson source "$(cat "$BASE_GPT2_CONFIG_JSON_PATH" | jq -c)" \
        'to_entries | .[] | select(($source[.key] | not) or ($source[.key] != .value))' \
    | jq -s 'from_entries'
