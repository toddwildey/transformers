#!/bin/bash

jq -S 'reduce inputs as $i (.; . * $i)' "$@"
