#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

source $SCRIPT_DIR/.env/bin/activate

wandb login

python $@
