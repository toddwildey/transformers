#!/bin/bash

source build.sh

pip install torch torchvision torchaudio datasets accelerate sentencepiece wandb --upgrade
