#!/bin/bash

[[ ! -d .env ]] && python -m venv .env

source .env/bin/activate

python setup.py install

pip install torch torchvision torchaudio datasets entmax matplotlib --upgrade
