#!/bin/bash

# sudo apt update
# sudo apt upgrade

sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

sudo apt install python3.11 python3.11-venv python3.11-distutils

sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
