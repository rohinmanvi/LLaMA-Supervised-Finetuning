#!/bin/bash
#SBATCH -J setup

ml python/3.9.0
ml cuda/12.0.0
ml gcc/12.1.0

pip3 install --user --upgrade pip

pip3 install torch
pip3 install git+https://github.com/huggingface/transformers.git
pip3 install git+https://github.com/huggingface/peft.git
pip3 install datasets loralib sentencepiece accelerate bitsandbytes
pip3 install trl
pip3 install gym
pip3 install wandb