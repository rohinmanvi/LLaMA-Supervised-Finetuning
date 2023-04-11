#!/bin/bash
#SBATCH -J LLaMA
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:24GB

nvidia-smi

git pull

ml python/3.9.0
ml cuda/12.0.0
ml gcc/12.1.0

python3 generate.py