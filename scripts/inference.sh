#!/bin/bash
#SBATCH -J LLaMA
#SBATCH -o ./output/inference-$(date +%Y-%m-%d_%H-%M-%S).out
#SBATCH -p gpu
#SBATCH -G 1

ml python/3.9.0 cuda/12.0.0 gcc/12.1.0
nvidia-smi

cd ..
python3 src/inference.py