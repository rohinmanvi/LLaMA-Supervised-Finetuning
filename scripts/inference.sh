#!/bin/bash
#SBATCH -J LLaMA
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:24GB

ml python/3.9.0 cuda/12.0.0 gcc/12.1.0
nvidia-smi

cd ..
python3 inference.py