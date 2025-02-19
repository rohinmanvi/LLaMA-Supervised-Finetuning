#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=00:59:00

ml python/3.9.0 cuda/12.0.0 gcc/12.1.0
nvidia-smi

python3 src/highway_llama_no_sequence_simulation.py