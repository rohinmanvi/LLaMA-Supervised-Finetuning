#!/bin/bash
#SBATCH --time=00:10:00

ml python/3.9.0 cuda/12.0.0 gcc/12.1.0
nvidia-smi

python3 src/highway_planner.py