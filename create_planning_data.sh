#!/bin/bash
#SBATCH --time=24:00:00

ml python/3.9.0 cuda/12.0.0 gcc/12.1.0

python3 src/create_planning_data.py