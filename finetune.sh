#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:32GB
#SBATCH --time=01:00:00

ml python/3.9.0 cuda/12.0.0 gcc/12.1.0
nvidia-smi

python3 src/finetune.py \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --data_path data/highway_planner_sequence_data_incremental_total.jsonl \
    --output_dir models/highway-driver-test-2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-3 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --fp16 True \
    --model_max_length 4096 \
    --report_to wandb