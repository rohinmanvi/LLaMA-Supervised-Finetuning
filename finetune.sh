#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:32GB
#SBATCH --time=00:09:59

ml python/3.9.0 cuda/12.0.0 gcc/12.1.0
nvidia-smi

python3 src/finetune.py \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --data_path data/highway_planner_data_detailed_final.jsonl \
    --output_dir models/highway-no-sequence-3 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-3 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --fp16 True \
    --model_max_length 4096 \
    --report_to wandb