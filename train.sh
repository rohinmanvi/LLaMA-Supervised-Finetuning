#!/bin/bash
#SBATCH -J LLaMA
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:32GB
#SBATCH --time=20:00:00

nvidia-smi

git pull

ml python/3.9.0
ml cuda/12.0.0
ml gcc/12.1.0

python3 train.py \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --data_path ./data.jsonl \
    --output_dir ./llama-driver3 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-3 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --fp16 True \
    --model_max_length 2048 \
    --report_to wandb