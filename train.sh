#!/bin/bash
#SBATCH -J LLaMA
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:32GB

nvidia-smi

git pull

ml python/3.9.0
ml cuda/12.0.0
ml gcc/12.1.0

python3 train.py \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --model_name_or_path chainyo/alpaca-lora-7b \
    --data_path ./test_data.json \
    --output_dir ./alpaca \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-3 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fp16 True \
    --model_max_length 2048