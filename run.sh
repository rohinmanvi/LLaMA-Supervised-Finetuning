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

python3 train_lora.py \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --data_path ./test_data.json \
    --bf16 True \
    --output_dir ./checkpoints \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048