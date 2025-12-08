#!/bin/bash

export PATH=/opt/rh/devtoolset-8/root/usr/bin:$PATH

WEIGHT_VERSION=v1
PROMPT_VERSION=v1
MODEL_VERSION=7b

deepspeed --include localhost:2 \
    --master_port 29503 \
    llava/train/train_procthor.py \
    --deepspeed /mnt/data-2/data/yh/LLaVA_3D/scripts/zero3_offload.json \
    --model_name_or_path /mnt/data-2/data/yh/LLaVA_3D/checkpoints/vicuna-7b-v1 \
    --version $WEIGHT_VERSION \
    --data_path /mnt/data-2/data/yh/LLaVA_3D/dataset_3D/TaPA/llava_vln_parsed_response_100_v6.json \
    --image_folder /mnt/data-2/data/yh/LLaVA_3D/dataset_3D/TaPA/v1_semantic_map \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_projector_type linear \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/deepspeed_llava-7b-TaPA-pretrain-multimodal-2024-1-15 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --mm_projector_hidden_size 900 \
    --poni_pretrain_model ./checkpoints/poni/poni_seed_123_MP3D.pt \
    --report_to wandb
