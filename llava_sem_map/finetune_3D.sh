#!/bin/bash

export PATH=/opt/rh/devtoolset-8/root/usr/bin:$PATH

WEIGHT_VERSION=v1
PROMPT_VERSION=v1
MODEL_VERSION=7b

deepspeed --include localhost:5 \
    --master_port 29506 \
    llava/train/train_procthor.py \
    --deepspeed /mnt/data-2/data/yh/LLaVA_3D/scripts/zero3_offload.json \
    --model_name_or_path /mnt/data-2/data/yh/LLaVA_3D/checkpoints/vicuna-7b-v1 \
    --version $WEIGHT_VERSION \
    --data_path ./dataset_3D/TaPA/llava_vln_parsed_response_100_v6.json \
    --image_folder ./dataset_3D/TaPA/v1_semantic_map \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/deepspeed_llava-7b-TaPA-pretrain-multimodal-2024-1-15/mm_projector.bin \
    --mm_projector_type linear \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./output/deepspeed_llava-7b-TaPA-finetune-multimodal-2024-1-15-step-2 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --mm_projector_hidden_size 900 \
    --poni_pretrain_model ./checkpoints/poni/poni_seed_123_MP3D.pt \
    --report_to wandb
