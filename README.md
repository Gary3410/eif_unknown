# Unknown Environment EIF
Official implementation of [Embodied Instruction Following in Unknown Environments](https://arxiv.org/abs/2406.11818).

The repository contains:
- The [data](#data-release) used for fine-tuning the model.
- The code for [generating the data](#data-generation-process).
- The code for [fine-tuning the models](#fine-tuning) on RTX 3090 GPUs with [LoRA](https://github.com/microsoft/LoRA).
- The code for [inference](#Running-the-inference).
- The code for [visualization](#visualization).

## 📝 TODO
- [x] Provide inference scripts.
- [ ] Step-by-step initialization tutorial.
- [x] Release pretrained models.
- [ ] Release fine-tuning datasets.
- [ ] Release data-generation scripts.
- [ ] Provide a Dockerfile for installation.

## Setup

### Environmental dependencies
This repository contains the following components:
- Extensive indoor simulation environment [ProcTHOR](https://github.com/allenai/procthor).
- High-level planner and low-level controller [LLaVA](https://github.com/haotian-liu/LLaVA).
- Scene semantic feature extraction [CLIP & OpenCLIP](https://github.com/openai/CLIP).
- Semantic feature map fusion [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).
- Adaptive weight generation [LongCLIP](https://github.com/beichenzbc/Long-CLIP).
- Open vocabulary instance segmentation model [Detic](https://github.com/facebookresearch/Detic).
- Instance segmentation model [Mask R-CNN](https://github.com/soyeonm/FILM/tree/public/models/segmentation).

### Data preparation
The following data needs to be downloaded:
- ProcTHOR simulator room [layout](https://huggingface.co/Gary3410/eif_unknown/tree/main/procthor_house).
- Fine-tuned [high-level planner](https://huggingface.co/Gary3410/eif_unknown/tree/main/llava-vicuna-v1-3-7b-finetune-planner-lora-high-level-planner) and [low-level controller](https://huggingface.co/Gary3410/eif_unknown/tree/main/llava-vicuna-v1-3-7b-finetune-frontier-lora-low-level-controller).
- Instance segmentation model [weights](https://huggingface.co/Gary3410/eif_unknown/tree/main/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size_procthor) (optional).
- Instruction tuning dataset (optional).

The file directory should be:

```
eif_unknown
├── checkpoints
│   ├── bert-large-uncased
│   ├── CLIP-ViT-H-14-laion2B-s32B-b79K
│   ├── clip-vit-large-patch14
│   ├── llava-vicuna-v1-3-7b-finetune-planner-lora-high-level-planner
│   ├── llava-vicuna-v1-3-7b-finetune-frontier-lora-low-level-controller
│   ├── vicuna-v1-3-7b
├── create_dataset
├── data
├── llava
│   ├── eval
│   ├── model
│   ├── serve
│   │   ├── cli_llava_v3_llm_planner.py
│   │   ├── cli_llava_v3_procthor_maskrcnn.py
│   │   ├── ...
│   ├── train
├── log_file
├── model
│   ├── bpe_simple_vocab_16e6.txt.gz
│   ├── longclip.py
│   ├── model_longclip.py
│   ├── ...
├── models
│   ├── Detic
│   │   ├──datasets
│   │   ├──models
│   │   │   ├──BoxSup-C2_L_CLIP_SwinB_896b32_4x.pth
│   │   │   ├──BoxSup-C2_Lbase_CLIP_SwinB_896b32_4x.pth
│   │   │   ├──BoxSup-C2_LCOCO_CLIP_SwinB_896b32_4x.pth
│   │   │   ├──Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size
│   │   │   ├──swin_base_patch4_window7_224_22k.pkl
│   │   │   ├──swin_base_patch4_window7_224_22k.pth
│   │   ├── third_party
│   │   │   ├── CenterNet2
│   │   │   ├── Deformable-DETR
│   │   ├── ...
│   ├── segmentation
│   │   ├──maskrcnn_alfworld
│   │   │   ├──mrcnn_alfred_objects_008_v3.pth
│   │   ├──segmentation_helper.py
│   │   ├──segmentation_helper_procthor.py
│   │   ├──segmentation_helper_procthor_detic.py
├── docs
├── procthor_house
│   ├── test.jsonl.gz
│   ├── train.jsonl.gz
│   ├── val.jsonl.gz
├── scripts
├── utils
├── visualization
......
```

## 🧪 Evaluation
It is recommended to use at least four 3090 GPUs, or you can evaluate by modifying the configuration with two 3090 GPUs
To evaluate the checkpoint, you can use:

```
# Oracle setting
# Easy task
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m llava.serve.cli_llava_v3_nav_seg_gt \
    --model-path ./checkpoints/llava-vicuna-v1-3-7b-finetune-frontier-lora-low-level-controller \
    --model-path-s1 ./checkpoints/llava-vicuna-v1-3-7b-finetune-planner-lora-high-level-planner \
    --model-base ./checkpoints/vicuna-v1-3-7b \
    --image-file ./vision_dataset/llava_dataset_v8_easy_train/frontiers_feature \
    --val-file ./data/spaced_parse_instruction_easy_v12_val.json
```
```
# Detic setting
# Easy task
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m llava.serve.cli_llava_v3_procthor_maskrcnn \
    --model-path ./checkpoints_output/llava-vicuna-v1-3-7b-finetune-frontier-lora-low-level-controller \
    --model-path-s1 ./checkpoints_output/llava-vicuna-v1-3-7b-finetune-planner-lora-high-level-planner \
    --model-base ./checkpoints/vicuna-v1-3-7b \
    --image-file ./vision_dataset/llava_dataset_v8_easy_train/frontiers_feature \
    --val-file ./data/spaced_parse_instruction_easy_v12_val.json
```

# 🏷️ License
This repository is released under the MIT license.

# 🥰 Citation
If you find this repository helpful, please consider citing:

```
@article{wu2024embodied,
  title={Embodied instruction following in unknown environments},
  author={Wu, Zhenyu and Wang, Ziwei and Xu, Xiuwei and Lu, Jiwen and Yan, Haibin},
  journal={arXiv preprint arXiv:2406.11818},
  year={2024}
}
```
