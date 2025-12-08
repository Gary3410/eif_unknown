import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import json
from tqdm import tqdm
import numpy as np
import os
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    hidden_size: Optional[int] = field(default=768)
    qformer_num_query_token: Optional[int] = field(default=32)
    qformer_vision_width: Optional[int] = field(default=768)
    poni_pretrain_model: Optional[str] = field(default=None)
    mm_projector_hidden_size: Optional[int] = field(default=900)


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def build_model_args():
    model_args = ModelArguments(
        freeze_backbone=False,
        mm_use_im_patch_token=False,
        mm_use_im_start_end=False,
        mm_vision_select_feature='patch',
        mm_vision_select_layer=-2,
        model_name_or_path='/mnt/data-2/data/yh/LLaVA_3D/checkpoints/deepspeed_llava-7b-TaPA-finetune-multimodal-2024-1-11',
        pretrain_mm_mlp_adapter='/mnt/data-2/data/yh/LLaVA_3D/checkpoints/deepspeed_llava-7b-TaPA-multimodal-2024-1-10/mm_projector.bin',
        tune_mm_mlp_adapter=False,
        version='v1',
        vision_tower='/mnt/data-2/data/yh/LLaVA_3D/checkpoints/vision_tower/openscene.pth',
        hidden_size=7680,
        qformer_num_query_token=32,
        qformer_vision_width=7680,
        poni_pretrain_model=None
    )
    return model_args


def main(args):
    # Model
    disable_torch_init()

    model_args = build_model_args()

    model_args.mm_projector_hidden_size = args.mm_projector_hidden_size
    model_args.poni_pretrain_model = args.poni_pretrain_model
    model_args.model_name_or_path = args.model_path
    model_args.pretrain_mm_mlp_adapter = args.pretrain_mm_mlp_adapter

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    print("启动视觉分支初始化")
    model.get_model().initialize_vision_modules(model_args=model_args)
    model.get_model().to(torch.bfloat16)
    model.lm_head.to(torch.bfloat16)
    model.get_model().cuda()

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    # 保存结果路径
    response_dict_list = []
    # dataset_base_path = "./dataset_3D/TaPA/llava_vln_parsed_response_100_v6_no_object_val.json"
    dataset_base_path = args.val_dataset_path
    dataset_list = json.load(open(dataset_base_path))
    for dataset_id, dataset_one in enumerate(tqdm(dataset_list)):

        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        conv_list = dataset_one["conversations"]

        # 先处理图像
        # map_base_path = "./dataset_3D/TaPA/v1_semantic_map"
        map_base_path = args.map_base_path
        map_one_name = dataset_one["image"]
        # map_one = np.load(os.path.join(map_base_path, map_one_name))
        map_one = np.load(os.path.join(map_base_path, map_one_name))["map"]
        map_one = map_one[:23, :, :]

        image_tensor = torch.from_numpy(map_one)
        # image_tensor = torch.tensor(map_one)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        image_tensor = image_tensor.to(torch.bfloat16)
        image_tensor = image_tensor.unsqueeze(0)

        for conv_input_id in [0, 2]:
            response_dict = {}
            inp = conv_list[conv_input_id]["value"]
            response_dict["input"] = inp
            response_dict["output_gt"] = conv_list[conv_input_id + 1]["value"]

            if conv_input_id == 0:
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
            else:
                conv.append_message(conv.roles[0], inp)

            # 生成后续回应
            conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = outputs

            response_dict["output"] = outputs

            response_dict_list.append(response_dict)

    # 保存结果
    with open("llava_map_response_v3_step_2.json", 'w') as f:
        json_str = json.dumps(response_dict_list, indent=2)
        f.write(json_str)
        f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--val-dataset-path", type=str, default=None)
    parser.add_argument("--map-base-path", type=str, default=None)
    parser.add_argument("--mm-projector-hidden-size", type=int, default=900)
    parser.add_argument("--poni-pretrain-model", type=str, default=None)
    parser.add_argument("--pretrain-mm-mlp-adapter", type=str, default=None)
    args = parser.parse_args()
    main(args)


