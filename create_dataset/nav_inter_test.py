from utils.generate_response_llava import Planner
import numpy as np
import os
import blosc
from tqdm import tqdm
import json
import argparse
import torch
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from utils.generate_response_llava import Planner


def main(args):
    # Model
    disable_torch_init()
    # load llava
    model_name = get_model_name_from_path(args.model_path)
    llava_tokenizer, llava_model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    # 定制llava conv_mode
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    print(conv_mode)
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    planner = Planner(
        llava_model_s2=llava_model,
        llava_args=args,
        llava_tokenizer=llava_tokenizer,
        llava_model_s1=None
    )

    base_path = "./vision_dataset/nav_inter_dataset"
    feature_base_path = os.path.join(base_path, "vision_feature_blosc_v3")
    test_file_path = os.path.join(base_path, "test_nav_inter.json")
    test_result_path = os.path.join(base_path, "test_nav_inter_response_v3.json")

    success_number = 0
    recep_number = 0
    recep_acc = 0
    test_file = json.load(open(test_file_path))
    result_list = []
    for test_file_one in tqdm(test_file):
        instruction = test_file_one["conversations"][0]["value"]
        gt_response = test_file_one["conversations"][1]["value"]
        image_file = test_file_one["image"]
        image_path = os.path.join(feature_base_path, image_file)
        feature_shape = (256, 1024)
        with open(image_path, 'rb') as f:
            loaded_compressed_array = f.read()
        # # Decompress the data
        decompressed_array = np.frombuffer(blosc.decompress(loaded_compressed_array), dtype=np.float16)
        decompressed_array = decompressed_array.reshape(feature_shape)
        feature = np.asarray(decompressed_array)

        response = planner.get_llava_response_s2(instruction, feature)

        # 解析llava response
        response = response.strip("</s>")
        response = response.strip("<s> ")
        nav_choose = response.split("The answer is ")[-1].split(", ")[0]
        gt_choice = gt_response.split("The answer is ")[-1].split(", ")[0]
        if gt_choice == nav_choose:
            success_number += 1
        if "Fridge" in instruction:
            recep_number += 1
            if gt_choice == nav_choose:
                recep_acc += 1
        result_one = dict(instruction=instruction, response=gt_response, pred_response=response)
        result_list.append(result_one)
    print("acc: ", success_number / len(test_file))
    print(recep_number)
    print("recep acc: ", recep_acc / recep_number)

    # 保存结果
    with open(test_result_path, 'w') as f:
        json_str = json.dumps(result_list, indent=2)
        f.write(json_str)
        f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    # add llava_s1 planner args
    parser.add_argument("--model-path-s1", type=str, default="facebook/opt-350m")
    parser.add_argument("--device-s1", type=str, default="cuda:1")
    args = parser.parse_args()
    main(args)
