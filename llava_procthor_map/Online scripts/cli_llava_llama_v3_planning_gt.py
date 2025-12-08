import argparse
import random

import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import json
from tqdm import tqdm
import blosc
import numpy as np
import os

# 加载llama_adapter文件
import lightning as L
from pathlib import Path
from typing import Optional
import sys
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from generate import generate
from lit_llama import Tokenizer
from lit_llama.adapter import LLaMA
from lit_llama.utils import lazy_load, llama_model_lookup, quantization
from scripts.prepare_alpaca import generate_prompt
from utils.prompt_utils import get_prompt_type_v5 as get_prompt_type
from utils.procthor_config import Config as proc_Config
import open_clip

from utils.generate_response import Planner
from utils.sem_map import Semantic_Mapping
from utils.thor_env_code_v2 import ThorEnvCode
import prior
import pickle

def perspective_camera_view(controller):
    # 默认视角为30
    normal_view_angle = 0.0
    # 先获取当前视角的角度
    cameraHorizon = controller.last_event.metadata["agent"]["cameraHorizon"]
    if int(cameraHorizon) - normal_view_angle < 0:
        # 需要进行look_down
        event = controller.step(action="LookDown", degrees=normal_view_angle - int(cameraHorizon))
    elif int(cameraHorizon) - normal_view_angle > 0:
        # 需要进行look_up
        event = controller.step(action="LookUp", degrees=int(cameraHorizon) - normal_view_angle)
    else:
        normal_view_angle = 30


def main(args):
    # Model
    disable_torch_init()

    # load llava
    model_name = get_model_name_from_path(args.model_path)
    llava_tokenizer, llava_model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    # load llama
    # reload模型参数
    adapter_path_s1 = "./checkpoints/lit_llama_v1_alpaca_v8_easy_s1/lit-llama-adapter-finetuned.pth"
    quantize = None
    pretrained_path = "./checkpoints/lit-llama/7B/lit-llama.pth"
    tokenizer_path = "./checkpoints/lit-llama/tokenizer.model"

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric_s1 = L.Fabric(devices=[1], precision=precision)
    print("Loading model ...", file=sys.stderr)
    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(adapter_path_s1) as adapter_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric_s1.init_module(empty_init=True), quantization(mode=quantize):
            llama_model = LLaMA.from_name(name)

        # 1. Load the pretrained weights
        llama_model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned adapter weights
        llama_model.load_state_dict(adapter_checkpoint, strict=False)

    llama_model.eval()
    llama_model = fabric_s1.setup(llama_model)
    llama_tokenizer = Tokenizer(tokenizer_path)

    llama_args_dict = dict(
        max_new_tokens=1024,
        top_k=200,
        temperature=0.8
    )

    # load clip
    open_clip_model = "ViT-H-14"
    torch.autograd.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Initializing model...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        open_clip_model, pretrained="./checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
    )
    clip_model.cuda("cuda:0")
    clip_model.eval()

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

    # 生成llama_prompt
    instruction_prompt = get_prompt_type(prompt_type="s1")
    # 加载文件
    total_cat2idx_path = "./utils/total_cat2idx.json"
    total_cat2idx = json.load(open(total_cat2idx_path))
    # 增加wall
    total_cat2idx["wall"] = 94
    total_cat2idx["floor"] = 95

    planner = Planner(
        llava_model=llava_model,
        llava_args=args,
        llama_model=llama_model,
        llava_tokenizer=llava_tokenizer,
        llama_tokenizer=llama_tokenizer,
        llama_args_dict=llama_args_dict
    )

    sem_map = Semantic_Mapping(
        args=proc_Config,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        total_cat2idx=total_cat2idx
    )
    sem_map.reset()

    # 加载测试数据集
    image_base_file = args.image_file
    instruction_dict_list = json.load(open(args.val_file))
    reslut_list = []

    # 开始执行具体的环境配置
    args = proc_Config
    dataset = prior.load_dataset("procthor-10k")
    dataset_train = dataset["train"]
    controller_env = ThorEnvCode(args, use_CloudRendering=True)
    # print(controller_env.last_event.metadata["agent"]["cameraHorizon"])
    # exit()
    # 初始化衡量指标
    success_number = 0
    path_distance_list = []
    success_index_list = []
    nav_fail = 0
    planning_fail = 0
    exec_fail = 0
    # 仅测试失败案例
    with open("./log_file/llava_vln_parsed_response_v8_val_result_v2.pkl", "rb") as f:
        result_info_dict = pickle.load(f)

    new_instruction_dict_list = []
    for val_dict_id, val_dict_one in enumerate(tqdm(instruction_dict_list)):
        result_info_dict_one = result_info_dict[val_dict_id]
        is_success = result_info_dict_one["success"]
        if not is_success:
            new_instruction_dict_list.append(val_dict_one)
    # check_list = [7, 17, 42, 55, 91, 108, 115, 119]
    # check_list = [7]
    # -------------------main loop--------------------------
    for val_dict_id, val_dict_one in enumerate(tqdm(instruction_dict_list)):
        # if val_dict_id not in check_list:
        #     continue
        instruction = val_dict_one["instruction"].strip(" ").strip("\"")
        house_id = int(val_dict_one["house_id"])
        print(house_id)
        print(instruction)
        # if house_id not in [2279]:
        #     continue
        task_type = val_dict_one["task_type"]
        pddl_params = val_dict_one["pddl_params"]
        print(pddl_params)

        rgb_frame, depth_frame = controller_env.load_scene(house_name=dataset_train[house_id])
        success = False
        done_list = "\n"
        done_action_number = 0
        done_action_dict = {}
        instruction_test_time = 0
        path_distance_all = 0
        # controller_env.open_door()
        # 首先进行环视
        sem_map.reset()
        for _ in range(4):
            # 增加转圈
            rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
            info_dict["add_robot_mask"] = True
            info_dict["region_size"] = 0.55
            global_sem_map, global_sem_feature_map = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
            event = controller_env.step(action="RotateRight", forceAction=True)
        print("end surrounding ----------------------")
        # 直接取最后一个进行初步规划
        # 用来已经交互, 但失败的物体
        used_object_id_list = []
        result_dict_one = {}
        result_dict_one["step_action_list"] = []
        is_replanning_action = False
        is_replanning_target = False
        response_action_list = val_dict_one["output"]
        response_action_list_index = 0
        while not success:
            if instruction_test_time >= 1:
                perspective_camera_view(controller_env)
                event = controller_env.step(action="LookDown", degrees=30)
                for _ in range(4):
                    # 增加转圈
                    rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
                    info_dict["add_robot_mask"] = True
                    info_dict["region_size"] = 0.20
                    global_sem_map, global_sem_feature_map = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
                    event = controller_env.step(action="RotateRight", forceAction=True)
                print("end surrounding ----------------------")
                perspective_camera_view(controller_env)
            response_action_dict_one = response_action_list[response_action_list_index]
            rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
            select_feature_list, frontiers_label_dict_list = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="frontiers")
            instruction_test_time = instruction_test_time + 1
            select_s2 = False
            if instruction_test_time > 30:
                success = False
                break
            # 统计尝试次数
            if done_list in done_action_dict.keys():
                done_action_dict[done_list] = done_action_dict[done_list] + 1
            else:
                done_action_dict[done_list] = 0
            try_times = done_action_dict[done_list]
            replanning = False
            if try_times > 10:
                break
            # if try_times > 1:
            #     if is_replanning_action:
            #         # print(controller_env.last_event.metadata["lastAction"])
            #         # print("PutObject" in [controller_env.last_event.metadata["lastAction"]])
            #         if "PutObject" in [controller_env.last_event.metadata["lastAction"]]:
            #             replanning_action = "Open the "
            #             replanning = True

            object_name_list = controller_env.seen_object_name()
            print("seen object name: ", object_name_list)
            # 生成第一段规划
            new_object_list = controller_env.object_dict2str_llama_s1_only()
            # 生成第一段input
            human_instruction_input = "Instruction: " + instruction + "\n" + "Object List: " + new_object_list + \
                                      "\n" + "Trying Times: " + str(try_times) + "\n" + "Done Actions: " + done_list.rstrip("\n")

            s1_input = {"instruction": instruction_prompt, "input": human_instruction_input}
            s1_response = planner.get_llama_response(s1_input)
            # print("s1_response: ", s1_response)
            # try:
            #     current_action = planner.parse_llama_s1_response(s1_response)
            #     if replanning:
            #         # current_action = current_action.split(".")[-1] + " " + replanning_action + target
            #         current_action = "Step " + str(done_action_number + 1) + ". " + replanning_action + target
            # except:
            #     continue
            current_action = response_action_dict_one["planning"]
            action = response_action_dict_one["action"]
            target = response_action_dict_one["arg"]
            print("current_action: ", current_action)
            # 生成第二阶段规划
            try:
                s2_input = planner.trans2_llava_input(current_action, frontiers_label_dict_list, robot_position=sem_map.robot_heightmap_point)
                s2_response = planner.get_llava_response(s2_input, select_feature_list)
            except:
                continue
            # if len(frontiers_label_dict_list) < 1:
            #     s2_response = s2_response + "\n" + "There is no frontiers, but still can not find object"
            # 解析具体动作
            nav_point, _, _ = planner.parse_llava_s2_response(s2_response)
            # _, action, target = planner.parse_llava_s2_response(s2_response)
            print("target: ", target)
            if "end" in action.lower():
                success = True
                continue
            # 根据target生成候选
            candidate_position_list, candidate_target_obj_id_list = controller_env.get_candidate_by_name(target, action_name=action)
            if len(candidate_target_obj_id_list) < 1:
                planning_fail = planning_fail + 1
            candidate_index_list = controller_env.get_candidate_id_by_object_id(candidate_target_obj_id_list, used_object_id=used_object_id_list)
            # 直接使用GT, 不进行巡航规划
            # candidate_index_list = range(0, len(candidate_position_list))
            if len(candidate_index_list) > 0:
                # 视野中有物体, 则直接选取s1规划
                print("find object, start s1")
                # select_candidate_index = candidate_index_list[0]
                # 加入use_id_list
                select_candidate_index = candidate_index_list[0]
                # for candidate_index in candidate_index_list:
                #     if candidate_index not in used_object_id_list:
                #         select_candidate_index = candidate_index
                target_nav_position = candidate_position_list[select_candidate_index]
                target_object_id = candidate_target_obj_id_list[select_candidate_index]
            else:
                # --------------------------------
                # random
                # if len(frontiers_label_dict_list) >= 1:
                #     frontiers_label_dict_one = random.choice(frontiers_label_dict_list)
                #     nav_point = frontiers_label_dict_one["centroid"]
                # else:
                #     nav_point = [0, 0, 0]
                # --------------------------------
                target_nav_position = sem_map.pixel2world_point(nav_point)
                target_object_id = None
            if target_object_id is not None:
                step_action_dict_one = dict(
                    nav_position=target_nav_position,
                    action=action,
                    target=target_object_id.split("|")[0],
                    target_id=target_object_id,
                    s1_input=s1_input,
                    s1_response=s1_response,
                    s2_input=s2_input,
                    s2_response=s2_response
                )
            else:
                step_action_dict_one = dict(
                    nav_position=target_nav_position,
                    action=action,
                    target=target,
                    target_id=target_object_id,
                    s1_input=s1_input,
                    s1_response=s1_response,
                    s2_input=s2_input,
                    s2_response=s2_response
                )

            if len(frontiers_label_dict_list) < 1:
                if target_object_id is None:
                    print("There is no frontiers, but still can not find object")
                    break

            # # 提供gt
            # target_nav_position = candidate_position_list[0]
            # target_object_id = candidate_target_obj_id_list[0]

            # 执行具体巡航动作
            if "goto" in action.lower() or target_object_id is None:
                perspective_camera_view(controller_env)
                print("target_nav_position: ", target_nav_position)
                nav_action, path = controller_env.parse_nav_action(target_nav_position)
                path_distance = (len(path) - 1) * 0.25
                for nav_action_id, action_one in enumerate(tqdm(nav_action)):
                    rgb, depth_frame, mask_list, info_dict = controller_env.to_thor_api_exec([action_one])
                    # if nav_action_id % 5 == 0 or nav_action_id in [len(nav_action)-2, len(nav_action)-1]:
                    if nav_action_id % 5 == 0:
                        _, _ = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
                # path_distance = controller_env.goto_location_nav(target_nav_position, sem_map)
            else:
                path_distance = 0
                # add 距离限制
                is_near = controller_env.check_is_near_target(target_object_id)
                if not is_near:
                    path_distance = controller_env.goto_location_nav(target_nav_position, sem_map)
                # 环视寻找物体
                is_find = controller_env.check_target_frame(target_object_id)
                if target_object_id not in controller_env.seen_object_dict_all.keys():
                    continue
                # 有可能是A*算法出问题
                # 在测试的时候, 先不适用
                # if not is_find:
                #     new_nav_point = sem_map.get_sem_map_object_position(target_object_id)
                #     if new_nav_point is not None:
                #         add_path_distance = controller_env.goto_location_nav(new_nav_point, sem_map)
                #         path_distance = path_distance + add_path_distance
                #         step_action_dict_one["nav_position"] = new_nav_point

                if "PutObject" in [action]:
                    hand_object_list = controller_env.last_event.metadata['inventoryObjects']
                    if len(hand_object_list) > 0:
                        hand_object = hand_object_list[0]["objectId"]
                        # print(hand_object_list[0].keys())
                        isInteractable = next(obj["isInteractable"] for obj in controller_env.last_event.metadata["objects"]
                           if obj["objectId"] == hand_object)
                        if not isInteractable:
                            rgb, depth_frame, mask_list, info_dict = controller_env.to_thor_api_exec(["EnableObject"], object_id=hand_object)
                            info_dict["interactive_object"] = hand_object
                            _, _ = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
                            # print(hand_object_list[0]["isInteractable"])

                rgb, depth_frame, mask_list, info_dict = controller_env.execAction(action=[
                    action], target_arg=target_object_id.split("|")[0],
                    target_position=target_nav_position, target_object_id=target_object_id)
                # print(controller_env.last_event)
                if action in ["OpenObject", "CloseObject", "PickupObject", "PutObject"]:
                    info_dict["interactive_object"] = target_object_id
                _, _ = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")

                if "PickupObject" in [action]:
                    isInteractable = next(obj["isInteractable"] for obj in controller_env.last_event.metadata["objects"]
                       if obj["objectId"] == target_object_id)
                    if isInteractable:
                        rgb, depth_frame, mask_list, info_dict = controller_env.to_thor_api_exec(["DisableObject"], object_id=target_object_id)
                        info_dict["interactive_object"] = target_object_id
                        # _, _ = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")

            # 统计路径长度
            path_distance_all = path_distance_all + path_distance
            if "goto" in action.lower():
                candidate_index_list = controller_env.get_candidate_id_by_object_id(candidate_target_obj_id_list, used_object_id=used_object_id_list)
                if len(candidate_index_list) > 0:
                    done_list = done_list + current_action + "\n"
                    is_replanning_action = False
                    is_replanning_target = False
                    used_object_id_list = []
                    done_action_number += 1
                    response_action_list_index += 1
            else:
                if controller_env.last_event.metadata["lastActionSuccess"]:
                    done_list = done_list + current_action + "\n"
                    is_replanning_action = False
                    is_replanning_target = False
                    used_object_id_list = []
                    done_action_number += 1
                    response_action_list_index += 1
                else:
                    if len(candidate_target_obj_id_list) <= 1:
                        is_replanning_action = True
                        is_replanning_target = False
                    else:
                        is_replanning_action = True
                        if try_times > 5:
                            is_replanning_target = True
                            used_object_id_list.append(target_object_id)

            # 记录信息
            result_dict_one["step_action_list"].append(step_action_dict_one)

        instruction_success = controller_env.check_task(task_type=task_type, task_label_dict=pddl_params)
        if instruction_success:
            success_number = success_number + 1
            success_index_list.append(val_dict_id)
        print("now success:", success_number / (val_dict_id + 1))
        result_dict_one["instruction"] = instruction
        result_dict_one["done_action"] = done_list
        result_dict_one["success"] = instruction_success
        result_dict_one["house_id"] = house_id
        result_dict_one["distance"] = path_distance_all
        reslut_list.append(result_dict_one)
        path_distance_list.append(path_distance_all)

    # 计算衡量指标
    print("测试指令数量", len(path_distance_list))
    print("成功数量", success_number)
    print("成功率:", success_number / len(path_distance_list))
    print("平均路径长度:", np.sum(np.asarray(path_distance_list)) / len(path_distance_list))
    print("完成测试--------------------------------------")
    print(success_index_list)

    # 保存结果
    # with open("llama_vln_parsed_response_v8_val_result.json", 'w') as f:
    #     json_str = json.dumps(reslut_list, indent=2)
    #     f.write(json_str)
    #     f.write('\n')
    with open("llava_vln_parsed_response_v8_val_result_v4_nav_gt_150_down.pkl", 'wb') as f:
        pickle.dump(reslut_list, f)

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
    args = parser.parse_args()
    main(args)