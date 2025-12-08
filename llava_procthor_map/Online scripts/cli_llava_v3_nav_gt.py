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
from utils.procthor_config import Config as proc_Config
import open_clip

from utils.generate_response_llava import Planner
from utils.sem_map import Semantic_Mapping
from utils.thor_env_code_v2 import ThorEnvCode
import prior
import pickle
import time
import gzip
from prior import LazyJsonDataset


# 加载具体场景
def load_dataset() -> prior.DatasetDict:
    """Load the houses dataset."""
    print(
        "[AI2-THOR WARNING] There has been an update to ProcTHOR-10K that must be used with AI2-THOR version 5.0+. To use the new version of ProcTHOR-10K, please update AI2-THOR to version 5.0+ by running:\n"
        "    pip install --upgrade ai2thor\n"
        "Alternatively, to downgrade to the old version of ProcTHOR-10K, run:\n"
        '   prior.load_dataset("procthor-10k", revision="ab3cacd0fc17754d4c080a3fd50b18395fae8647")'
    )
    data = {}
    for split, size in [("train", 10_000), ("val", 1_000), ("test", 1_000)]:
        with gzip.open(f"./procthor_house/{split}.jsonl.gz", "r") as f:
            houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
        data[split] = LazyJsonDataset(
            data=houses, dataset="procthor-dataset", split=split
        )
    return prior.DatasetDict(**data)


def main(args):
    # Model
    disable_torch_init()
    # load llava
    model_name = get_model_name_from_path(args.model_path)
    llava_tokenizer, llava_model_s2, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    # load llava_s1
    model_name_s1 = get_model_name_from_path(args.model_path_s1)
    _, llava_model_s1, _, _ = load_pretrained_model(args.model_path_s1, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device_s1)

    # load clip for semantic feature map
    open_clip_model = "ViT-H-14"
    torch.autograd.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Initializing model...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        open_clip_model, pretrained="./checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
    )
    clip_model.cuda("cuda:2")
    clip_model.eval()

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

    # 加载配置文件
    total_cat2idx_path = "./utils/total_cat2idx.json"
    total_cat2idx = json.load(open(total_cat2idx_path))
    # 增加wall
    total_cat2idx["wall"] = 94
    total_cat2idx["floor"] = 95

    planner = Planner(
        llava_model_s2=llava_model_s2,
        llava_args=args,
        llava_tokenizer=llava_tokenizer,
        llava_model_s1=llava_model_s1
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
    # dataset = prior.load_dataset("procthor-10k")
    dataset = load_dataset()
    dataset_train = dataset["train"]
    controller_env = ThorEnvCode(args, use_CloudRendering=True)

    # 初始化衡量指标
    success_number = 0
    path_distance_list = []
    success_index_list = []
    nav_fail = 0
    planning_fail = 0
    exec_fail = 0

    # 初始化保存路径
    base_save_path = "./log_file/llava_s1_s2_vln_parsed_response_v8_val_result_nav_gt_v3"
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)

    # 增加val数据集过滤
    textworld_parse_result = json.load(open("./log_file/parsed_spaced_parse_instruction_easy_v8_val.json"))
    new_instruction_dict_list = []
    for scene_index, textworld_parse_result_one in enumerate(textworld_parse_result):
        instruction_dict_list_one = instruction_dict_list[scene_index]
        instruction_house_id = instruction_dict_list_one["house_id"]
        textworld_house_id = textworld_parse_result_one["house_id"]
        assert str(instruction_house_id) == str(textworld_house_id)
        success = textworld_parse_result_one["instruction_success"]
        if success:
            new_instruction_dict_list.append(instruction_dict_list_one)

    print("过滤后数据集长度: ", len(new_instruction_dict_list))

    # -------------------main loop--------------------------
    for val_dict_id, val_dict_one in enumerate(tqdm(new_instruction_dict_list)):
        instruction = val_dict_one["instruction"].strip(" ").strip("\"")
        house_id = int(val_dict_one["house_id"])
        print(house_id)
        print(instruction)
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

        # 首先进行环视
        sem_map.reset()
        for _ in range(4):
            # 增加转圈
            rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
            info_dict["add_robot_mask"] = True
            info_dict["region_size"] = 0.55
            global_sem_map, global_sem_feature_map = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
            event = controller_env.step(action="RotateRight", forceAction=True)

        # 用来已经交互, 但失败的物体
        used_object_id_list = []
        result_dict_one = {}
        result_dict_one["step_action_list"] = []
        is_replanning_action = False
        is_replanning_target = False

        while not success:
            if instruction_test_time >= 1:
                # 在新的规划中, 开始进行环视
                for _ in range(4):
                    # 增加转圈
                    rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
                    info_dict["add_robot_mask"] = True
                    info_dict["region_size"] = 0.20
                    global_sem_map, global_sem_feature_map = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
                    event = controller_env.step(action="RotateRight", forceAction=True)
                print("end surrounding ----------------------")
            # 开始生成边界
            rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
            select_feature_list, frontiers_label_dict_list = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="frontiers")
            # 查看是否有过小的边界
            need_surround = controller_env.check_for_down_surround(frontiers_label_dict_list, sem_map)
            if need_surround:
                # 执行低头环视策略
                print("need look down surround")
                controller_env.down_surround(sem_map)
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
            if try_times > 1:
                if is_replanning_action:
                    # print(controller_env.last_event.metadata["lastAction"])
                    # print("PutObject" in [controller_env.last_event.metadata["lastAction"]])
                    if "PutObject" in [controller_env.last_event.metadata["lastAction"]]:
                        replanning_action = "Open the "
                        replanning = True

            # ---------------生成具体规划-------------------
            object_name_list = controller_env.seen_object_name()
            print("seen object name: ", object_name_list)
            # 生成第一阶段规划
            new_object_list = controller_env.object_dict2str_llama_s1_only()
            # 生成第一段input
            s1_input = "Instruction: " + instruction + "\n" + "Object List: " + new_object_list + \
                                      "\n" + "Done Actions: " + done_list.rstrip("\n")
            s1_response = planner.get_llava_response_s1(input=s1_input, feature=np.zeros([256, 1024], dtype=np.float16))
            try:
                current_action = planner.parse_llava_s1_response(s1_response)
                if replanning:
                    # current_action = current_action.split(".")[-1] + " " + replanning_action + target
                    current_action = "Step " + str(done_action_number + 1) + ". " + replanning_action + target
            except:
                continue
            # 生成第二阶段规划
            try:
                s2_input = planner.trans2_llava_input(current_action, frontiers_label_dict_list, robot_position=sem_map.robot_heightmap_point)
                s2_response = planner.get_llava_response_s2(s2_input, select_feature_list)
            except:
                continue
            # if len(frontiers_label_dict_list) < 1:
            #     s2_response = s2_response + "\n" + "There is no frontiers, but still can not find object"
            # 解析具体动作
            nav_point, action, target = planner.parse_llava_s2_response(s2_response)
            print("target: ", target)

            if "end" in action.lower():
                success = True
                done_list = done_list + current_action + "\n"
                step_action_dict_one = dict(
                    nav_position=None,
                    action=action,
                    target=None,
                    target_id=None,
                    s1_input=s1_input,
                    s1_response=s1_response,
                    s2_input=s2_input,
                    s2_response=s2_response
                )
                result_dict_one["step_action_list"].append(step_action_dict_one)
                continue

            # 根据target生成候选
            candidate_position_list, candidate_target_obj_id_list = controller_env.get_candidate_by_name(target, action_name=action)
            if len(candidate_target_obj_id_list) < 1:
                planning_fail = planning_fail + 1
            # candidate_index_list = controller_env.get_candidate_id_by_object_id(candidate_target_obj_id_list, used_object_id=used_object_id_list)
            # 直接使用GT, 不进行巡航规划
            candidate_index_list = range(0, len(candidate_position_list))
            if len(candidate_index_list) > 0:
                # 视野中有物体, 则直接选取s1规划
                print("find object, start s1")
                select_candidate_index = candidate_index_list[0]
                for candidate_index in candidate_index_list:
                    if candidate_index not in used_object_id_list:
                        select_candidate_index = candidate_index
                        break
                target_nav_position = candidate_position_list[select_candidate_index]
                target_object_id = candidate_target_obj_id_list[select_candidate_index]
            else:
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

            # if len(frontiers_label_dict_list) < 1:
            #     if target_object_id is None:
            #         print("There is no frontiers, but still can not find object")
            #         break

            # -------------------执行具体动作-----------------------
            if "goto" in action.lower() or target_object_id is None:
                controller_env.perspective_camera_view()
                print("target_nav_position: ", target_nav_position)
                nav_action, path = controller_env.parse_nav_action(target_nav_position)
                path_distance = (len(path) - 1) * 0.25
                for nav_action_id, action_one in enumerate(tqdm(nav_action)):
                    rgb, depth_frame, mask_list, info_dict = controller_env.to_thor_api_exec([action_one])
                    if nav_action_id % 5 == 0 or nav_action_id in [len(nav_action)-1]:
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
                if "PutObject" in [action]:
                    hand_object_list = controller_env.last_event.metadata['inventoryObjects']
                    if len(hand_object_list) > 0:
                        hand_object = hand_object_list[0]["objectId"]
                        # print(hand_object_list[0].keys())
                        isInteractable = next(
                            obj["isInteractable"] for obj in controller_env.last_event.metadata["objects"]
                            if obj["objectId"] == hand_object)
                        if not isInteractable:
                            rgb, depth_frame, mask_list, info_dict = controller_env.to_thor_api_exec([
                                "EnableObject"], object_id=hand_object)
                            info_dict["interactive_object"] = hand_object
                            # _, _ = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")

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
                # candidate_index_list = controller_env.get_candidate_id_by_object_id(candidate_target_obj_id_list, used_object_id=used_object_id_list)
                # if len(candidate_index_list) > 0:
                done_list = done_list + current_action + "\n"
                is_replanning_action = False
                is_replanning_target = False
                used_object_id_list = []
                done_action_number += 1
            else:
                if controller_env.last_event.metadata["lastActionSuccess"]:
                    done_list = done_list + current_action + "\n"
                    is_replanning_action = False
                    is_replanning_target = False
                    used_object_id_list = []
                    done_action_number += 1
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

        # check指令是否成功完成
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

        # 每个进行单独保存
        save_name_one = "house_" + str(house_id) + ".pkl"
        save_path_one = os.path.join(base_save_path, save_name_one)
        with open(save_path_one, 'wb') as f:
            pickle.dump(result_dict_one, f)

    # 计算衡量指标
    print("测试指令数量", len(path_distance_list))
    print("成功数量", success_number)
    print("成功率:", success_number / len(path_distance_list))
    print("平均路径长度:", np.sum(np.asarray(path_distance_list)) / len(path_distance_list))
    print("完成测试--------------------------------------")
    print(success_index_list)

    # 保存最终结果
    with open("llava_s1_s2_vln_parsed_response_v8_val_result_nav_gt_v3.pkl", 'wb') as f:
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
    # add llava_s1 planner args
    parser.add_argument("--model-path-s1", type=str, default="facebook/opt-350m")
    parser.add_argument("--device-s1", type=str, default="cuda:1")
    args = parser.parse_args()
    main(args)