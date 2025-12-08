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
from utils.done_action import DoneAction
from utils.interactive_script import InteractiveScript


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
    # model_name_s1 = get_model_name_from_path(args.model_path_s1)
    # _, llava_model_s1, _, _ = load_pretrained_model(args.model_path_s1, args.model_base, model_name_s1, args.load_8bit, args.load_4bit, device=args.device_s1)

    # load clip for semantic feature map
    open_clip_model = "ViT-H-14"
    torch.autograd.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Initializing model...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        open_clip_model, pretrained="./checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
    )
    # clip_model.cuda("cuda:1")
    clip_model.cuda("cuda:" + str(proc_Config.open_clip_gpu))
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
    # total_cat2idx_path = "./utils/total_cat2idx.json"
    total_cat2idx_path = proc_Config.total_cat2idx_procthor_path
    total_cat2idx = json.load(open(total_cat2idx_path))

    planner = Planner(
        llava_model_s2=llava_model_s2,
        llava_args=args,
        llava_tokenizer=llava_tokenizer,
        llava_model_s1=None
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

    # 解析object name
    house_id2object_list_dict = {}
    for space_dataset_one in instruction_dict_list:
        object_list_one = []
        output_one = space_dataset_one["output"]
        house_id_one = int(space_dataset_one["house_id"])
        for action_dict_one in output_one:
            arg = action_dict_one["arg"]
            if arg != " ":
                object_list_one.append(arg)
        object_list_one = list(set(object_list_one))
        house_id2object_list_dict[house_id_one] = object_list_one

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
    base_save_path = "./log_file/llava_s1_s2_vln_parsed_response_planning_gt_detic_easy_large_0730"
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)

    # -------------------main loop--------------------------
    for val_dict_id, val_dict_one in enumerate(tqdm(instruction_dict_list)):
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

        # parse object name list
        planning_object_name_list = house_id2object_list_dict[int(house_id)]

        interactive_tool = InteractiveScript(
            agent=controller_env,
            sem_map=sem_map
        )
        interactive_tool.reset_distance()

        done_action = DoneAction(
            agent=controller_env
        )
        done_action.reset()

        # 首先进行环视
        sem_map.reset()
        long_clip_prompt = planner.parse_long_clip_prompt(planning_object_name_list)
        print(long_clip_prompt)
        controller_env.perspective_camera_view(reset_view_angle=30)
        for _ in range(4):
            # 增加转圈
            rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
            info_dict["text_prompt"] = long_clip_prompt
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

        response_action_list = val_dict_one["output"]
        response_action_list_index = 0

        while not success:
            if instruction_test_time >= 1:
                controller_env.perspective_camera_view(reset_view_angle=30)
                # 在新的规划中, 开始进行环视
                for _ in range(4):
                    # 增加转圈
                    rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
                    info_dict["add_robot_mask"] = True
                    info_dict["region_size"] = 0.30
                    global_sem_map, global_sem_feature_map = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
                    event = controller_env.step(action="RotateRight", forceAction=True)
                print("end surrounding ----------------------")
            # 开始生成边界
            rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
            select_feature_list, frontiers_label_dict_list = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="frontiers")
            # 查看是否有过小的边界
            # need_surround = controller_env.check_for_down_surround(frontiers_label_dict_list, sem_map)
            # if need_surround:
            #     # 执行低头环视策略
            #     print("need look down surround")
            #     controller_env.down_surround(sem_map)
            #     rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
            #     select_feature_list, frontiers_label_dict_list = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="frontiers")

            instruction_test_time = instruction_test_time + 1
            select_s2 = False
            has_check_recv = False
            if instruction_test_time > 30:
                success = False
                break
            # 统计尝试次数
            done_list_str = done_action.get_done_low_action_str()
            if done_list_str in done_action_dict.keys():
                done_action_dict[done_list_str] = done_action_dict[done_list_str] + 1
            else:
                done_action_dict[done_list_str] = 0
            try_times = done_action_dict[done_list_str]
            replanning = False
            if try_times > 10:
                break

            # putobject失败原因很多, 无法直接使用replanning
            if try_times > 1:
                if is_replanning_action:
                    # print(controller_env.last_event.metadata["lastAction"])
                    # print("PutObject" in [controller_env.last_event.metadata["lastAction"]])
                    if "PutObject" in [controller_env.last_event.metadata["lastAction"]]:
                        replanning_action = "Open the "
                        replanning = True

            # ---------------生成具体规划-------------------
            # object_name_list = controller_env.seen_object_name_alfred()
            # 从sem_map种获取物体, 避免出现找到物体但没有target
            object_name_list = sem_map.get_object_name_list_from_map()
            print("seen object name: ", object_name_list)
            # 生成第一阶段规划
            new_object_list = controller_env.object_dict2str_llama_s1_only_alfred(object_name_list)
            # 生成第一段input
            done_list_str = done_action.get_done_action_str()
            s1_input = "Instruction: " + instruction + "\n" + "Object List: " + new_object_list + \
                                      "\n" + "Done Actions: " + done_list_str.rstrip("\n")
            # s1_response = planner.get_llava_response_s1(input=s1_input, feature=np.zeros([256, 1024], dtype=np.float16))

            # try:
            #     current_action = planner.parse_llava_s1_response(s1_response)
            #     if replanning:
            #         # current_action = current_action.split(".")[-1] + " " + replanning_action + target
            #         done_action_number = len(done_action.done_action_list)
            #         current_action = "Step " + str(done_action_number + 1) + ". " + replanning_action + target
            # except:
            #     continue
            response_action_dict_one = response_action_list[response_action_list_index]
            current_action = response_action_dict_one["planning"]
            s1_response = current_action
            action = response_action_dict_one["action"]
            target = response_action_dict_one["arg"]
            print("current_action: ", current_action)
            # 生成第二阶段规划
            # try:
            s2_input = planner.trans2_llava_input(current_action, frontiers_label_dict_list, robot_position=sem_map.robot_heightmap_point)
            s2_response = planner.get_llava_response_s2(s2_input, select_feature_list)
            # except:
            #     continue
            # 解析具体动作
            nav_point, _, _ = planner.parse_llava_s2_response(s2_response)
            print("target: ", target)
            if "end" in action.lower():
                success = True
                continue
            if target in object_name_list:
                # 视野中有物体, 则直接选取s1规划
                print("find object, start s1")
                # 直接获取region中最大的候选
                target_nav_position = sem_map.get_sem_map_object_position(target)
            else:
                target_nav_position = sem_map.pixel2world_point(nav_point)
                target_object_id = None
            # 取消保存id
            step_action_dict_one = dict(
                nav_position=target_nav_position,
                action=action,
                target=target,
                s1_input=s1_input,
                s1_response=s1_response,
                s2_input=s2_input,
                s2_response=s2_response
            )

            if len(frontiers_label_dict_list) < 1:
                if target not in object_name_list:
                    print("There is no frontiers, but still can not find object")
                    # 记录信息
                    result_dict_one["step_action_list"].append(step_action_dict_one)
                    break

            action_execution = False
            target_object_id = None
            use_history_nav = False
            use_history_inter = False
            # -------------------执行具体动作-----------------------
            if "goto" in action.lower() or target not in object_name_list:
                # 抬头走路
                controller_env.perspective_camera_view()
                print("target_nav_position: ", target_nav_position)
                if target_nav_position is None:
                    continue
                nav_action, path = controller_env.parse_nav_action(target_nav_position)
                path_distance = (len(path) - 1) * 0.25
                for nav_action_id, action_one in enumerate(tqdm(nav_action)):
                    rgb, depth_frame, mask_list, info_dict = controller_env.to_thor_api_exec([action_one])
                    info_dict["text_prompt"] = long_clip_prompt
                    if nav_action_id % 5 == 0:
                        _, _ = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update", only_seg=False)
                    else:
                        _, _ = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update", only_seg=True)
                # path_distance = controller_env.goto_location_nav(target_nav_position, sem_map)
                # 增加物体同类遍历
                is_find = controller_env.check_target_frame_alfred(target)
                # 如果没有找到, 启动记录(针对巡航)
                if not is_find:
                    check_action_dict = {"target_name": target,
                                         "low_action": action,
                                         "action_type": "nav"}
                    response_dict = done_action.check_previous_pose(check_action_dict)
                    if response_dict["success"]:
                        use_history_nav = True
                step_action_dict_one["action_response"] = controller_env.last_event.metadata["errorMessage"]
            else:
                path_distance = 0
                # 环视寻找物体
                # 增加物体同类遍历
                is_find = controller_env.check_target_frame_alfred(target)

                # 先check是否在附近
                if not is_find:
                    input_dict = {"metadata": controller_env.last_event.metadata,
                                  "target_name": target}
                    response_dict = sem_map.get_object_id_from_sem_map(input_dict)
                    if response_dict["success"]:
                        is_find = True
                        target_object_id = response_dict["target_id"]

                # 先启动记录
                if not is_find:
                    if action in ["OpenObject", "CloseObject", "PutObject", "ToggleObjectOn", "ToggleObjectOff"]:
                        target_object_id = None
                        check_action_dict = {"target_name": target,
                                             "low_action": action,
                                             "action_type": "inter"}
                        response_dict = done_action.check_previous_pose(check_action_dict)
                        if response_dict["success"]:
                            use_history_inter = True
                            is_find = True
                            target_object_id = response_dict["object_id"]

                if not is_find:
                    if "openobject" in done_action.done_low_level_action_list[-1].lower() and\
                            "PutObject" in done_action.done_low_level_action_list:
                        is_find = interactive_tool.nav_surround_for_target({"target": target})
                        path_distance = path_distance + interactive_tool.distance
                        has_check_recv = True
                    else:
                        print("not find, move other position")
                        nav_point_list = sem_map.get_sem_map_object_position_list(target)
                        for nav_point_one in nav_point_list[:3]:
                            target_nav_position_one = sem_map.pixel2world_point(nav_point_one)
                            distance_one = controller_env.goto_location_nav(target_nav_position_one)
                            path_distance = path_distance + distance_one
                            is_find = controller_env.check_target_frame_alfred(target)
                            # 每到一个地点就需要重新check
                            if not is_find:
                                input_dict = {"metadata": controller_env.last_event.metadata,
                                              "target_name": target}
                                response_dict = sem_map.get_object_id_from_sem_map(input_dict)
                                if response_dict["success"]:
                                    is_find = True
                                    target_object_id = response_dict["target_id"]
                            # 每次都要去掉
                            if not is_find:
                                robot_position = np.asarray(list(
                                    controller_env.last_event.metadata["agent"]["position"].values()))
                                sem_map.remove_unseen_target_map(object_name=target, robot_position=robot_position)
                            if is_find:
                                break
                print("is find target:", is_find)
                # 如果没有找到物体, 不允许进行交互
                if not is_find:
                    # 没有找到物体, 需要更新object map
                    # robot_position = np.asarray(list(controller_env.last_event.metadata["agent"]["position"].values()))
                    # sem_map.remove_unseen_target_map(object_name=target, robot_position=robot_position)
                    # 同时, 上一步done action需要进行剔除
                    done_action.reflective_action()
                    response_action_list_index -= 1
                    continue
                # 如果找到物体, 保存信息, 在这里以什么情况可以找到物体
                object_info_dict = {"low_action": action,
                                    "target_name": target}
                done_action.add_nav_object_position(object_info_dict)

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

                    # 同时修正sink name
                    if "sink" in target.lower():
                        target = "SinkBasin"

                rgb, depth_frame, mask_list, info_dict = controller_env.execAction(action=[
                    action], target_arg=target, target_position=target_nav_position, target_object_id=target_object_id)
                print(controller_env.last_event)
                action_execution = controller_env.last_event.metadata["lastActionSuccess"]
                step_action_dict_one["action_response"] = controller_env.last_event.metadata["errorMessage"]
                # 增加后处理
                if not action_execution:
                    # if "PutObject" in [action]:
                    if target in ["Microwave"]:
                        if action in ["ToggleObjectOn"]:
                            step_action_dict_one["add_action"] = "CloseObject"
                            _, _, _, _ = controller_env.to_thor_api_exec([
                                "CloseObject"], object_id=controller_env.interactive_object_id)
                            result_dict_one["step_action_list"].append(step_action_dict_one)
                        elif action in ["OpenObject"]:
                            step_action_dict_one["add_action"] = "ToggleObjectOff"
                            _, _, _, _ = controller_env.to_thor_api_exec([
                                "ToggleObjectOff"], object_id=controller_env.interactive_object_id)
                            result_dict_one["step_action_list"].append(step_action_dict_one)
                        elif action in ["PickupObject"]:
                            step_action_dict_one["add_action"] = "OpenObject"
                            _, _, _, _ = controller_env.to_thor_api_exec([
                                "OpenObject"], object_id=controller_env.interactive_object_id)
                            result_dict_one["step_action_list"].append(step_action_dict_one)
                        elif action in ["PutObject"]:
                            step_action_dict_one["add_action"] = "OpenObject"
                            _, _, _, _ = controller_env.to_thor_api_exec([
                                "OpenObject"], object_id=controller_env.interactive_object_id)
                            result_dict_one["step_action_list"].append(step_action_dict_one)
                    elif target in ["Fridge"]:
                        if action in ["PutObject"]:
                            step_action_dict_one["add_action"] = "OpenObject"
                            _, _, _, _ = controller_env.to_thor_api_exec([
                                "OpenObject"], object_id=controller_env.interactive_object_id)
                            result_dict_one["step_action_list"].append(step_action_dict_one)
                        elif action in ["PickupObject"]:
                            step_action_dict_one["add_action"] = "OpenObject"
                            _, _, _, _ = controller_env.to_thor_api_exec([
                                "OpenObject"], object_id=controller_env.interactive_object_id)
                            result_dict_one["step_action_list"].append(step_action_dict_one)
                    elif "PutObject" in [action]:
                        if try_times > 2:
                            if len(controller_env.last_event.metadata['inventoryObjects']) > 0:
                                event = controller_env.step(
                                    action="MoveHeldObjectAhead",
                                    moveMagnitude=0.2,
                                    forceVisible=True
                                )
                                drop_object_acion = dict(action="DropHandObject", forceAction=True)
                                event = controller_env.step(drop_object_acion)
                                result_dict_one["step_action_list"].append(step_action_dict_one)
                                # 去执行下一个目标
                                action_execution = True
                        else:
                            step_action_dict_one["add_action"] = "OpenObject"
                            _, _, _, _ = controller_env.to_thor_api_exec([
                                "OpenObject"], object_id=controller_env.interactive_object_id)
                            result_dict_one["step_action_list"].append(step_action_dict_one)
                    elif len(controller_env.last_event.metadata['inventoryObjects']) > 0:
                        if action in ["PickupObject"]:
                            event = controller_env.step(
                                action="MoveHeldObjectAhead",
                                moveMagnitude=0.2,
                                forceVisible=True
                            )
                            drop_object_acion = dict(action="DropHandObject", forceAction=True)
                            event = controller_env.step(drop_object_acion)
                            result_dict_one["step_action_list"].append(step_action_dict_one)
                    else:
                        pass

                # 重新规划可以移动的空间
                controller_env.reset_arrival_position()

                if "PutObject" in [action]:
                    _, _ = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update", only_seg=True)

                if "ToggleObjectOn" in [action] or "ToggleObjectOff" in [action]:
                    if not action_execution:
                        if "already" in controller_env.last_event.metadata["errorMessage"]:
                            action_execution = True

                if "PutObject" in [action]:
                    if not action_execution:
                        # 如果没有执行成功, 需要重新消除物体
                        hand_object_list = controller_env.last_event.metadata['inventoryObjects']
                        if len(hand_object_list) > 0:
                            hand_object = hand_object_list[0]["objectId"]
                            isInteractable = next(
                                obj["isInteractable"] for obj in controller_env.last_event.metadata["objects"]
                                if obj["objectId"] == hand_object)
                            if isInteractable:
                                rgb, depth_frame, mask_list, info_dict = controller_env.to_thor_api_exec([
                                    "DisableObject"], object_id=hand_object)
                                info_dict["interactive_object"] = hand_object
                        # robot_position = np.asarray(list(
                        #     controller_env.last_event.metadata["agent"]["position"].values()))
                        # sem_map.remove_unseen_target_map(object_name=target, robot_position=robot_position, remove_type="pickup")

                if action in ["OpenObject", "CloseObject", "PickupObject", "PutObject"]:
                    hand_object_list = controller_env.last_event.metadata['inventoryObjects']
                    if len(hand_object_list) > 0:
                        info_dict["interactive_object"] = hand_object_list[0]["objectId"]
                _, _ = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update", only_seg=True)

                if action == "OpenObject" and target == "Fridge":
                    # agent后退两步, 从而规避碰撞
                    for _ in range(2):
                        move_back_action = dict(action="MoveBack", forceAction=True)
                        controller_env.step(move_back_action)

                # open Fridge后, 需要进行环视感知物体
                if action == "OpenObject" and target == "Fridge":
                    interactive_tool.nav_surround()

                if "PickupObject" in [action]:
                    hand_object_list = controller_env.last_event.metadata['inventoryObjects']
                    if len(hand_object_list) > 0:
                        hand_object = hand_object_list[0]["objectId"]
                        isInteractable = next(obj["isInteractable"] for obj in controller_env.last_event.metadata["objects"]
                           if obj["objectId"] == hand_object)
                        if isInteractable:
                            rgb, depth_frame, mask_list, info_dict = controller_env.to_thor_api_exec(["DisableObject"], object_id=hand_object)
                            info_dict["interactive_object"] = hand_object
                        # _, _ = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
                    # 同时更新消除抓取物体
                    robot_position = np.asarray(list(controller_env.last_event.metadata["agent"]["position"].values()))
                    sem_map.remove_unseen_target_map(object_name=target, robot_position=robot_position, remove_type="pickup")

            # 记录信息
            result_dict_one["step_action_list"].append(step_action_dict_one)
            # 统计路径长度
            path_distance_all = path_distance_all + path_distance
            if "goto" in action.lower():
                # candidate_index_list = controller_env.get_candidate_id_by_object_id(candidate_target_obj_id_list, used_object_id=used_object_id_list)
                # if target in controller_env.seen_object_name_list_all:
                object_name_list = sem_map.get_object_name_list_from_map()
                if target in object_name_list:
                    done_action.add_done_action(current_action)
                    done_action.add_low_level_action(action, target)
                    done_list = done_list + current_action + "\n"
                    is_replanning_action = False
                    is_replanning_target = False
                    used_object_id_list = []
                    done_action_number += 1

                    response_action_list_index += 1
            else:
                # if controller_env.last_event.metadata["lastActionSuccess"]:
                if action_execution:
                    done_list = done_list + current_action + "\n"
                    done_action.add_done_action(current_action)
                    done_action.add_low_level_action(action, target)
                    is_replanning_action = False
                    is_replanning_target = False
                    used_object_id_list = []
                    done_action_number += 1
                    # save_info
                    # 如果找到物体, 保存信息, 在这里以什么情况可以找到物体
                    object_info_dict = {"low_action": action,
                                        "target_name": target,
                                        "target_object_id": controller_env.interactive_object_id}
                    done_action.add_interactive_object(object_info_dict)

                    response_action_list_index += 1

                else:
                    is_replanning_action = True
                    # if try_times > 5:
                    #     used_object_id_list.append(target_object_id)

        # check指令是否成功完成
        # instruction_success = controller_env.check_task(task_type=task_type, task_label_dict=pddl_params)

        instruction_response_dict = controller_env.check_task(task_type=task_type, task_label_dict=pddl_params)
        instruction_success = instruction_response_dict["success"]
        ts = instruction_response_dict["ts"]
        s = instruction_response_dict["s"]

        if instruction_success:
            success_number = success_number + 1
            success_index_list.append(val_dict_id)
        print("now success:", success_number / (val_dict_id + 1))
        result_dict_one["instruction"] = instruction
        result_dict_one["done_action"] = done_action.get_done_action_str()
        result_dict_one["success"] = instruction_success
        result_dict_one["house_id"] = house_id
        # 增加set_pose修正
        path_distance_all += done_action.distance_all
        result_dict_one["distance"] = path_distance_all
        reslut_list.append(result_dict_one)
        path_distance_list.append(path_distance_all)

        result_dict_one["ts"] = ts
        result_dict_one["s"] = s

        # add action list
        result_dict_one["exec_action_dict_list"] = dict(task_id=controller_env.exec_action_list)

        # 每个进行单独保存
        if proc_Config.alfred_scene:
            save_name_one = str(val_dict_id) + "_" + "house_" + str(house_id) + ".pkl"
        else:
            save_name_one = str(val_dict_id) + "_" + "house_" + str(house_id) + ".pkl"
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
    with open("llava_s1_s2_vln_parsed_response_v8_val_result_frontiers_200.pkl", 'wb') as f:
        pickle.dump(reslut_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    # add llava_s1 planner args
    parser.add_argument("--model-path-s1", type=str, default="facebook/opt-350m")
    parser.add_argument("--device-s1", type=str, default="cuda:3")
    args = parser.parse_args()
    main(args)