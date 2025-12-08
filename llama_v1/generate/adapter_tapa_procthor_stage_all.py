import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer
from lit_llama.adapter import LLaMA
from lit_llama.utils import lazy_load, llama_model_lookup, quantization
from scripts.prepare_alpaca import generate_prompt
from tqdm import tqdm
import json
from utils.prompt_utils import get_prompt_type_v4

# 加载具体仿真环境配置
import prior
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from utils.thor_env_code import ThorEnvCode
from tqdm import tqdm
from utils.prompt_utils import get_prompt_type_v4 as get_prompt_type
from procthor_config import Config as proc_Config
from utils.procthor_utils_v2 import get_frame_object_list, find_object_rotation, look_around_for_object, look_around_for_object_id
from utils.procthor_utils_v2 import get_frame_object_list_by_properties, nav_openable_object, nav_surrounded_object
from utils.procthor_utils_v2 import find_object_rotation_id
import random


def get_output(model, tokenizer, sample, max_new_tokens, temperature, top_k):
    # 生成sample_s1结果
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
    y = generate(
        model,
        idx=encoded,
        max_seq_length=max_new_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id
    )
    output = tokenizer.decode(y)
    output = output.split("### Response:")[1].strip()
    return output


def get_candidate_by_name(controller_env, target_name):
    candidate_position_list = []
    candidate_target_obj_id_list = []
    object_dict = controller_env.object_dict
    for obj_id, obj_dict_one in object_dict.items():
        obj_name_one = obj_dict_one["objectId"].split("|")[0]
        position_dict_one = obj_dict_one["position"]
        position_tuple = (position_dict_one["x"], position_dict_one["z"])
        # if obj_name_one == target_name:
        if target_name.lower() in obj_name_one.lower():
            candidate_position_list.append(position_tuple)
            candidate_target_obj_id_list.append(obj_id)

    return candidate_position_list, candidate_target_obj_id_list


def get_candidate_id_by_object_id(candidate_target_obj_id_list, object_id_list):
    # 根据生成的候选, 判断所观察的frame中是否有物体
    candidate_index_list = []
    for candidate_index, candidate_id in enumerate(candidate_target_obj_id_list):
        if candidate_id in object_id_list:
            candidate_index_list.append(candidate_index)

    return candidate_index_list


def get_nearest_nav_position(position_list, target_position):
    if target_position is not None:
        distance_list = []
        for position_one in position_list:
            distance_one = np.sqrt((target_position[0] - position_one[0]) ** 2 + (target_position[1] - position_one[1]) ** 2)
            distance_list.append(distance_one)
        min_index = np.argmin(np.asarray(distance_list))
        nearest_nav_position = position_list[min_index]
    else:
        nearest_nav_position = random.choice(position_list)
    return nearest_nav_position


def get_farthest_nav_position(position_list, agent_position):
    distance_list = []
    for position_one in position_list:
        distance_one = np.sqrt((agent_position[0] - position_one[0]) ** 2 + (
                    agent_position[1] - position_one[1]) ** 2)
        distance_list.append(distance_one)
    min_index = np.argmax(np.asarray(distance_list))
    farthest_nav_position = position_list[min_index]
    return farthest_nav_position


def teleport_nav_new(controller, position_tuple):
    is_arrive = False
    target_position = {"x": position_tuple[0], "y": position_tuple[1], "z": position_tuple[2]}
    for i in range(4):
        event = controller.step(action="Teleport", position=target_position)
        if controller.last_event.metadata["lastActionSuccess"]:
            is_arrive = True
            break
        else:
            event = controller.step(action="RotateRight", forceAction=True)
    return is_arrive


def nav_step_by_step(controller, nav_action):
    for action_one in nav_action:
        rgb, depth, done, event, api_action = controller.to_thor_api_exec([action_one])
    if controller.last_event.metadata["lastActionSuccess"]:
        return True
    else:
        return False


def main(
    prompt: str = "What food do lamas eat?",
    input: str = "",
    adapter_path: Path = Path("out/adapter/alpaca/lit-llama-adapter-finetuned.pth"),
    pretrained_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    quantize: Optional[str] = None,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
) -> None:
    assert adapter_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    # reload模型参数
    adapter_path_s1 = "./out/adapter/alpaca/lit_llama_v1_alpaca_v13_s1/lit-llama-adapter-finetuned.pth"
    adapter_path_s2 = "./out/adapter/alpaca/lit_llama_v1_alpaca_v13_s2/lit-llama-adapter-finetuned.pth"

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric_s1 = L.Fabric(devices=[0], precision=precision)
    fabric_s2 = L.Fabric(devices=[1], precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(adapter_path_s1) as adapter_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric_s1.init_module(empty_init=True), quantization(mode=quantize):
            model_s1 = LLaMA.from_name(name)

        # 1. Load the pretrained weights
        model_s1.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned adapter weights
        model_s1.load_state_dict(adapter_checkpoint, strict=False)

    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(adapter_path_s2) as adapter_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric_s2.init_module(empty_init=True), quantization(mode=quantize):
            model_s2 = LLaMA.from_name(name)

        # 1. Load the pretrained weights
        model_s2.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned adapter weights
        model_s2.load_state_dict(adapter_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model_s1.eval()
    model_s1 = fabric_s1.setup(model_s1)
    model_s2.eval()
    model_s2 = fabric_s2.setup(model_s2)

    tokenizer = Tokenizer(tokenizer_path)

    # model_s1: 负责第一阶段推理
    # model_s2: 负责第二阶段推理

    # 读取测试集
    val_dataset_path = "./data/alpaca/data_with_instruction_GPT4_v13_val.json"
    instruction_dict_list = json.load(open(val_dataset_path))
    # 读取instruction_list
    # 配置仿真环境
    # 模型初始化完成--------------------------------------------
    # 初始化仿真环境--------------------------------------------
    args = proc_Config
    dataset = prior.load_dataset("procthor-10k")
    dataset_train = dataset["train"]
    controller_env = ThorEnvCode(args, use_CloudRendering=False)
    # load初始化场景
    # rgb_frame, depth_frame = controller_env.load_scene(house_name=dataset_train[83])

    # 初始化衡量指标
    success_number = 0
    path_distance_list = []
    success_index_list = []
    nav_fail = 0
    planning_fail = 0
    exec_fail = 0

    # -------------------main loop--------------------------
    for val_dict_id, val_dict_one in enumerate(tqdm(instruction_dict_list[17:18])):
        instruction = val_dict_one["instruction"].strip(" ").strip("\"")
        house_id = int(val_dict_one["house_id"])
        rgb_frame, depth_frame = controller_env.load_scene(house_name=dataset_train[house_id])
        success = False
        done_list = "\n"
        done_action_dict = {}
        instrution_test_time = 0
        path_distance_all = 0
        while not success:
            instrution_test_time = instrution_test_time + 1
            select_s2 = False
            if instrution_test_time > 100:
                success = False
                break
            rgb_frame, depth_frame = controller_env.get_obs()
            # 环视寻找所有物体
            object_name_list, object_id_list, object_name_str = look_around_for_object_id(controller_env)
            object_name_str, position_str, agent_position_str = controller_env.get_llama_input_s1_only()
            # 生成第一段prompt
            instruction_prompt = get_prompt_type(prompt_type="s1")
            # 生成第一段input
            human_instruction_input = "Instruction: " + instruction + "\n" + "Object List: " + object_name_str + \
                                      "\n" + "Done Actions: " + done_list.rstrip("\n")
            print("-------------------")
            print(done_list)
            print("-------------------")
            # 统计尝试次数
            if done_list in done_action_dict.keys():
                done_action_dict[done_list] = done_action_dict[done_list] + 1
            else:
                done_action_dict[done_list] = 0

            try_times = done_action_dict[done_list]
            if try_times > 10:
                break

            s1_sample = {"instruction": instruction_prompt, "input": human_instruction_input}
            s1_output = get_output(model_s1, tokenizer, sample=s1_sample, max_new_tokens=max_new_tokens,
                                    temperature=temperature, top_k=top_k)

            print(s1_output)
            # 解析第一阶段output
            s1_output_list = s1_output.split("\n")
            current_action = s1_output_list[1]
            subsequent_planning_actions = s1_output_list[1].split("Subsequent Planning Actions: ")[-1]
            if "end" in current_action.lower():
                success = True
                continue

            # 处理current_action
            s1_action = current_action.split(", ")[0].split("action: ")[-1]
            target_name = current_action.split(", ")[-1].split("target object: ")[-1]

            # 执行第二阶段规划
            object_name_str, position_str, agent_position_str, position_list = controller_env.get_llama_input()
            print("------------------------")
            print(position_str)
            print(object_name_str)
            print("------------------------")
            instruction_prompt = get_prompt_type(prompt_type="s2")
            s2_input = "Execute Action:" + "\n" + current_action + "\n" + "Agent Position: " + agent_position_str + "\n" + \
                          "Object List: " + object_name_str + "\n" + "Position List: " + position_str

            s2_sample = {"instruction": instruction_prompt, "input": s2_input}
            s2_output = get_output(model_s2, tokenizer, sample=s2_sample, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k)

            print(s2_output)

            # 解析第二阶段输出
            single_step_action_dict = s2_output.split("Specific single-step action: ")[-1]
            s2_action = single_step_action_dict.split(", target object")[0].split("action: ")[-1]
            s2_target_name = single_step_action_dict.split(", target object: ")[-1].split(", target position")[0]
            s2_target_position = single_step_action_dict.split("target position: ")[-1]
            try:
                s2_target_position = eval(s2_target_position)
            except:
                s2_target_position = None

            # 生成具体的候选
            candidate_position_list, candidate_target_obj_id_list = get_candidate_by_name(controller_env, target_name)
            if len(candidate_target_obj_id_list) < 1:
                # continue
                planning_fail = planning_fail + 1
            # 根据agent的观察, 判断是否有候选物体在视野中
            # 不使用当前, 使用全局进行判定
            # candidate_index_list = get_candidate_id_by_object_id(candidate_target_obj_id_list, object_id_list)
            seen_obj_id_list = list(controller_env.seen_object_dict_all.keys())
            candidate_index_list = get_candidate_id_by_object_id(candidate_target_obj_id_list, seen_obj_id_list)
            if len(candidate_index_list) > 0:
                # 视野中有物体, 则直接选取s1规划
                print("find object, start s1")
                select_candidate_index = candidate_index_list[0]
                target_nav_position = candidate_position_list[select_candidate_index]
                target_object_id = candidate_target_obj_id_list[select_candidate_index]
                select_action = s1_action
            else:
                # 所观察的视野中没有物体, 启动第二阶段plannIng
                nav_fail = nav_fail + 1
                select_action = s2_action
                # target_nav_position = get_nearest_nav_position(position_list, s2_target_position)
                agent_position_dict = controller_env.last_event.metadata["agent"]["position"]
                agent_position_tuple = (agent_position_dict["x"], agent_position_dict["z"])
                target_nav_position = get_farthest_nav_position(position_list, agent_position=agent_position_tuple)
                target_object_id = None
                select_s2 = True

            # 如果选择了s2, 则需要重新进行规划
            # 完成抉择后进行具体执行
            smooth_nav = True
            if "GotoLocation" in [select_action] and select_s2:
                nav_action, path = controller_env.parse_nav_action(target_nav_position)
                print(nav_action)
                path_distance = (len(path) - 1) * 0.25
                find_object_in_nav = False
                if not smooth_nav:
                    for path_index in range(0, len(path), 3):
                        is_arrive = teleport_nav_new(controller_env, path[path_index])
                        object_name_list, object_id_list, object_name_str = look_around_for_object_id(controller_env)
                        object_name_str, position_str, agent_position_str = controller_env.get_llama_input_s1_only()
                        seen_obj_id_list = list(controller_env.seen_object_dict_all.keys())
                        candidate_index_list = get_candidate_id_by_object_id(candidate_target_obj_id_list, seen_obj_id_list)
                        if len(candidate_index_list) > 0:
                            print("###find object in nav-----------")
                            find_object_in_nav = True
                            break
                    if not find_object_in_nav:
                        is_arrive = teleport_nav_new(controller_env, path[-1])
                    else:
                        print("巡航执行结果:", is_arrive)
                        continue

                    print("巡航执行结果:", is_arrive)
                else:
                    for nav_action_id, action_one in enumerate(nav_action):
                        rgb, depth, done, event, api_action = controller_env.to_thor_api_exec([action_one])
                        if nav_action_id > 0 and nav_action_id % 3 == 0:
                            object_name_list, object_id_list, object_name_str = look_around_for_object_id(controller_env)
                            object_name_str, position_str, agent_position_str = controller_env.get_llama_input_s1_only()
                            seen_obj_id_list = list(controller_env.seen_object_dict_all.keys())
                            candidate_index_list = get_candidate_id_by_object_id(candidate_target_obj_id_list, seen_obj_id_list)
                            if len(candidate_index_list) > 0:
                                print("###find object in nav-----------")
                                find_object_in_nav = True
                                break
                    print("巡航执行结果:", find_object_in_nav)
                """
                for action_one in nav_action:
                    rgb, depth, done, event, api_action = controller_env.to_thor_api_exec([action_one])
                    # print("巡航执行结果: ", controller_env.last_event.metadata["lastActionSuccess"])
                """
            elif "GotoLocation" in [select_action] and not select_s2:
                nav_action, path = controller_env.parse_nav_action(target_nav_position)
                print(nav_action)
                path_distance = (len(path) - 1) * 0.25
                if not smooth_nav:
                    is_arrive = teleport_nav_new(controller_env, path[-1])
                    print("到达物体旁边:", is_arrive)
                else:
                    for action_one in nav_action:
                        rgb, depth, done, event, api_action = controller_env.to_thor_api_exec([action_one])

                    print("到达物体旁边:", controller_env.last_event.metadata["lastActionSuccess"])
            else:
                path_distance = 0
                find_object, target_id = find_object_rotation_id(controller_env, target_name, target_object_id)
                print("是否找到物体:", find_object)

                # 增加巡航
                nav_action, path = controller_env.parse_nav_action(target_nav_position)
                for action_one in nav_action:
                    rgb, depth, done, event, api_action = controller_env.to_thor_api_exec([action_one])

                # 旋转保持物体在视野中
                find_object, target_id = find_object_rotation_id(controller_env, target_name, target_id)

                # 增加具体动作分类
                if "Toggle" in select_action and target_id is not None:
                    is_Toggled = controller_env.object_dict[target_id]["isToggled"]
                    if is_Toggled:
                        select_action = "ToggleObjectOff"
                    else:
                        select_action = "ToggleObjectOn"

                rgb, depth, _, done = controller_env.execAction(action=[
                    select_action], target_arg=target_name,
                    target_position=target_nav_position, target_object_id=target_id)
                if not controller_env.last_event.metadata["lastActionSuccess"]:
                    if "PutObject" in select_action:
                        event = controller_env.step(
                            action="MoveHeldObjectAhead",
                            moveMagnitude=0.2,
                            forceVisible=True
                        )
                        drop_object_acion = dict(action="DropHandObject", forceAction=True)
                        event = controller_env.step(drop_object_acion)

                target_object_id = target_id

            find_object = find_object_rotation(controller_env, target_name)
            # 统计路径长度
            path_distance_all = path_distance_all + path_distance
            if "GotoLocation" in select_action and not find_object and select_s2:
                continue
            if select_s2:
                continue
            if controller_env.last_event.metadata["lastActionSuccess"]:
                done_list = done_list + current_action + "\n"
            else:
                action_exec = False
                if target_object_id is not None:
                    target_object_class = target_object_id.split("|")[0]
                    object_data = controller_env.last_event.metadata["objects"]
                    for object_one in object_data:
                        object_one_name = object_one["objectId"].split("|")[0]
                        if target_object_class.lower() in object_one_name.lower():
                            is_toggle = object_one["toggleable"]
                            if is_toggle:
                                action_exec = True
                if action_exec:
                    done_list = done_list + current_action + "\n"
                else:
                    exec_fail = exec_fail + 1

        # 统计衡量指标
        if success:
            success_number = success_number + 1
            success_index_list.append(val_dict_id)
        path_distance_list.append(path_distance_all)
    # 计算衡量指标
    print("成功率:", success_number / 20)
    print("平均路径长度:", np.sum(np.asarray(path_distance_list)) / 20)
    print("完成测试--------------------------------------")
    print(success_index_list)

    print("失败原因:---------------------")
    print("巡航问题, not find object", nav_fail)
    print("幻觉问题", planning_fail)
    print("执行问题", exec_fail)
if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)


