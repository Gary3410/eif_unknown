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

# 加载具体仿真环境配置
import prior

from utils.prompt_utils import get_prompt_type_v5 as get_prompt_type
from utils.procthor_config import Config as proc_Config
import open_clip

from utils.generate_response_llama import Planner_LLaMA
from utils.sem_map import Semantic_Mapping
from utils.thor_env_code_v2 import ThorEnvCode
import prior
import pickle


def get_farthest_nav_position(position_list, agent_position):
    distance_list = []
    for position_one in position_list:
        distance_one = np.sqrt((agent_position[0] - position_one[0]) ** 2 + (
                    agent_position[1] - position_one[1]) ** 2)
        distance_list.append(distance_one)
    min_index = np.argmax(np.asarray(distance_list))
    farthest_nav_position = position_list[min_index]
    return farthest_nav_position


def perspective_camera_view(controller):
    # 默认视角为30
    normal_view_angle = 30
    # 先获取当前视角的角度
    cameraHorizon = controller.last_event.metadata["agent"]["cameraHorizon"]
    if int(cameraHorizon) - normal_view_angle < 0:
        # 需要进行look_down
        event = controller.step(action="LookDown", degrees=30)
    elif int(cameraHorizon) - normal_view_angle > 0:
        # 需要进行look_up
        event = controller.step(action="LookUp", degrees=30)
    else:
        normal_view_angle = 30


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

    # 配置文件路径
    # reload模型参数
    adapter_path_s1 = "./checkpoints/lit_llama_v1_alpaca_v8_easy_s1/lit-llama-adapter-finetuned.pth"
    adapter_path_s2 = "./checkpoints/lit_llama_v1_alpaca_v8_easy_s2/lit-llama-adapter-finetuned.pth"

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

    llama_args_dict = dict(
        max_new_tokens=1024,
        top_k=200,
        temperature=0.8
    )

    planner = Planner_LLaMA(
        llama_model_s1=model_s1,
        llama_model_s2=model_s2,
        llama_args_dict=llama_args_dict,
        llama_tokenizer=tokenizer
    )

    # 读取测试集
    val_dataset_path = "./data/spaced_parse_instruction_easy_v8_val.json"
    instruction_dict_list = json.load(open(val_dataset_path))

    args = proc_Config
    dataset = prior.load_dataset("procthor-10k")
    dataset_train = dataset["train"]
    controller_env = ThorEnvCode(args, use_CloudRendering=True)

    instruction_prompt_s1 = get_prompt_type(prompt_type="s1")
    instruction_prompt_s2 = get_prompt_type(prompt_type="s2")

    # 初始化衡量指标
    success_number = 0
    path_distance_list = []
    success_index_list = []
    nav_fail = 0
    planning_fail = 0
    exec_fail = 0
    reslut_list = []
    # -------------------main loop--------------------------
    for val_dict_id, val_dict_one in enumerate(tqdm(instruction_dict_list)):
        instruction = val_dict_one["instruction"].strip(" ").strip("\"")
        house_id = int(val_dict_one["house_id"])
        task_type = val_dict_one["task_type"]
        pddl_params = val_dict_one["pddl_params"]

        rgb_frame, depth_frame = controller_env.load_scene(house_name=dataset_train[house_id])
        success = False
        done_list = "\n"
        done_action_dict = {}
        instruction_test_time = 0
        path_distance_all = 0
        # controller_env.open_door()
        # 首先进行环视
        for _ in range(4):
            # 增加转圈
            rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
            event = controller_env.step(action="RotateRight", forceAction=True)
        print("end surrounding ----------------------")
        result_dict_one = {}
        result_dict_one["step_action_list"] = []
        while not success:
            rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
            select_s2 = False
            step_action_dict_one = {}
            instruction_test_time = instruction_test_time + 1
            if instruction_test_time > 30:
                success = False
                break
            # 统计尝试次数
            if done_list in done_action_dict.keys():
                done_action_dict[done_list] = done_action_dict[done_list] + 1
            else:
                done_action_dict[done_list] = 0
            try_times = done_action_dict[done_list]
            if try_times > 10:
                break
            object_name_list = controller_env.seen_object_name()
            # 生成第一段规划
            new_object_list = controller_env.object_dict2str_llama_s1_only()
            # 生成第一段input
            human_instruction_input = "Instruction: " + instruction + "\n" + "Object List: " + new_object_list + \
                                      "\n" + "Trying Times: " + str(try_times) + "\n" + "Done Actions: " + done_list.rstrip("\n")

            s1_input = {"instruction": instruction_prompt_s1, "input": human_instruction_input}
            s1_response = planner.get_llama_response(s1_input, stage="s1")
            print("s1_response: ", s1_response)
            try:
                current_action = planner.parse_llama_s1_response(s1_response)
            except:
                continue
            current_action_for_s2 = current_action.split(". ")[-1]

            step_action_dict_one["s1_planning"] = current_action

            # 进行s2 规划
            object_name_str, position_str, agent_position_str, position_list = controller_env.get_llama_input()
            s2_input = "Execute Action:" + "\n" + current_action_for_s2 + "\n" + "Agent Position: " + agent_position_str + "\n" + \
                       "Object List: " + object_name_str + "\n" + "Position List: " + position_str

            print(s2_input)

            s2_sample = {"instruction": instruction_prompt_s2, "input": s2_input}
            try:
                s2_response = planner.get_llama_response(s2_sample, stage="s2")
            except:
                continue
            print(s2_response)
            step_action_dict_one["s2_action"] = s2_response.split("Specific single-step action: ")[-1]
            s2_action, target_name = planner.parse_llama_s2_response(s2_response)

            # 根据s2来判断是否成功
            if "end" in s2_action.lower():
                success = True
                continue

            # 根据target生成候选
            candidate_position_list, candidate_target_obj_id_list = controller_env.get_candidate_by_name(target_name)
            if len(candidate_target_obj_id_list) < 1:
                planning_fail = planning_fail + 1
            candidate_index_list = controller_env.get_candidate_id_by_object_id(candidate_target_obj_id_list)
            if len(candidate_index_list) > 0:
                # 视野中有物体, 则直接选取s1规划
                print("find object, start s1")
                select_candidate_index = candidate_index_list[0]
                target_nav_position = candidate_position_list[select_candidate_index]
                target_object_id = candidate_target_obj_id_list[select_candidate_index]
                select_action = s2_action
            else:
                nav_fail = nav_fail + 1
                select_action = s2_action
                # target_nav_position = get_nearest_nav_position(position_list, s2_target_position)
                agent_position_dict = controller_env.last_event.metadata["agent"]["position"]
                agent_position_tuple = (agent_position_dict["x"], agent_position_dict["z"])
                target_nav_position = get_farthest_nav_position(position_list, agent_position=agent_position_tuple)
                target_object_id = None
                select_s2 = True
            if target_object_id is not None:
                step_action_dict_one = dict(nav_position=target_nav_position, action=select_action, target=
                target_object_id.split("|")[0], target_id=target_object_id)
            else:
                step_action_dict_one = dict(nav_position=target_nav_position, action=select_action, target=target_name, target_id=target_object_id)
            result_dict_one["step_action_list"].append(step_action_dict_one)

            # 执行动作
            if "goto" in select_action.lower() or target_object_id is None:
                perspective_camera_view(controller_env)
                print("target_nav_position: ", target_nav_position)
                nav_action, path = controller_env.parse_nav_action(target_nav_position)
                path_distance = (len(path) - 1) * 0.25
                for nav_action_id, action_one in enumerate(tqdm(nav_action)):
                    rgb, depth_frame, mask_list, info_dict = controller_env.to_thor_api_exec([action_one])
            else:
                path_distance = 0
                # add 距离限制
                is_near = controller_env.check_is_near_target(target_object_id)
                if not is_near:
                    path_distance = controller_env.goto_location_nav(target_nav_position)
                # 环视寻找物体
                is_find = controller_env.check_target_frame(target_object_id)
                rgb, depth_frame, mask_list, info_dict = controller_env.execAction(action=[
                    select_action], target_arg=target_object_id.split("|")[0],
                    target_position=target_nav_position, target_object_id=target_object_id)

            path_distance_all = path_distance_all + path_distance
            if "goto" in select_action.lower():
                candidate_index_list = controller_env.get_candidate_id_by_object_id(candidate_target_obj_id_list)
                if len(candidate_index_list) > 0:
                    done_list = done_list + current_action + "\n"
            else:
                if controller_env.last_event.metadata["lastActionSuccess"]:
                    done_list = done_list + current_action + "\n"

        instruction_success = controller_env.check_task(task_type=task_type, task_label_dict=pddl_params)
        if instruction_success:
            success_number = success_number + 1
            success_index_list.append(val_dict_id)
        print("now success:", success_number / (val_dict_id + 1))
        result_dict_one["instruction"] = instruction
        result_dict_one["done_action"] = done_list
        result_dict_one["success"] = success
        result_dict_one["house_id"] = house_id
        reslut_list.append(result_dict_one)
        path_distance_list.append(path_distance_all)

    # 计算衡量指标
    print("成功率:", success_number / len(path_distance_list))
    print("平均路径长度:", np.sum(np.asarray(path_distance_list)) / len(path_distance_list))
    print("完成测试--------------------------------------")
    print(success_index_list)

    print("失败原因:---------------------")
    print("巡航问题, not find object", nav_fail)
    print("幻觉问题", planning_fail)
    print("执行问题", exec_fail)

    # 保存结果
    with open("llama_vln_parsed_response_v8_val_easy_result.json", 'w') as f:
        json_str = json.dumps(reslut_list, indent=2)
        f.write(json_str)
        f.write('\n')


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)
