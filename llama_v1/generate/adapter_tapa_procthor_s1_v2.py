import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer
from lit_llama.adapter import LLaMA
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
from scripts.prepare_alpaca import generate_prompt

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
import numpy as np
import json


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
        if obj_name_one == target_name:
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

    fabric = L.Fabric(devices=1)
    dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(adapter_path) as adapter_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with EmptyInitOnDevice(
                device=fabric.device, dtype=dtype, quantization_mode=quantize
        ):
            model = LLaMA.from_name(name)

        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned adapter weights
        model.load_state_dict(adapter_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)
    print(f"Max_seq_len input model: {max_new_tokens}", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)
    tokenizer = Tokenizer(tokenizer_path)
    # 模型初始化完成--------------------------------------------
    # 初始化仿真环境--------------------------------------------
    args = proc_Config
    dataset = prior.load_dataset("procthor-10k")
    dataset_train = dataset["train"]
    controller_env = ThorEnvCode(args, use_CloudRendering=True)

    # 读取测试集
    val_dataset_path = "./data/alpaca/data_with_instruction_GPT4_v13_val.json"
    instruction_dict_list = json.load(open(val_dataset_path))

    # 初始化衡量指标
    success_number = 0
    path_distance_list = []
    success_index_list = []

    for val_dict_id, val_dict_one in enumerate(tqdm(instruction_dict_list)):
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
            if instrution_test_time > 40:
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
            # 统计尝试次数
            if done_list in done_action_dict.keys():
                done_action_dict[done_list] = done_action_dict[done_list] + 1
            else:
                done_action_dict[done_list] = 0

            print("-------------------")
            print(done_list)
            print("-------------------")

            try_times = done_action_dict[done_list]
            if try_times > 10:
                break

            s1_sample = {"instruction": instruction_prompt, "input": human_instruction_input}
            s1_output = get_output(model, tokenizer, sample=s1_sample, max_new_tokens=max_new_tokens,
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
            if len(candidate_position_list) < 1:
                print("Not find object, please re-planning")
                # continue
                # 随机去一个位置
                event = controller_env.step(action="GetReachablePositions")
                positions = event.metadata["actionReturn"]
                initial_position = random.choice(positions)
                event = controller_env.step(action="Teleport", position=initial_position, rotation=dict(x=0, y=90, z=0))
                continue

            for candidate_index_one in range(len(candidate_position_list)):

                if "GotoLocation" in [s1_action]:
                    nav_action, path = controller_env.parse_nav_action(candidate_position_list[candidate_index_one])
                    print(nav_action)
                    path_distance = (len(path) - 1) * 0.25
                    if len(path) > 1:
                        is_arrive = teleport_nav_new(controller_env, path[-2])
                    else:
                        is_arrive = teleport_nav_new(controller_env, path[-1])
                    print("巡航执行结果:", is_arrive)
                    """
                    for action_one in nav_action:
                        rgb, depth, done, event, api_action = controller_env.to_thor_api_exec([action_one])
                        print("巡航执行结果: ", controller_env.last_event.metadata["lastActionSuccess"])
                    """
                else:
                    find_object = find_object_rotation(controller_env, target_name)
                    if not find_object:
                        continue
                    # find_object, target_id = find_object_rotation_id(controller_env, target_name, target_object_id)
                    path_distance = 0
                    # 增加具体动作分类
                    target_object_id = candidate_target_obj_id_list[candidate_index_one]
                    if "Toggle" in s1_action:
                        is_Toggled = controller_env.object_dict[target_object_id]["isToggled"]
                        if is_Toggled:
                            s1_action = "ToggleObjectOff"
                        else:
                            s1_action = "ToggleObjectOn"
                    rgb, depth, _, done = controller_env.execAction(action=[
                        s1_action], target_arg=target_name,
                        target_position=candidate_position_list[candidate_index_one], target_object_id=candidate_target_obj_id_list[candidate_index_one])
                    if "PutObject" in s1_action:
                        event = controller_env.step(
                            action="MoveHeldObjectAhead",
                            moveMagnitude=0.1,
                            forceVisible=True
                        )
                        drop_object_acion = dict(action="DropHandObject", forceAction=True)
                        event = controller_env.step(drop_object_acion)
                    print("交互执行结果: ", controller_env.last_event.metadata["lastActionSuccess"])
                path_distance_all = path_distance_all + path_distance
                if controller_env.last_event.metadata["lastActionSuccess"]:
                    break

            if controller_env.last_event.metadata["lastActionSuccess"]:
                done_list = done_list + current_action + "\n"

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

if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)