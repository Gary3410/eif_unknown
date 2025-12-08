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
from utils.prompt_utils import get_prompt_type_v3 as get_prompt_type
from procthor_config import Config as proc_Config
from utils.procthor_utils_v2 import get_frame_object_list, find_object_rotation, look_around_for_object
from utils.procthor_utils_v2 import get_frame_object_list_by_properties, nav_openable_object, nav_surrounded_object
from utils.procthor_utils_v2 import teleport_nav_new

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
    controller_env = ThorEnvCode(args, use_CloudRendering=False)
    # load初始化场景
    rgb_frame, depth_frame = controller_env.load_scene(house_name=dataset_train[83])
    # 仿真环境初始化完成----------------------------------------

    # -------------------main loop--------------------------
    success = False
    # instruction = "Please bring the egg from the fridge to the countertop."
    # instruction = "Please bring the soap-bottle to the countertop"
    instruction = "Please bring the soap-bottle to the toilet"
    done_list = "\n"
    done_action_dict = {}
    while not success:

        rgb_frame, depth_frame = controller_env.get_obs()
        object_name_list, object_name_str = look_around_for_object(controller_env)

        # 生成第一段prompt
        instruction_prompt = get_prompt_type(prompt_type="s1")
        # 生成第一段input
        human_instruction_input = "Instruction: " + instruction + "\n" + "Object List: " + object_name_str + \
                                  "\n" + "Done Actions: " + done_list.rstrip("\n")

        print(human_instruction_input)
        if done_list in done_action_dict.keys():
            done_action_dict[done_list] = done_action_dict[done_list] + 1
        else:
            done_action_dict[done_list] = 0

        sample_s1 = {"instruction": instruction_prompt, "input": human_instruction_input}
        # 生成sample_s1结果
        prompt = generate_prompt(sample_s1)
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
        s1_output = output.split("### Response:")[1].strip()

        print("s1_output--------------------------")
        print(s1_output)

        # 解析第一阶段output
        s1_output_list = s1_output.split("\n")
        current_action = s1_output_list[1]
        subsequent_planning_actions = s1_output_list[1].split("Subsequent Planning Actions: ")[-1]
        if "end" in current_action.lower():
            success = True
            continue

        # action: GotoLocation, target object: Toilet
        # 处理current_action
        executable_action = current_action.split(", ")[0].split("action: ")[-1]
        target_name = current_action.split(", ")[-1].split("target object: ")[-1]

        # 根据target_name 生成具体的巡航位置
        # 巡航场景的GT [object_dict]
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
        if len(candidate_position_list) < 1:
            continue
        if "GotoLocation" in [executable_action]:
            nav_action, path = controller_env.parse_nav_action(candidate_position_list[0])
            print(nav_action)
            for action_one in nav_action:
                rgb, depth, done, event, api_action = controller_env.to_thor_api_exec([action_one])
                print("巡航执行结果: ", controller_env.last_event.metadata["lastActionSuccess"])
        else:
            find_object = find_object_rotation(controller_env, target_name)
            if not find_object:
                # 有可能包含在物体中
                # 进行遍历开启
                openable_target_id = nav_openable_object(controller_env, candidate_target_obj_id_list[0])
                open_action = dict(action="OpenObject", objectId=openable_target_id, forceAction=True)
                event = controller_env.step(open_action)
                # 增加周围巡航
                nav_surrounded_position_list = nav_surrounded_object(controller_env, openable_target_id)
                for position_list_one in nav_surrounded_position_list:
                    is_arrive = teleport_nav_new(controller_env, position_list_one)
                    find_object = find_object_rotation(controller_env, target_name)
                    if find_object:
                        break
                if not find_object:
                    continue
            rgb, depth, _, done = controller_env.execAction(action=[
                executable_action], target_arg=target_name,
                target_position=candidate_position_list[0], target_object_id=candidate_target_obj_id_list[0])
            if "PutObject" in executable_action:
                event = controller_env.step(
                    action="MoveHeldObjectAhead",
                    moveMagnitude=0.1,
                    forceVisible=True
                )
                drop_object_acion = dict(action="DropHandObject", forceAction=True)
                event = controller_env.step(drop_object_acion)
            print("交互执行结果: ", controller_env.last_event.metadata["lastActionSuccess"])
        try_times = done_action_dict[done_list]
        if controller_env.last_event.metadata["lastActionSuccess"] or try_times > 6:
            done_list = done_list + current_action + "\n"
    print(done_action_dict)

if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)