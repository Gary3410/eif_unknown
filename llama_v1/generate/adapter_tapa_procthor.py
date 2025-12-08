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
from utils.procthor_utils import get_prompt_type_v2 as get_prompt_type
from procthor_config import Config as proc_Config

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
    instruction = "Please bring the egg from the fridge to the countertop."
    done_list = "\n"
    while not success:
        while True:
            test_time = 0
            rgb_frame, depth_frame = controller_env.get_obs()
            object_name_str, position_str, agent_position_str = controller_env.get_llama_input()
            if "[]" in position_str:
                print("look around for nav position-------")
                controller_env.look_around()
                test_time = test_time + 1
            elif test_time > 4:
                break
            else:
                break
        # 生成第一段prompt
        instruction_prompt = get_prompt_type(prompt_type="s1")
        # 生成第一段input
        human_instruction_input = "Instruction: " + instruction + "\n" + "Agent Position: " + agent_position_str + "\n" + "Object List: " + object_name_str + \
                                  "\n" + "Done Actions: " + done_list.rstrip("\n")
        print("s1 input------------------------")
        print(human_instruction_input)

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

        print("第一阶段输出-------------------------")
        print(s1_output)

        # 解析第一阶段output
        s1_output_list = s1_output.split("\n")
        current_action = s1_output_list[0].split("Current Action: ")[-1]
        subsequent_planning_actions = s1_output_list[1].split("Subsequent Planning Actions: ")[-1]
        if "end" in current_action:
            success = True
            continue

        # 生成第二段输入
        step_action_finish = False
        while not step_action_finish:
            instruction_prompt = get_prompt_type(prompt_type="s2")
            human_instruction_input = "Execute Action: " + current_action + "\n" + "Agent Position: " + agent_position_str + "\n" + \
                                      "Object List: " + object_name_str + "\n" + "Reachable Position List: " + position_str

            print("s2 input------------------------")
            print(human_instruction_input)

            # 执行第二段生成
            sample_s2 = {"instruction": instruction_prompt, "input": human_instruction_input}

            # 生成sample_s2结果
            prompt = generate_prompt(sample_s2)
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
            s2_output = output.split("### Response:")[1].strip()

            print("第二阶段输出-------------------------")
            print(s2_output)


            # 解析第二段output
            single_step_action_dict = s2_output.split("Specific single-step action: ")[-1]
            action = single_step_action_dict.split(", target position:")[0].split("action: ")[-1]
            target_arg = single_step_action_dict.split("target object: ")[-1].strip("}")
            target_position = single_step_action_dict.split(", target object:")[0].split("target position: ")[-1]
            target_position = eval(target_position)

            # 开始具体执行
            if "GotoLocation" in [action]:
                nav_action = controller_env.parse_nav_action(target_position)
                print(nav_action)
                for action_one in nav_action:
                    rgb, depth, done, event, api_action = controller_env.to_thor_api_exec([action_one])
                    print("巡航执行结果: ", controller_env.last_event.metadata["lastActionSuccess"])
            else:
                rgb, depth, _, done = controller_env.execAction(action=[
                    action], target_arg=target_arg, target_position=target_position)
                print("交互执行结果: ", controller_env.last_event.metadata["lastActionSuccess"])

            rgb_frame, depth_frame = controller_env.get_obs()
            object_name_str, position_str, agent_position_str = controller_env.get_llama_input()
            # 生成第三段prompt
            human_instruction_prompt = get_prompt_type(prompt_type="s3")
            human_instruction_input = "Execute Action: " + current_action + "\n" + "Agent Position: " + agent_position_str + "\n" + \
                                      "Subsequent Planning Actions: " + subsequent_planning_actions
            print("s3 input------------------------")
            print(human_instruction_input)

            # 执行第三段生成
            sample_s3 = {"instruction": human_instruction_prompt, "input": human_instruction_input}

            # 生成sample_s3结果
            prompt = generate_prompt(sample_s3)
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
            s3_output = output.split("### Response:")[1].strip()

            print("第三阶段输出-------------------------")
            print(s3_output)
            print("第三阶段输出-------------------------")

            if "no" in s3_output:
                continue
            if "yes" in s3_output:
                step_action_finish = True
                # 更新done action
                done_list = done_list + current_action + "\n"



if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)