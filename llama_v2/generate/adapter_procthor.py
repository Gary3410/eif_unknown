import sys
import time
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import Tokenizer
from lit_gpt.adapter import GPT, Block, Config
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, gptq_quantization, lazy_load
from scripts.prepare_alpaca import generate_prompt

# 加载具体仿真环境配置
import prior
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from utils.thor_env_code import ThorEnvCode
from tqdm import tqdm
from arguments import get_args
from utils.procthor_utils import get_prompt_type
from lit_gpt.procthor_config import Config as proc_Config


def main(
    prompt: str = "What food do lamas eat?",
    input: str = "",
    adapter_path: Path = Path("out/adapter/alpaca/lit_model_adapter_finetuned.pth"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    strategy: str = "auto",
    devices: int = 1,
    precision: Optional[str] = None,
) -> None:

    # 初始化llama_model
    # 初始化具体模型环境
    # args = get_args()
    args = proc_Config
    dataset = prior.load_dataset("procthor-10k")
    dataset_train = dataset["train"]
    controller_env = ThorEnvCode(args, use_CloudRendering=False)
    # load初始化场景
    rgb_frame, depth_frame = controller_env.load_scene(house_name=dataset_train[83])

    # 初始化llama_v2模型---------------------------
    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None:
        if devices > 1:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 when using the"
                " --quantize flag."
            )
        if quantize.startswith("bnb."):
            if "mixed" in precision:
                raise ValueError("Quantization and mixed precision is not supported.")
            dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
            plugins = BitsandbytesPrecision(quantize[4:], dtype)
            precision = None

    if strategy == "fsdp":
        strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)

    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy, plugins=plugins)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(checkpoint_dir / "lit_config.json")

    if quantize is not None and devices > 1:
        raise NotImplementedError
    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True), gptq_quantization(quantize == "gptq.int4"):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.perf_counter()
    checkpoint = lazy_load(checkpoint_path)
    adapter_checkpoint = lazy_load(adapter_path)
    checkpoint.update(adapter_checkpoint.get("model", adapter_checkpoint))
    model.load_state_dict(checkpoint)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    # 模型初始化完成--------------------------------------------
    success = False
    instruction = "Clean the windows."
    tokenizer = Tokenizer(checkpoint_dir)
    done_list = "\n"
    while not success:

        while True:
            rgb_frame, depth_frame = controller_env.get_obs()
            object_name_str, position_str, agent_position_str = controller_env.get_llama_input()
            if "[]" in position_str:
                print("look around for nav position-------")
                controller_env.look_around()
            else:
                break
        # 生成第一段prompt
        instruction_prompt = get_prompt_type(prompt_type="s1")
        # 生成第一段input
        human_instruction_input = "Instruction: " + instruction + "\n" + "Agent Position: " + agent_position_str + "\n" + "Object List: " + object_name_str + \
                                  "\n" + "Done Actions: " + done_list
        # sample = {"instruction": prompt, "input": input}
        sample = {"instruction": instruction_prompt, "input": human_instruction_input}
        print("一阶段输入----------------------")
        print(instruction_prompt)
        print(human_instruction_input)
        print("一阶段输输入----------------------")
        # 生成具体output
        prompt = generate_prompt(sample)
        encoded = tokenizer.encode(prompt, device=fabric.device)
        prompt_length = encoded.size(0)
        max_returned_tokens = prompt_length + max_new_tokens
        with fabric.init_tensor():
            # set the max_seq_length to limit the memory usage to what we need
            model.max_seq_length = max_returned_tokens
            # enable the kv cache
            model.set_kv_cache(batch_size=1)
        y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
        output = tokenizer.decode(y)
        s1_output = output.split("### Response:")[1].strip()
        print("第一阶段输出-------------------------")
        print(s1_output)
        print("第一阶段输出-------------------------")

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

            # 执行第二段生成
            sample = {"instruction": instruction_prompt, "input": human_instruction_input}
            # 生成具体output
            prompt = generate_prompt(sample)
            encoded = tokenizer.encode(prompt, device=fabric.device)
            prompt_length = encoded.size(0)
            max_returned_tokens = prompt_length + max_new_tokens
            with fabric.init_tensor():
                # set the max_seq_length to limit the memory usage to what we need
                model.max_seq_length = max_returned_tokens
                # enable the kv cache
                model.set_kv_cache(batch_size=1)
            y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
            output = tokenizer.decode(y)
            s2_output = output.split("### Response:")[1].strip()

            print("第二阶段输出-------------------------")
            print(s2_output)
            print("第二阶段输出-------------------------")

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
                rgb, depth, _, done = controller_env.execAction(action=[action], target_arg=target_arg, target_position=target_position)
                print("交互执行结果: ", controller_env.last_event.metadata["lastActionSuccess"])
            # 感知具体环境
            # 第二段新的感知
            rgb_frame, depth_frame = controller_env.get_obs()
            object_name_str, position_str, agent_position_str = controller_env.get_llama_input()

            # 生成第三段prompt
            human_instruction_prompt = get_prompt_type(prompt_type="s3")
            human_instruction_input = "Execute Action: " + current_action + "\n" + "Agent Position: " + agent_position_str + "\n" + \
                                      "Subsequent Planning Actions: " + subsequent_planning_actions

            # 执行第三段生成
            sample = {"instruction": instruction_prompt, "input": human_instruction_input}
            # 生成具体output
            prompt = generate_prompt(sample)
            encoded = tokenizer.encode(prompt, device=fabric.device)
            prompt_length = encoded.size(0)
            max_returned_tokens = prompt_length + max_new_tokens
            with fabric.init_tensor():
                # set the max_seq_length to limit the memory usage to what we need
                model.max_seq_length = max_returned_tokens
                # enable the kv cache
                model.set_kv_cache(batch_size=1)
            y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
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
    CLI(main)