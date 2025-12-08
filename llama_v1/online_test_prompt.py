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
from utils.procthor_utils import get_prompt_type
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

    args = proc_Config
    dataset = prior.load_dataset("procthor-10k")
    dataset_train = dataset["train"]
    controller_env = ThorEnvCode(args, use_CloudRendering=True)
    # load初始化场景
    rgb_frame, depth_frame = controller_env.load_scene(house_name=dataset_train[83])

    instruction = "Clean the windows."
    done_list = "\n"

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
                              "\n" + "Done Actions: " + done_list
    sample_s1 = {"instruction": instruction_prompt, "input": human_instruction_input}
    print(sample_s1["instruction"])
    print("----------------------")
    print(sample_s1["input"])

    prompt = generate_prompt(sample_s1)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
    print(f"token shape: {encoded.shape}", file=sys.stderr)

    current_action = "test"
    instruction_prompt = get_prompt_type(prompt_type="s2")
    human_instruction_input = "Execute Action: " + current_action + "\n" + "Agent Position: " + agent_position_str + "\n" + \
                              "Object List: " + object_name_str + "\n" + "Reachable Position List: " + position_str

    # 执行第二段生成
    sample_s2 = {"instruction": instruction_prompt, "input": human_instruction_input}
    print(sample_s2["instruction"])
    print("----------------------")
    print(sample_s2["input"])

    prompt = generate_prompt(sample_s2)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
    print(f"token shape: {encoded.shape}", file=sys.stderr)

if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)