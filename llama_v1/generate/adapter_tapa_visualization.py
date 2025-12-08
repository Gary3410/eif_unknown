import os
import random
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
from lit_llama.utils import lazy_load, llama_model_lookup, quantization
from scripts.prepare_alpaca import generate_prompt
from tqdm import tqdm
import json
from utils.prompt_utils import get_prompt_type_v5

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

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=[0], precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(adapter_path) as adapter_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric.init_module(empty_init=True), quantization(mode=quantize):
            model = LLaMA.from_name(name)

        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned adapter weights
        model.load_state_dict(adapter_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)
    tokenizer = Tokenizer(tokenizer_path)

    # 读取测试集
    val_dataset_path = "./data/alpaca/llama_vln_parsed_response_v15_val_s1.json"
    dataset_name = val_dataset_path.split("/")[-1]
    prompt_type = dataset_name.split(".")[0].split("_")[-1]
    input_prompt = get_prompt_type_v5(prompt_type=prompt_type)

    val_dict_list = json.load(open(val_dataset_path))
    output_dict_list = []
    counterfactual_input_list = []
    for val_id, val_dict_one in enumerate(tqdm(val_dict_list)):
        input = val_dict_one["input"]
        output_gt = val_dict_one["output"]
        s1_output_list = output_gt.split("\n")
        current_action = s1_output_list[1]
        sample = {"instruction": input_prompt, "input": input, "output_gt": output_gt}
        if "s2" in prompt_type:
            if "replace" in input:
                if len(counterfactual_input_list) < 50:
                    counterfactual_input_list.append(sample)
        if "s1" in prompt_type:
            if "replace" in current_action:
                if len(counterfactual_input_list) < 50:
                    counterfactual_input_list.append(sample)

    print("反事实样本:", len(counterfactual_input_list))

    for val_id, val_dict_one in enumerate(tqdm(val_dict_list[:50])):
        input_instruction = val_dict_one["instruction"]
        input = val_dict_one["input"]
        output_gt = val_dict_one["output"]
        sample = {"instruction": input_prompt, "input": input}


        prompt = generate_prompt(sample)
        encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
        prompt_length = encoded.size(0)
        y = generate(model, encoded, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)

        model.reset_cache()
        output = tokenizer.decode(y)
        output = output.split("### Response:")[1].strip()
        output_dict_one = {}
        output_dict_one["instruction"] = input_prompt
        output_dict_one["input"] = input
        output_dict_one["output_gt"] = output_gt
        output_dict_one["output"] = output

        output_dict_list.append(output_dict_one)

    # 增加反事实测试样本
    for sample_one in tqdm(counterfactual_input_list):
        input_prompt = sample_one["instruction"]
        output_gt = sample_one["output_gt"]
        input = sample_one["input"]

        sample = {"instruction": input_prompt, "input": input}

        prompt = generate_prompt(sample)
        encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
        prompt_length = encoded.size(0)
        y = generate(model, encoded, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)

        model.reset_cache()
        output = tokenizer.decode(y)
        output = output.split("### Response:")[1].strip()
        output_dict_one = {}
        output_dict_one["instruction"] = input_prompt
        output_dict_one["input"] = input
        output_dict_one["output_gt"] = output_gt
        output_dict_one["output"] = output
        output_dict_list.append(output_dict_one)

    # 进行保存
    with open("out_json/llama_output_response_v15_val_s1.json", 'w') as f:
        json_str = json.dumps(output_dict_list, indent=2)
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