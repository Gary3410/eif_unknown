import os
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
from utils.prompt_utils import get_prompt_type_v4

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
    adapter_path_s1 = "./out/adapter/alpaca/lit_llama_v1_alpaca_v12_s1/lit-llama-adapter-finetuned.pth"
    adapter_path_s2 = "./out/adapter/alpaca/lit_llama_v1_alpaca_v12_s2/lit-llama-adapter-finetuned.pth"

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

    # 读取测试集
    val_dataset_path_s1 = "./data/alpaca/llama_vln_parsed_response_v12_val_s1.json"
    val_dataset_path_s2 = "./data/alpaca/llama_vln_parsed_response_v12_val_s2.json"
    input_prompt_s1 = get_prompt_type_v4(prompt_type="s1")
    input_prompt_s2 = get_prompt_type_v4(prompt_type="s2")
    val_dict_list_s1 = json.load(open(val_dataset_path_s1))
    val_dict_list_s2 = json.load(open(val_dataset_path_s2))
    output_dict_s1_list = []
    output_dict_s2_list = []
    for val_id, val_dict_one_s1 in enumerate(tqdm(val_dict_list_s1)):
        input_instruction_s1 = val_dict_one_s1["instruction"]
        input_s1 = val_dict_one_s1["input"]
        output_gt_s1 = val_dict_one_s1["output"]

        val_dict_one_s2 = val_dict_list_s2[val_id]
        input_instruction_s2 = val_dict_one_s2["instruction"]
        input_s2 = val_dict_one_s2["input"]
        output_gt_s2 = val_dict_one_s2["output"]

        sample_s1 = {"instruction": input_instruction_s1, "input": input_s1}
        prompt_s1 = generate_prompt(sample_s1)
        sample_s2 = {"instruction": input_instruction_s2, "input": input_s2}
        prompt_s2 = generate_prompt(sample_s2)

        encoded_s1 = tokenizer.encode(prompt_s1, bos=True, eos=False, device=model_s1.device)
        encoded_s2 = tokenizer.encode(prompt_s2, bos=True, eos=False, device=model_s2.device)
        y_s1 = generate(model_s1, encoded_s1, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
        y_s2 = generate(model_s2, encoded_s2, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)

        model_s1.reset_cache()
        model_s2.reset_cache()
        output_s1 = tokenizer.decode(y_s1)
        output_s1 = output_s1.split("### Response:")[1].strip()
        output_s2 = tokenizer.decode(y_s2)
        output_s2 = output_s2.split("### Response:")[1].strip()

        output_dict_one_s1 = {}
        output_dict_one_s1["instruction"] = input_instruction_s1
        output_dict_one_s1["input"] = input
        output_dict_one_s1["output_gt"] = output_gt_s1
        output_dict_one_s1["output"] = output_s1

        output_dict_one_s2 = {}
        output_dict_one_s2["instruction"] = input_instruction_s2
        output_dict_one_s2["input"] = input
        output_dict_one_s2["output_gt"] = output_gt_s2
        output_dict_one_s2["output"] = output_s2

        output_dict_s1_list.append(output_dict_one_s1)
        output_dict_s2_list.append(output_dict_one_s2)

    # 进行保存
    with open("llama_output_response_v12_val_s1.json", 'w') as f:
        json_str = json.dumps(output_dict_s1_list, indent=2)
        f.write(json_str)
        f.write('\n')

    with open("llama_output_response_v12_val_s2.json", 'w') as f:
        json_str = json.dumps(output_dict_s2_list, indent=2)
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



