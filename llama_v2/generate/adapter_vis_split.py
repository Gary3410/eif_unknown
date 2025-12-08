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
import json
from tqdm import tqdm
import random


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
    # reload新参数
    adapter_path_s1 = Path("/home/wzy/workplace/lit-gpt/out/adapter_split/iter-031999-ckpt_s1.pth")
    adapter_path_s2 = Path("/home/wzy/workplace/lit-gpt/out/adapter_split/iter-047999-ckpt_s2.pth")

    # 配置具体的精度
    precision = precision or get_default_supported_precision(training=False)
    # 配置具体参数
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

    # 具体模型进行读取
    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy, plugins=plugins)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)
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
        # model_s1 = GPT(config)
        model_s2 = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.perf_counter()
    # 这是总体的checkpoint
    checkpoint_s1 = lazy_load(checkpoint_path)
    checkpoint_s2 = lazy_load(checkpoint_path)
    adapter_checkpoint_s1 = lazy_load(adapter_path_s1)
    adapter_checkpoint_s2 = lazy_load(adapter_path_s2)
    checkpoint_s1.update(adapter_checkpoint_s1.get("model", adapter_checkpoint_s1))
    checkpoint_s2.update(adapter_checkpoint_s2.get("model", adapter_checkpoint_s2))
    # model_s1.load_state_dict(checkpoint_s1)
    model_s2.load_state_dict(checkpoint_s2)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    # model_s1.eval()
    model_s2.eval()
    # model_s1 = fabric.setup(model_s1)
    model_s2 = fabric.setup(model_s2)

    # tokenizer进行复用
    tokenizer = Tokenizer(checkpoint_dir)

    # 增加具体可视化指令
    llama_dict = json.load(open("/home/wzy/workplace/lit-gpt/data/alpaca/llama_vln_parsed_response_100_v2.json"))
    respone_s1_list = []
    respone_s2_list = []
    vis_number = 0
    for i in tqdm(range(50)):
        llama_one = random.choice(llama_dict)
        prompt = llama_one["instruction"]
        input = llama_one["input"]
        output_gt = llama_one["output"]
        respone_one = {}
        sample = {"instruction": prompt, "input": input}
        respone_one["instruction"] = prompt
        respone_one["input"] = input
        respone_one["output_gt"] = output_gt

        prompt = generate_prompt(sample)
        encoded = tokenizer.encode(prompt, device=fabric.device)
        prompt_length = encoded.size(0)
        max_returned_tokens = prompt_length + max_new_tokens

        with fabric.init_tensor():
            # set the max_seq_length to limit the memory usage to what we need
            # model_s1.max_seq_length = max_returned_tokens
            model_s2.max_seq_length = max_returned_tokens
            # enable the kv cache
            # model_s1.set_kv_cache(batch_size=1)
            model_s2.set_kv_cache(batch_size=1)

        t0 = time.perf_counter()
        if "Current Action" in output_gt:
            # y = generate(model_s1, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
            continue
        elif "Specific single-step action" in output_gt:
            y = generate(model_s2, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
        else:
            continue
        t = time.perf_counter() - t0

        output = tokenizer.decode(y)
        output = output.split("### Response:")[1].strip()

        respone_one["output"] = output
        if "Current Action" in output_gt:
            respone_s1_list.append(respone_one)
        if "Specific single-step action" in output_gt:
            respone_s2_list.append(respone_one)
    # 进行保存
    with open("llama_finetune_s1.json", 'w') as f:
        json_str = json.dumps(respone_s1_list, indent=2)
        f.write(json_str)
        f.write('\n')

    with open("llama_finetune_s2.json", 'w') as f:
        json_str = json.dumps(respone_s2_list, indent=2)
        f.write(json_str)
        f.write('\n')

    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)

if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)