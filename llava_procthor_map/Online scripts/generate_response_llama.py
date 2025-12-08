import torch
from generate import generate
from scripts.prepare_alpaca import generate_prompt
import random

class Planner_LLaMA(object):
    def __init__(self, llama_model_s1, llama_model_s2, llama_args_dict, llama_tokenizer):
        self.llama_model_s1 = llama_model_s1
        self.llama_model_s2 = llama_model_s2
        self.llama_args_dict = llama_args_dict
        self.llama_tokenizer = llama_tokenizer
        self.frontiers_dict = None

    def get_llama_response(self, input, stage="s1"):
        max_new_tokens = self.llama_args_dict["max_new_tokens"]
        temperature = self.llama_args_dict["temperature"]
        top_k = self.llama_args_dict["top_k"]
        prompt = generate_prompt(input)
        if stage == "s1":
            encoded = self.llama_tokenizer.encode(prompt, bos=True, eos=False, device=self.llama_model_s1.device)
            y = generate(
                self.llama_model_s1,
                idx=encoded,
                max_seq_length=max_new_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_id=self.llama_tokenizer.eos_id
            )
        else:
            encoded = self.llama_tokenizer.encode(prompt, bos=True, eos=False, device=self.llama_model_s2.device)
            y = generate(
                self.llama_model_s2,
                idx=encoded,
                max_seq_length=max_new_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_id=self.llama_tokenizer.eos_id
            )
        output = self.llama_tokenizer.decode(y)
        output = output.split("### Response:")[1].strip()
        return output

    def parse_llama_s1_response(self, response):
        s1_output_list = response.split("\n")
        current_action = s1_output_list[1]
        return current_action

    def parse_llama_s2_response(self, response):
        single_step_action_dict = response.split("Specific single-step action: ")[-1]
        s2_action = single_step_action_dict.split(", target object")[0].split("action: ")[-1]
        target_name = single_step_action_dict.split(", target object: ")[-1].split(", target position")[0]
        # s2_target_position = single_step_action_dict.split("target position: ")[-1]
        return s2_action, target_name