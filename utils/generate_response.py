import numpy as np
import os
import json
from tqdm import tqdm
from llava.conversation import conv_templates, SeparatorStyle
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from transformers import TextStreamer

from generate import generate
from scripts.prepare_alpaca import generate_prompt
import random

class Planner(object):
    def __init__(self,
                 llava_model=None,
                 llama_model=None,
                 llava_args=None,
                 llama_args_dict=None,
                 llama_tokenizer=None,
                 llava_tokenizer=None,
                 llava_model_s1=None):
        self.llava_model = llava_model
        self.llama_model = llama_model
        self.llava_args = llava_args
        self.llama_args_dict = llama_args_dict
        self.llama_tokenizer = llama_tokenizer
        self.llava_tokenizer = llava_tokenizer
        self.llava_model_s1 = llava_model_s1
        self.frontiers_dict = None

    def get_llava_response(self, input, feature, model_type="s2"):
        conv = conv_templates[self.llava_args.conv_mode].copy()
        image_tensor = torch.from_numpy(feature)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.llava_model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.llava_model.device, dtype=torch.float16)
        conv.append_message(conv.roles[0], input)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.llava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.llava_model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(self.llava_tokenizer, skip_prompt=True, skip_special_tokens=True)
        image_size = image_tensor.size
        if model_type == "s1":
            with torch.inference_mode():
                output_ids = self.llava_model_s1.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if self.llava_args.temperature > 0 else False,
                    temperature=self.llava_args.temperature,
                    max_new_tokens=self.llava_args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True)
        elif model_type == "s2":
            with torch.inference_mode():
                output_ids = self.llava_model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if self.llava_args.temperature > 0 else False,
                    temperature=self.llava_args.temperature,
                    max_new_tokens=self.llava_args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True)

        outputs = self.llava_tokenizer.decode(output_ids[0]).strip()
        return outputs

    def get_llama_response(self, input):
        max_new_tokens = self.llama_args_dict["max_new_tokens"]
        temperature = self.llama_args_dict["temperature"]
        top_k = self.llama_args_dict["top_k"]
        prompt = generate_prompt(input)
        encoded = self.llama_tokenizer.encode(prompt, bos=True, eos=False, device=self.llama_model.device)
        y = generate(
            self.llama_model,
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

    def trans2_llava_input(self, instruction, frontiers_label_dict_list):
        question = "\n"
        self.frontiers_dict = {}
        for frontiers_one in frontiers_label_dict_list:
            choose = frontiers_one["choose"]
            label = frontiers_one["label"]
            centroid = frontiers_one["centroid"]
            self.frontiers_dict[choose] = centroid  # 用于解析具体值
            choose_one = choose + ". " + "[" + str(int(centroid[1])) + ", " + str(int(centroid[-1])) + "]"
            question = question + choose_one + "\n"
        question = question.rstrip("\n")
        human_value = instruction + " and where to go?" + question
        return human_value

    def parse_llava_s2_response(self, response):
        if "finish" in response.lower():
            return [0, 0, 0], "End", None
        response = response.strip("</s>")
        response = response.strip("<s> ")
        response_answer = response.split("\n")[0]
        action_answer = response.split("\n")[-1]
        nav_choose = response_answer.split("The answer is ")[-1].strip(".")
        action = action_answer.split("I will ")[-1].split(" the")[0]
        if "so I" in action_answer:
            target = action_answer.split("the ")[-1].split(" so")[0]
        else:
            target = action_answer.split("the ")[-1].strip(".")
        if nav_choose in self.frontiers_dict.keys():
            nav_point = self.frontiers_dict[nav_choose]
        else:
            if len(list(self.frontiers_dict.keys())) > 0:
                random_choose = random.choice(list(self.frontiers_dict.keys()))
                nav_point = self.frontiers_dict[random_choose]
            else:
                nav_point = [0, 0, 0]
        return nav_point, action, target

    def parse_llava_s1_response(self, response):
        response = response.strip("</s>")
        response = response.strip("<s> ")
        response_list = response.split("\n")
        if len(response_list) > 1:
            current_action = response_list[1]
        else:
            current_action = None
        return current_action
