import numpy as np
import os
import json
from tqdm import tqdm
from llava.conversation import conv_templates, SeparatorStyle
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from transformers import TextStreamer

import random

class Planner(object):
    def __init__(self, llava_model_s2, llava_args, llava_tokenizer, llava_model_s1, llava_model_nav_inter=None):
        self.llava_model_s2 = llava_model_s2
        self.llava_args = llava_args
        self.llava_tokenizer = llava_tokenizer
        self.llava_model_s1 = llava_model_s1
        self.frontiers_dict = None
        self.surround_point_dict = None

        # add nav_inter
        self.llava_model_nav_inter = llava_model_nav_inter

    def get_llava_response_s1(self, input, feature):
        conv = conv_templates[self.llava_args.conv_mode].copy()
        image_tensor = torch.from_numpy(feature)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.llava_model_s1.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.llava_model_s1.device, dtype=torch.float16)
        conv.append_message(conv.roles[0], input)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.llava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.llava_model_s1.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(self.llava_tokenizer, skip_prompt=True, skip_special_tokens=True)
        image_size = image_tensor.size

        with torch.inference_mode():
            # output_ids = self.llava_model_s1.generate(
            #     input_ids,
            #     images=image_tensor,
            #     image_sizes=[image_size],
            #     do_sample=True if self.llava_args.temperature > 0 else False,
            #     temperature=self.llava_args.temperature,
            #     max_new_tokens=self.llava_args.max_new_tokens,
            #     streamer=streamer,
            #     use_cache=True)
            output_ids = self.llava_model_s1.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if self.llava_args.temperature > 0 else False,
                temperature=self.llava_args.temperature,
                max_new_tokens=self.llava_args.max_new_tokens,
                use_cache=True)

        # outputs = self.llava_tokenizer.decode(output_ids[0]).strip()
        outputs = self.llava_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(outputs)
        return outputs

    def get_llava_response_s2(self, input, feature):
        conv = conv_templates[self.llava_args.conv_mode].copy()
        if isinstance(feature, np.ndarray):
            image_tensor = torch.from_numpy(feature)
        else:
            image_tensor = feature
        if type(image_tensor) is list:
            image_tensor = [image.to(self.llava_model_s2.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.llava_model_s2.device, dtype=torch.float16)
        conv.append_message(conv.roles[0], input)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.llava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.llava_model_s2.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(self.llava_tokenizer, skip_prompt=True, skip_special_tokens=True)
        image_size = image_tensor.size

        with torch.inference_mode():
            # output_ids = self.llava_model_s2.generate(
            #     input_ids,
            #     images=image_tensor,
            #     image_sizes=[image_size],
            #     do_sample=True if self.llava_args.temperature > 0 else False,
            #     temperature=self.llava_args.temperature,
            #     max_new_tokens=self.llava_args.max_new_tokens,
            #     streamer=streamer,
            #     use_cache=True)
            output_ids = self.llava_model_s2.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if self.llava_args.temperature > 0 else False,
                temperature=self.llava_args.temperature,
                max_new_tokens=self.llava_args.max_new_tokens,
                use_cache=True)

        # outputs = self.llava_tokenizer.decode(output_ids[0]).strip()
        outputs = self.llava_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs

    def get_llava_response_nav_inter(self, input, feature):
        conv = conv_templates[self.llava_args.conv_mode].copy()
        if isinstance(feature, np.ndarray):
            image_tensor = torch.from_numpy(feature)
        else:
            image_tensor = feature
        if type(image_tensor) is list:
            image_tensor = [image.to(self.llava_model_nav_inter.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.llava_model_nav_inter.device, dtype=torch.float16)
        conv.append_message(conv.roles[0], input)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.llava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.llava_model_nav_inter.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(self.llava_tokenizer, skip_prompt=True, skip_special_tokens=True)
        image_size = image_tensor.size

        with torch.inference_mode():
            # output_ids = self.llava_model_s2.generate(
            #     input_ids,
            #     images=image_tensor,
            #     image_sizes=[image_size],
            #     do_sample=True if self.llava_args.temperature > 0 else False,
            #     temperature=self.llava_args.temperature,
            #     max_new_tokens=self.llava_args.max_new_tokens,
            #     streamer=streamer,
            #     use_cache=True)
            output_ids = self.llava_model_nav_inter.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if self.llava_args.temperature > 0 else False,
                temperature=self.llava_args.temperature,
                max_new_tokens=self.llava_args.max_new_tokens,
                use_cache=True)

        # outputs = self.llava_tokenizer.decode(output_ids[0]).strip()
        outputs = self.llava_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs

    def trans2_llava_input(self, instruction, frontiers_label_dict_list, robot_position):
        question = "\n"
        self.frontiers_dict = {}
        for frontiers_one in frontiers_label_dict_list:
            choose = frontiers_one["choose"]
            label = frontiers_one["label"]
            centroid = frontiers_one["centroid"]
            self.frontiers_dict[choose] = centroid  # 用于解析具体值
            choose_one = choose + ". " + "[" + str(int(centroid[1])) + ", " + str(int(centroid[-1])) + "]"
            question = question + choose_one + "\n"
        if len(frontiers_label_dict_list) < 1:
            add_choose = "A" + ". " + "[" + str(int(robot_position[1])) + ", " + str(int(robot_position[-1])) + "]"
            question = question + add_choose + "\n"
        # 没有frontiers, 但仍然需要解析
        question = question.rstrip("\n")
        human_value = instruction + " and where to go?" + question
        return human_value

    def trans2_llava_nav_inter_input(self, target_name, surround_point_list):
        question = "\n"
        self.surround_point_dict = {}
        start_ascii = 65
        prompt = "I need move to " + target_name + " , which point I should go?"
        for point_index, point_one in enumerate(surround_point_list):
            choose_str = chr(start_ascii + point_index)
            choice_one = np.around(point_one, 2)
            choose_one_str = choose_str + ". " + "[" + str(choice_one[0]) + ", " + str(choice_one[1]) + "]"
            question = question + choose_one_str + "\n"
            self.surround_point_dict[choose_str] = choice_one
        question = question.rstrip("\n")
        human_value = prompt + question
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

    def parse_llava_nav_inter_response(self, response):
        response = response.strip("</s>")
        response = response.strip("<s> ")
        nav_choose = response.split("The answer is ")[-1].split(", ")[0]
        if nav_choose in self.surround_point_dict.keys():
            nav_point = self.surround_point_dict[nav_choose]
        else:
            nav_point = None
        return nav_point

    def parse_llava_s1_response(self, response):
        response = response.strip("</s>")
        response = response.strip("<s> ")
        response_list = response.split("\n")
        if len(response_list) > 1:
            current_action = response_list[1]
        else:
            current_action = None
        return current_action

    def parse_long_clip_prompt(self, object_list):
        image_prompt = "The image captures one of either"
        if isinstance(object_list, str):
            if "[" in object_list:
                object_list = object_list[1:-1]
            object_name_list = object_list.split(", ")
        else:
            object_name_list = object_list
        object_list_str = " "
        for object_index, object_one in enumerate(object_name_list):
            if object_index != len(object_list) - 1:
                object_list_str = object_list_str + "a " + object_one.lower() + ", "
            else:
                object_list_str = object_list_str + "or " + "a " + object_one.lower()
        long_clip_prompt = image_prompt + object_list_str + "."
        return long_clip_prompt
