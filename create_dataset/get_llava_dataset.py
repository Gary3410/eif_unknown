import numpy
import os
from tqdm import tqdm
from torch.utils.data import random_split
import torch
import json

seed = 42
base_path = "./vision_dataset/nav_inter_dataset"
json_file = os.path.join(base_path, "label")
feature_file = os.path.join(base_path, "vision_feature_blosc")
json_file_list = sorted(os.listdir(json_file))
train_split_size = int(len(json_file_list) * 0.8)
test_split_size = len(json_file_list) - train_split_size
train_save_path = os.path.join(base_path, "train_nav_inter.json")
test_save_path = os.path.join(base_path, "test_nav_inter.json")

train_set, test_set = random_split(
    json_file_list,
    lengths=(train_split_size, test_split_size),
    generator=torch.Generator().manual_seed(seed),
)
train_set, test_set = list(train_set), list(test_set)

llava_id = 0
llava_json_list = []
for file_one in tqdm(train_set):
    llava_json_one = {}
    json_file_one = json.load(open(os.path.join(json_file, file_one)))
    llava_id_str = str(llava_id)
    padded_str = llava_id_str.zfill(12)
    llava_json_one["id"] = padded_str
    llava_json_one["image"] = file_one.split(".")[0] + ".blosc"
    llava_json_one["conversations"] = []
    human_value = json_file_one["human_input"]
    gpt_value = json_file_one["response"]
    llava_json_one["conversations"].append({"from": "human", "value": human_value})
    llava_json_one["conversations"].append({"from": "gpt", "value": gpt_value})
    llava_json_list.append(llava_json_one)
    llava_id = llava_id + 1

with open(train_save_path, 'w') as f:
    json_str = json.dumps(llava_json_list, indent=2)
    f.write(json_str)
    f.write('\n')

llava_id = 0
llava_json_list = []
for file_one in tqdm(test_set):
    llava_json_one = {}
    json_file_one = json.load(open(os.path.join(json_file, file_one)))
    llava_id_str = str(llava_id)
    padded_str = llava_id_str.zfill(12)
    llava_json_one["id"] = padded_str
    llava_json_one["image"] = file_one.split(".")[0] + ".blosc"
    llava_json_one["conversations"] = []
    human_value = json_file_one["human_input"]
    gpt_value = json_file_one["response"]
    llava_json_one["conversations"].append({"from": "human", "value": human_value})
    llava_json_one["conversations"].append({"from": "gpt", "value": gpt_value})
    llava_json_list.append(llava_json_one)
    llava_id = llava_id + 1

with open(test_save_path, 'w') as f:
    json_str = json.dumps(llava_json_list, indent=2)
    f.write(json_str)
    f.write('\n')
