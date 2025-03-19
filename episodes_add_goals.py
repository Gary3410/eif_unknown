import gzip
import json
import numpy as np
import os
import inspect
from tqdm import tqdm
from copy import deepcopy

# 定义文件路径
file_path = "/home/wzy/workplace/llava_procthor/data/datasets/ovmm/train/episodes_ori.json.gz"
# print(os.path.dirname(inspect.getabsfile(inspect.currentframe())))
# 打开并读取.gz文件
with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    data = json.load(f)  # 将解压的JSON内容加载为字典或列表

episodes = data["episodes"]
new_data = {"config": data["config"],
            "obj_category_to_obj_category_id": data["obj_category_to_obj_category_id"],
            "recep_category_to_recep_category_id": data["recep_category_to_recep_category_id"]}

new_episode_list = []
for episode_one in tqdm(episodes):
    new_episode_one = deepcopy(episode_one)
    # new_episode_one = {}

    new_episode_one["goals"] = [{'position': [2.2896811962127686, 0.11950381100177765, 1.97636604309082], 'radius': None}]
    # new_episode_one["episode_id"] = episode_one["episode_id"]
    # new_episode_one["scene_id"] = episode_one["scene_id"]
    # new_episode_one["start_position"] = episode_one["start_position"]
    # new_episode_one["start_rotation"] = episode_one["start_rotation"]
    # new_episode_one["info"] = episode_one["info"]

    new_episode_list.append(new_episode_one)

assert len(new_episode_list) == len(episodes)
new_data["episodes"] = new_episode_list

# 进行保存
save_path = os.path.join("/home/wzy/workplace/llava_procthor/data/datasets/ovmm/train", "episodes.json.gz")
with gzip.open(save_path, 'wt', encoding="utf-8") as f:
    json.dump(new_data, f)