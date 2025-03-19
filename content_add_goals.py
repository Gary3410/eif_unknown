import gzip
import json
import numpy as np
import os
import inspect
from tqdm import tqdm
from copy import deepcopy


def load_json_gz(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)  # 将解压的JSON内容加载为字典或列表
    return data


def save_json_gz(file_path, data):
    with gzip.open(file_path, 'wt', encoding="utf-8") as f:
        json.dump(data, f)

# 获取文件路径
base_content_path = "/home/wzy/workplace/llava_procthor/data/datasets/ovmm/train/content"
base_content_file_list = os.listdir(base_content_path)
save_content_path = "/home/wzy/workplace/llava_procthor/data/datasets/ovmm/train/content_nav"

# add goals
for content_file_name in tqdm(base_content_file_list):
    content_file_path_one = os.path.join(base_content_path, content_file_name)
    content_one = load_json_gz(content_file_path_one)

    new_content_data = {"config": content_one["config"],
                "obj_category_to_obj_category_id": content_one["obj_category_to_obj_category_id"],
                "recep_category_to_recep_category_id": content_one["recep_category_to_recep_category_id"]}
    episodes = content_one["episodes"]

    new_episode_list = []
    for episode_one in episodes:
        new_episode_one = deepcopy(episode_one)
        # new_episode_one = {}

        new_episode_one["goals"] = [
            {'position': [2.2896811962127686, 0.11950381100177765, 1.97636604309082], 'radius': None}]
        # new_episode_one["episode_id"] = episode_one["episode_id"]
        # new_episode_one["scene_id"] = episode_one["scene_id"]
        # new_episode_one["start_position"] = episode_one["start_position"]
        # new_episode_one["start_rotation"] = episode_one["start_rotation"]
        # new_episode_one["info"] = episode_one["info"]

        new_episode_list.append(new_episode_one)
    assert len(new_episode_list) == len(episodes)
    new_content_data["episodes"] = new_episode_list
    new_content_save_path = os.path.join(save_content_path, content_file_name)
    save_json_gz(new_content_save_path, new_content_data)
