import os
import pickle
import numpy as np
import json

base_path = "./log_file/llava_s1_s2_vln_parsed_response_procthor_maskrcnn_alfred_valid_seen"
label_base_path = "./data/json_2.1.0/valid_seen"

# 读取测试集
valid_unseen_path = "./data/json_2.1.0/valid_seen"
instruction_dict_list = []
split_dict = json.load(open("./data/splits/oct21.json"))
tests_unseen_scene_list = split_dict["valid_seen"]
for tests_unseen_scene_one in tests_unseen_scene_list:
    instruction_dict_one = {}
    task_file_name = tests_unseen_scene_one["task"]
    desc_index = tests_unseen_scene_one["repeat_idx"]
    task_file_path = os.path.join(valid_unseen_path, task_file_name, "traj_data.json")
    task_file_dict = json.load(open(task_file_path))
    instruction_list = task_file_dict['turk_annotations']['anns']
    instruction_dict_one["instruction"] = instruction_list[int(desc_index)]["task_desc"]
    instruction_dict_one["scene"] = task_file_dict['scene']["floor_plan"]
    instruction_dict_one["task_type"] = task_file_dict["task_type"]
    instruction_dict_one["pddl_params"] = task_file_dict["pddl_params"]
    instruction_dict_one["repeat_idx"] = desc_index
    # add 计算衡量指标的label
    label_action_number = len(task_file_dict["plan"]["high_pddl"])
    instruction_dict_one["label_action_number"] = label_action_number
    instruction_dict_list.append(instruction_dict_one)

success_number = 0
distance_all = 0
pred_id2dict = {}
plwsr_list = []
gc_list = []
plwgc_list = []
instruction_number = len(instruction_dict_list)

for index, instruction_dict_one in enumerate(instruction_dict_list):
    # 生成result的label
    scene = instruction_dict_one["scene"]
    repeat_idx = instruction_dict_one["repeat_idx"]
    save_name_one = str(index) + "_" + str(scene) + "_" + str(repeat_idx) + ".pkl"

    file_one_path = os.path.join(base_path, save_name_one)
    with open(file_one_path, "rb") as f:
        info_dict = pickle.load(f)

    success = info_dict["success"]
    distance = info_dict["distance"]
    house_id = info_dict["house_id"]
    label_action_number = instruction_dict_one["label_action_number"]
    pred_action_number = max(0, len(info_dict["step_action_list"]) - 1)
    plw = label_action_number / max(pred_action_number, label_action_number)
    if "ts" in info_dict and "s" in info_dict:
        ts = info_dict["ts"]
        s = info_dict["s"]
        gc_one = s / ts
        plwgc_list.append(gc_one * plw)
        gc_list.append(gc_one)
    else:
        gc_one = 0
        plwgc_list.append(gc_one * plw)
        gc_list.append(gc_one)

    if success:
        success_number += 1
        # print(info_dict["instruction"])
        # print(info_dict["house_id"])
        # print(info_dict["done_action"])
        # print(info_dict["distance"])
        # print("==========================")
        plwsr_list.append(1 * plw)
    pred_id2dict[house_id] = info_dict
    distance_all = distance_all + distance

# 输出结果
print("场景数量", instruction_number)
print("success: ", success_number / instruction_number)
print("distance: ", distance_all / instruction_number)
print("plwsr: ", np.sum(np.asarray(plwsr_list)) / instruction_number)
print("gc: ", np.sum(np.asarray(gc_list)) / instruction_number)
print("plwgc: ", np.sum(np.asarray(plwgc_list)) / instruction_number)