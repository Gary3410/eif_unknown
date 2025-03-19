import os
import pickle
import numpy as np
import json

"""
文件dict
easy_long_large: spaced_parse_instruction_easy_long_v14_val.json
hard_large: spaced_parse_instruction_hard_v15_val.json
"""

# wrong_list = [73, 74, 126, 128, 130, 131, 132, 134]
wrong_list = []
base_path = "/home/wzy/workplace/llava_procthor/log_file/llava_s1_s2_nav_inter_vln_parsed_response_procthor_detic_v12_easy_small_attention_without_3D_0703"
label_base_path = "./data/spaced_parse_instruction_easy_v12_val.json"

file_list = sorted(os.listdir(base_path))
label_file_list = json.load(open(label_base_path))
label_id2action_dict = {}
label_id2action_dict_add = {}
new_file_list = []
for house_index, label_file_one in enumerate(label_file_list):
    house_id = int(label_file_one["house_id"])
    label_id2action_dict[str(house_index) + "_" + str(house_id)] = label_file_one
    # if "house_result_name" in label_file_one:
    #     new_file_list.append(label_file_one["house_result_name"] + ".pkl")
    #     label_id2action_dict_add[label_file_one["house_result_name"]] = label_file_one
# print(file_list)
if len(new_file_list) > 0:
    file_list = new_file_list

instruction_number = 0
success_number = 0
distance_all = 0
pred_id2dict = {}
plwsr_list = []
gc_list = []
plwgc_list = []
for file_one in file_list[:20]:
    file_one_path = os.path.join(base_path, file_one)
    file_index = int(file_one.split("_")[0])
    if file_index in wrong_list:
        continue
    instruction_number += 1
    with open(file_one_path, "rb") as f:
        info_dict = pickle.load(f)
    success = info_dict["success"]
    distance = info_dict["distance"]
    house_id = info_dict["house_id"]
    house_index = file_one.split("_")[0]
    if len(new_file_list) > 0:
        label_dict_one = label_id2action_dict_add[file_one.split(".")[0]]
    else:
        label_dict_one = label_id2action_dict[house_index + "_" + str(house_id)]
    label_action_number = len(label_dict_one["output"]) - 1
    # pred_action_number = max(0, len(info_dict["step_action_list"]))
    pred_action_number = max(0, len(info_dict["step_action_list"]))
    add_action_number = 0
    # for step_action_one in info_dict["step_action_list"]:
    #     if "add_action" in step_action_one:
    #         add_action_number += 1
    # pred_action_number = pred_action_number + add_action_number

    plw = label_action_number / max(pred_action_number, label_action_number)
    if "ts" in info_dict and "s" in info_dict:
        ts = info_dict["ts"]
        s = info_dict["s"]
        if ts <= 0:
            gc_one = 0
        else:
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
        # print(file_one)
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
print("plwsr: ",  np.sum(np.asarray(plwsr_list)) / instruction_number)
print("gc: ", np.sum(np.asarray(gc_list)) / instruction_number)
print("plwgc: ", np.sum(np.asarray(plwgc_list)) / instruction_number)


