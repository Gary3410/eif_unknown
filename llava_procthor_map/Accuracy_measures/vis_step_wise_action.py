import os
import json
import pickle

base_path = "./result_file/llava_s1_s2_vln_parsed_response_procthor_detic_v12_hard_large_attention"

file_list = os.listdir(base_path)
file_list.sort(key=lambda x: int(x.split("_")[0]))
for file_one in file_list:
    # print(file_one)
    file_one_path = os.path.join(base_path, file_one)
    with open(file_one_path, "rb") as f:
        info_dict = pickle.load(f)
    success = info_dict["success"]
    # print(info_dict.keys())
    # print(info_dict["exec_action_dict_list"]["task_id"])
    # print("instruction: ", info_dict["instruction"])
    # exit()
    if success:
        print("house_id: ", file_one)
        print("instruction: ", info_dict["instruction"])
        step_action_list = info_dict["step_action_list"]
        for step_action_one in step_action_list:
            action = step_action_one["action"]
            target = step_action_one["target"]
            print(action + ": " + target)
        print("=============================")
