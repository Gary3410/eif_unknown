import os
import pickle
import numpy as np
import json


def parse_success(val_file_path, gt_file_path):

    label_file_list = json.load(open(gt_file_path))
    label_id2action_dict = {}
    for label_file_one in label_file_list:
        house_id = int(label_file_one["house_id"])
        label_id2action_dict[house_id] = label_file_one

    file_list = sorted(os.listdir(val_file_path))

    # 进行统计
    instruction_number = len(file_list)
    success_number = 0
    distance_all = 0
    pred_id2dict = {}
    plwsr_list = []
    gc_list = []
    plwgc_list = []
    success_house_id_list = []

    for file_one in file_list:
        file_one_path = os.path.join(val_file_path, file_one)
        with open(file_one_path, "rb") as f:
            info_dict = pickle.load(f)
        success = info_dict["success"]
        distance = info_dict["distance"]
        house_id = info_dict["house_id"]
        label_dict_one = label_id2action_dict[int(house_id)]
        label_action_number = len(label_dict_one["output"]) - 1
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
            # print("==========================")
            plwsr_list.append(1 * plw)
            success_house_id_list.append(int(house_id))
        pred_id2dict[house_id] = info_dict
        distance_all = distance_all + distance
    # 输出结果
    print("场景数量", instruction_number)
    print("success: ", success_number / instruction_number)
    print("distance: ", distance_all / instruction_number)
    print("plwsr: ", np.sum(np.asarray(plwsr_list)) / instruction_number)
    print("gc: ", np.sum(np.asarray(gc_list)) / instruction_number)
    print("plwgc: ", np.sum(np.asarray(plwgc_list)) / instruction_number)
    print("----------------------------------------")

    response_dict = {"house_dict": pred_id2dict,
                     "success_list": success_house_id_list}

    return response_dict


def parse_success_only(val_file_path):
    file_list = sorted(os.listdir(val_file_path))

    # 进行统计
    instruction_number = len(file_list)
    success_number = 0
    distance_all = 0
    pred_id2dict = {}
    plwsr_list = []
    gc_list = []
    plwgc_list = []
    success_house_id_list = []
    for file_one in file_list:
        file_one_path = os.path.join(val_file_path, file_one)
        with open(file_one_path, "rb") as f:
            info_dict = pickle.load(f)

        success = info_dict["success"]
        distance = info_dict["distance"]
        house_id = info_dict["house_id"]
        if success:
            success_number += 1
            # print(info_dict["instruction"])
            # print(info_dict["house_id"])
            # print(info_dict["done_action"])
            # print("==========================")
            success_house_id_list.append(int(house_id))
        pred_id2dict[house_id] = info_dict

    response_dict = {"house_dict": pred_id2dict,
                     "success_list": success_house_id_list}

    return response_dict


def get_low_action_list(step_action_list):
    low_action_list = ""
    for step_action_one in step_action_list:
        low_action = step_action_one["action"]
        target = step_action_one["target"]
        low_action_one = low_action + ": " + target
        low_action_list = low_action_list + low_action_one + "\n"
    low_action_list = low_action_list.rstrip("\n")
    return low_action_list


def main():
    # 设定val
    planning_gt_file_path = "./result_file/llava_s1_s2_nav_inter_vln_parsed_response_procthor_detic_v12_easy_small_attention_0630"
    # label_gt_file_path = "./data/spaced_parse_instruction_easy_v9_val.json"
    pred_file_path = "./result_file/llava_s1_s2_vln_parsed_response_procthor_detic_v13_easy_small_attention_0519"

    # pred_response_dict = parse_success(val_file_path=pred_file_path, gt_file_path=label_gt_file_path)
    # planning_response_dict = parse_success(val_file_path=planning_gt_file_path, gt_file_path=label_gt_file_path)
    pred_response_dict = parse_success_only(val_file_path=pred_file_path)
    planning_response_dict = parse_success_only(val_file_path=planning_gt_file_path)

    planning_success_list = planning_response_dict["success_list"]
    planning_house_dict = planning_response_dict["house_dict"]

    pred_success_list = pred_response_dict["success_list"]
    pred_house_dict = pred_response_dict["house_dict"]
    print(len(planning_success_list))
    print(len(pred_success_list))
    # 开始统计
    for planning_gt_index, planning_gt_house_id in enumerate(planning_success_list):
        if planning_gt_house_id not in pred_success_list:
            print(planning_gt_index)
            print("house_id: ", planning_gt_house_id)
            print("instruction: ", planning_house_dict[planning_gt_house_id]["instruction"])
            print("----------------")
            print("planning")
            print(planning_house_dict[planning_gt_house_id]["done_action"])
            # print(planning_house_dict[planning_gt_house_id]["step_action_list"])
            step_action_list = planning_house_dict[planning_gt_house_id]["step_action_list"]
            low_action_list = get_low_action_list(step_action_list)
            print(low_action_list)
            print("----------------")
            print("pred")
            print(pred_house_dict[planning_gt_house_id]["done_action"])
            step_action_list = pred_house_dict[planning_gt_house_id]["step_action_list"]
            low_action_list = get_low_action_list(step_action_list)
            print(low_action_list)
            print("=====================")

if __name__ == "__main__":
    main()

