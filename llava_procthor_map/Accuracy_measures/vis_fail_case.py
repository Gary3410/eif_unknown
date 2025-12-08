import os
import pickle
import numpy as np
import json

val_file_path = "./result_file/llava_s1_s2_vln_parsed_response_procthor_detic_v11_easy_small"
object_not_found_number = 0
interaction_failures_number = 0
planning_failures_number = 0
other_failures_number = 0
success_number = 0
seg_wrong_number = 0
receptacle_error_number = 0

file_list = sorted(os.listdir(val_file_path))
for file_one in file_list:
    file_one_path = os.path.join(val_file_path, file_one)
    with open(file_one_path, "rb") as f:
        info_dict = pickle.load(f)
    # print(info_dict.keys())
    step_action_list = info_dict["step_action_list"]
    success = info_dict["success"]
    # if success:
    #     print(info_dict["instruction"])
    if not success:
        final_s2_response = step_action_list[-1]["s2_response"]
        final_s1_response = step_action_list[-1]["s1_input"]
        if final_s1_response is None:
            final_s1_response = "asdadsadadsadadadsadsadsadsadsadasdsadasdasd"
        if final_s2_response is None:
            final_s2_response = []
        target = step_action_list[-1]["target"]
        object_list = step_action_list[-1]["s1_input"].split("\n")[1]
        done_action = step_action_list[-1]["s1_input"].split("\n")[-1]
        final_action = step_action_list[-1]["action"]
        # print(step_action_list[-1])
        # exit()

        if len(step_action_list) < 2 or "step" not in done_action.lower():
            # object not found
            if target in object_list:
                seg_wrong_number += 1
            else:
                object_not_found_number += 1
        else:
            print(final_s1_response)
            print(final_s2_response)
            print("--------------------")
            continue
    #     elif "action_response" in step_action_list[-1] and step_action_list[-1]["action_response"] != '':
    #         interaction_failures_number += 1
    #         # print(step_action_list[-1]["action_response"])
    #     elif "GotoLocation" in final_action and len(final_s1_response) > 20:
    #         # print(step_action_list[-1])
    #         # print("----------------------")
    #         if target in object_list:
    #             seg_wrong_number += 1
    #         elif "fridge" in done_action.lower():
    #             receptacle_error_number += 1
    #             # print(done_action)
    #         else:
    #             object_not_found_number += 1
    #     elif len(step_action_list) > 2 and "Step" in final_s1_response and "GotoLocation" not in final_action:
    #         planning_failures_number += 1
    #         print(final_s1_response)
    #         print("--------------------")
    #     else:
    #         other_failures_number += 1
    # else:
    #     success_number += 1

print("object_not_found_number: ", object_not_found_number)
print("seg_wrong_number: ", seg_wrong_number)
print("interaction_failures_number: ", interaction_failures_number)
print("planning_failures_number: ", planning_failures_number)
print("other_failures_number: ", other_failures_number)
print("receptacle_error_number: ", receptacle_error_number)
print("success: ", success_number)
print(success_number + other_failures_number + interaction_failures_number + planning_failures_number + object_not_found_number + \
    receptacle_error_number + seg_wrong_number)