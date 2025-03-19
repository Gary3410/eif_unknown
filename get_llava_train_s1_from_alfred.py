import os
import json
import random
from tqdm import tqdm
from utils.object_constants import OBJECTS, OBJECTS_LANG


def check_step_planning(planning_action_list):
    is_check = True
    for planning_action_one in planning_action_list:
        if "turn left" in planning_action_one.lower() or "turn right" in planning_action_one.lower():
            is_check = False
            break
    return is_check


def cover_2_s1_format(planning_action_list):
    # Task completed.
    new_action_list = []
    for index, planning_action_one in enumerate(planning_action_list):
        planning_action_one = planning_action_one.capitalize()

        # add object_name replace
        planning_action_one = replace_object_name(planning_action_one)

        planning_action_one = "Step " + str(index + 1) + ". " + planning_action_one
        new_action_list.append(planning_action_one)
    # add finish prompt
    finish_action = "Step " + str(len(new_action_list)) + ". " + "Task completed"
    new_action_list.append(finish_action)
    return new_action_list


def list_2_str(object_class_name):
    input_str = "["
    for object_name_one in object_class_name:
        new_str = object_name_one + ", "
        input_str = input_str + new_str
    input_str = input_str.strip(" ").strip(",") + "]"
    return input_str


def get_action_str(action_list):
    action_str = "\n"
    for action_one in action_list:
        action_str = action_str + action_one + "\n"
    return action_str.rstrip("\n")


def get_object_list_str(object_name_list):
    object_name_list_str = "["
    for object_name_one in object_name_list:
        object_name_list_str = object_name_list_str + object_name_one + ", "
    object_name_list_str = object_name_list_str.rstrip(", ")
    object_name_list_str = object_name_list_str + "]"
    return object_name_list_str


def replace_object_name(input_str):
    for object_lang_one in OBJECTS_LANG:
        if object_lang_one in input_str:
            object_lang_index = OBJECTS_LANG.index(object_lang_one)
            object_replace_name = OBJECTS[object_lang_index]
            input_str = input_str.replace(object_lang_one, object_replace_name)
    return input_str


def main():
    parsed_action_list = []
    base_file_path = "./data/json_2.1.0/train"
    save_path = "./data/llava_vln_parsed_response_100_v8_easy_train_alfred.json"
    task_type_file_list = os.listdir(base_file_path)
    for task_type_file_name_one in tqdm(task_type_file_list[:1]):
        task_type_file_path_one = os.path.join(base_file_path, task_type_file_name_one)
        traj_file_list = os.listdir(task_type_file_path_one)
        for traj_file_name_one in traj_file_list:
            traj_file_path_one = os.path.join(task_type_file_path_one,  traj_file_name_one, "traj_data.json")
            traj_file = json.load(open(traj_file_path_one))
            # 获取planning
            action_planning = traj_file["plan"]
            # print(action_planning.keys())
            object_list = []
            high_pddl = action_planning["high_pddl"]
            for high_pddl_one in high_pddl:
                planner_action = high_pddl_one["planner_action"]
                if "objectId" in planner_action:
                    object_list.append(planner_action["objectId"].split("|")[0])
            object_list = list(set(object_list))
            # print(object_list)
            # 获取场景物体
            scene_object = traj_file["scene"]["object_poses"]
            scene_object_list = [object_one["objectName"].split("_")[0] for object_one in scene_object]
            scene_object_list = list(set(scene_object_list))
            # print(scene_object_list)

            # 获取指令与step by step planning
            turk_annotations = traj_file["turk_annotations"]["anns"]
            for anns_one in turk_annotations:
                high_descs = anns_one["high_descs"]
                task_desc = anns_one["task_desc"].capitalize()
                task_desc = replace_object_name(task_desc)
                is_check = check_step_planning(high_descs)
                if is_check:
                    scene_object_list_copy = scene_object_list.copy()
                    random.shuffle(scene_object_list_copy)
                    s1_format_action = cover_2_s1_format(high_descs)
                    parsed_action_list.append({"instruction": task_desc,
                                               "object_list": scene_object_list_copy,
                                               "planning": s1_format_action,
                                               "planning_object": object_list})

            # print(parsed_action_list)
    # 开始填充为llava_s1 format
    parse_dict_list = []
    for train_dict_one in parsed_action_list:
        instruction = train_dict_one["instruction"]
        object_input_list = train_dict_one["object_list"]
        planning_action_list = train_dict_one["planning"]
        object_list = train_dict_one["planning_object"]
        done_action = "\n"

        for action_index, action_one in enumerate(planning_action_list):
            parse_dict_one = {}
            object_input_list_copy = object_input_list.copy()
            random.shuffle( object_input_list_copy)
            object_name_str = list_2_str( object_input_list_copy)
            # 生成human_value
            human_instruction_input = "Instruction: " + instruction + "\n" + "Object List: " + object_name_str + \
                                      "\n" + "Done Actions: " + done_action.rstrip("\n")
            if action_index < len(planning_action_list) - 1:
                action_str = get_action_str(planning_action_list[action_index + 1:])
            else:
                action_str = "\n"

            # 生成response
            # 增加交互物体
            target_name_list_str = get_object_list_str(object_list)
            target_planning = "Planning Target: " + target_name_list_str
            # response_str = "Current Action:" + "\n" + planning + \
            #                "\n" + "Planning Actions:" + action_str
            response_str = "Current Action:" + "\n" + action_one + \
                           "\n" + "Planning Actions:" + action_str + "\n" + target_planning
            parse_dict_one["human"] = human_instruction_input
            parse_dict_one["response"] = response_str
            parse_dict_list.append(parse_dict_one)
            done_action = done_action + action_one + "\n"

    # 转化为llava格式
    llava_json_list = []
    if "train" in save_path:
        random.shuffle(parse_dict_list)

    llava_id = 0
    for parse_dict_one in parse_dict_list:
        llava_json_one = {}
        human_value = parse_dict_one["human"]
        gpt_value = parse_dict_one["response"]
        llava_id_str = str(llava_id)
        padded_str = llava_id_str.zfill(12)
        llava_json_one["id"] = padded_str
        llava_json_one["image"] = None
        llava_json_one["conversations"] = []

        llava_json_one["conversations"].append({"from": "human", "value": human_value})
        llava_json_one["conversations"].append({"from": "gpt", "value": gpt_value})

        llava_json_list.append(llava_json_one)
        llava_id += 1

    # 进行保存
    print(len(llava_json_list))
    with open(save_path, 'w') as f:
        json_str = json.dumps(llava_json_list, indent=2)
        f.write(json_str)
        f.write('\n')

if __name__ == '__main__':
    main()
