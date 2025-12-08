"""
使用规划GT、巡航GT
仅用来过滤生成数据集
"""
import numpy as np
from prior import LazyJsonDataset
import gzip
import prior
import json
from tqdm import tqdm
from utils.thor_env_code_v2 import ThorEnvCode
from utils.procthor_config import Config as proc_Config
from utils.target_object import Target_Object


# 加载具体场景
def load_dataset() -> prior.DatasetDict:
    """Load the houses dataset."""
    print(
        "[AI2-THOR WARNING] There has been an update to ProcTHOR-10K that must be used with AI2-THOR version 5.0+. To use the new version of ProcTHOR-10K, please update AI2-THOR to version 5.0+ by running:\n"
        "    pip install --upgrade ai2thor\n"
        "Alternatively, to downgrade to the old version of ProcTHOR-10K, run:\n"
        '   prior.load_dataset("procthor-10k", revision="ab3cacd0fc17754d4c080a3fd50b18395fae8647")'
    )
    data = {}
    for split, size in [("train", 10_000), ("val", 1_000), ("test", 1_000)]:
        with gzip.open(f"{split}.jsonl.gz", "r") as f:
            houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
        data[split] = LazyJsonDataset(
            data=houses, dataset="procthor-dataset", split=split
        )
    return prior.DatasetDict(**data)


def perspective_camera_view(controller):
    # 默认视角为0.0
    normal_view_angle = 0.0
    # 先获取当前视角的角度
    cameraHorizon = controller.last_event.metadata["agent"]["cameraHorizon"]
    if int(cameraHorizon) - normal_view_angle < 0:
        # 需要进行look_down
        event = controller.step(action="LookDown", degrees=normal_view_angle - int(cameraHorizon))
    elif int(cameraHorizon) - normal_view_angle > 0:
        # 需要进行look_up
        event = controller.step(action="LookUp", degrees=int(cameraHorizon) - normal_view_angle)
    else:
        normal_view_angle = 30


def main():
    val_file_path = "./data/spaced_parse_instruction_easy_v8_train.json"
    instruction_dict_list = json.load(open(val_file_path))
    # 加载环境
    args = proc_Config
    controller_env = ThorEnvCode(args, use_CloudRendering=True)
    # dataset = prior.load_dataset("procthor-10k")
    dataset = load_dataset()
    dataset_train = dataset["train"]

    global_target_object = Target_Object()
    interaction_response_dict_list = []
    success_number = 0
    # -------------------main loop--------------------------
    for val_dict_id, val_dict_one in enumerate(tqdm(instruction_dict_list)):
        interaction_response_dict = {}
        response_action_list = val_dict_one["output"]
        task_type = val_dict_one["task_type"]
        pddl_params = val_dict_one["pddl_params"]
        house_id = int(val_dict_one["house_id"])
        interaction_response_dict["house_id"] = house_id
        interaction_response_dict["index"] = val_dict_id
        rgb_frame, depth_frame = controller_env.load_scene(house_name=dataset_train[house_id])
        interaction_response_one_list = []
        global_target_object.reset_inter_object()
        # 按照顺序执行所有动作
        for action_dict_one in response_action_list:
            interaction_response_one = {}
            action = action_dict_one["action"]
            target_name = action_dict_one["arg"]
            interaction_response_one["action"] = action
            interaction_response_one["arg"] = target_name
            # 生成候选交互对象
            candidate_position_list, candidate_target_obj_id_list = controller_env.get_candidate_by_name(target_name, action_name=action)
            global_target_object.reset(candidate_position_list, candidate_target_obj_id_list)
            test_time = 0
            while True:
                target_nav_position, target_object_id = global_target_object.get_target(target_name)
                if target_nav_position is None:
                    break
                if len(target_nav_position) <= 0:
                    break
                if test_time >= 5:
                    break
                if "goto" in action.lower() or target_object_id is None:
                    controller_env.perspective_camera_view()
                    print("target_nav_position: ", target_nav_position)
                    # 先解析路径, 然后执行传送
                    nav_action, path = controller_env.parse_nav_action(target_nav_position)
                    teleport_nav_position = path[-1]
                    # 传送过去
                    is_arrive = controller_env.teleport_nav_new(teleport_nav_position)
                    # if is_arrive:
                    #     break
                    # else:
                    #     test_time += 1
                    break
                else:

                    is_near = controller_env.check_is_near_target(target_object_id)
                    if not is_near:
                        # path_distance = controller_env.goto_location_nav(target_nav_position)
                        is_arrive = controller_env.goto_location_nav_teleport(target_nav_position)

                    # 环视寻找物体
                    is_find = controller_env.check_target_frame(target_object_id)
                    # 直接check target_id 是否在open_able且close的容器中
                    isin_recept = controller_env.check_target_is_recept(target_object_id)
                    # 还是要判断是否被看到
                    candidate_index_list = controller_env.get_candidate_id_by_object_id(candidate_target_obj_id_list)
                    seen_object_id = [candidate_target_obj_id_list[can_one] for can_one in candidate_index_list]
                    if isin_recept and target_object_id not in seen_object_id:
                        force = False
                    else:
                        force = True
                    rgb, depth_frame, mask_list, info_dict = controller_env.execAction(action=[
                        action], target_arg=target_object_id.split("|")[0],
                        target_position=target_nav_position, target_object_id=target_object_id, force=force)
                    # 判断是否成功
                    # if "PutObject" in [controller_env.last_event.metadata["lastAction"]]:

                    if controller_env.last_event.metadata["lastActionSuccess"]:
                        break
                    test_time += 1
                    global_target_object.update_target()
            # 直接记录交互结果
            interaction_response_one["action_success"] = controller_env.last_event.metadata["lastActionSuccess"]
            interaction_response_one["action_response"] = controller_env.last_event.metadata["errorMessage"]
            interaction_response_one["target_object_id"] = target_object_id
            global_target_object.add_inter_object(target_object_id)
            interaction_response_one_list.append(interaction_response_one)
        # 衡量整体任务是否成功
        try:
            # print(controller_env.cleaned_objects)
            instruction_success = controller_env.check_task(task_type=task_type, task_label_dict=pddl_params)
            if instruction_success:
                success_number += 1
            interaction_response_dict["instruction_success"] = instruction_success
            interaction_response_dict["action_list"] = interaction_response_one_list
            interaction_response_dict_list.append(interaction_response_dict)
            interaction_response_dict["pddl_fail"] = False
        except:
            interaction_response_dict["instruction_success"] = False
            interaction_response_dict["action_list"] = interaction_response_one_list
            interaction_response_dict["pddl_fail"] = True
            interaction_response_dict_list.append(interaction_response_dict)
            continue
    # 进行保存
    with open("./log_file/parsed_spaced_parse_instruction_easy_v8_train.json", 'w') as f:
        json_str = json.dumps(interaction_response_dict_list, indent=2)
        f.write(json_str)
        f.write('\n')

    # 输出结果
    print("success: ", success_number / len(instruction_dict_list))

if __name__ == "__main__":
    main()





