import numpy as np
from prior import LazyJsonDataset
import gzip
import prior
import json
from tqdm import tqdm
from utils.thor_env_code_v2 import ThorEnvCode
from utils.procthor_config import Config as proc_Config
from utils.target_object import Target_Object


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
        with gzip.open(f"./procthor_house/{split}.jsonl.gz", "r") as f:
            houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
        data[split] = LazyJsonDataset(
            data=houses, dataset="procthor-dataset", split=split
        )
    return prior.DatasetDict(**data)


def main():
    val_file_path = "./data/xxx.json"
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
        rgb_frame, depth_frame = controller_env.load_scene(house_name=dataset_train[house_id])
        interaction_response_dict["house_id"] = house_id
        interaction_response_dict["index"] = val_dict_id
        interaction_action_list = []
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
            # while True:
            target_nav_position, target_object_id = global_target_object.get_target(target_name)
            if target_nav_position is None:
                break
            if len(target_nav_position) <= 0:
                break
            if test_time >= 5:
                break
            if "goto" in action.lower() or target_object_id is None:
                controller_env.perspective_camera_view()
                # print("target_nav_position: ", target_nav_position)
                # 先解析路径, 然后执行传送
                nav_action, path = controller_env.parse_nav_action(target_nav_position)
                for action_one in tqdm(nav_action):
                    rgb, depth_frame, mask_list, info_dict = controller_env.to_thor_api_exec([action_one])
                    interaction_action_list.append(action_one)

            else:
                rgb, depth_frame, mask_list, info_dict = controller_env.execAction(action=[
                    action], target_arg=target_object_id.split("|")[0],
                    target_position=target_nav_position, target_object_id=target_object_id, force=True)
                interaction_action_list.append(action)

        interaction_response_dict["action_list"] = interaction_action_list
        interaction_response_dict_list.append(interaction_response_dict)

    # 进行保存
    with open("./log_file/xxx.json", 'w') as f:
        json_str = json.dumps(interaction_response_dict_list, indent=2)
        f.write(json_str)
        f.write('\n')


if __name__ == "__main__":
    main()