import os
import json
import numpy as np
from tqdm import tqdm
import prior
from prior import LazyJsonDataset
import cv2
import gzip
from utils.procthor_config import Config as proc_Config
from utils.thor_env_code_v2 import ThorEnvCode


def perspective_camera_view(controller):
    # 默认视角为30
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


def teleport_nav_new(controller, position_tuple):
    is_arrive = False
    target_position = {"x": position_tuple[0], "y": position_tuple[1], "z": position_tuple[2]}
    for i in range(4):
        event = controller.step(action="Teleport", position=target_position)
        if controller.last_event.metadata["lastActionSuccess"]:
            is_arrive = True
            break
        else:
            event = controller.step(action="RotateRight", forceAction=True)
    return is_arrive

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


def get_nearest_recv_object(controller_env):
    agent_position_dict = controller_env.last_event.metadata["agent"]["position"]
    # robot_position_tuple = (robot_position_dict["x"], robot_position_dict["y"], robot_position_dict["z"])
    # 得到所有容器
    receptacle_object_list = [obj for obj in controller_env.last_event.metadata["objects"] if obj["receptacle"]]
    distance_list = []
    for receptacle_object_one in receptacle_object_list:
        position_dict_one = receptacle_object_one["position"]
        # distance = np.sqrt((agent_position_dict["x"] - position_dict_one["x"]) ** 2 +
        #                    (agent_position_dict["z"] - position_dict_one["z"]) ** 2)
        position_tuple_one = (position_dict_one["x"], position_dict_one["y"], position_dict_one["z"])
        _, path = controller_env.parse_nav_action(position_tuple_one)
        distance = (len(path) - 1) * 0.25
        distance_list.append(distance)
    nearest_index = np.argmin(np.asarray(distance_list))
    nearest_objectId = receptacle_object_list[nearest_index]["objectId"]
    return nearest_objectId, receptacle_object_list[nearest_index]


def get_surround_point(object_position_tuple):
    surround_point_list = []
    bias_mat = np.asarray([[0.25, 0, 0],
                           [0, 0, 0.25],
                           [-0.25, 0, 0],
                           [0, 0, -0.25]])
    object_position = np.asarray(object_position_tuple)
    for i in range(4):
        surround_point_list.append(object_position + bias_mat[i])
    return surround_point_list


def main():
    base_file_path = "/data1/wzy/vision_dataset/test_find_obejct.json"
    result_file = json.load(open(base_file_path))
    fail_scene_file = result_file["fail"]
    house_name_list = [file_one.split("_")[0].split("-")[-1] for file_one in fail_scene_file]
    # 去重
    house_name_list = list(set(house_name_list))
    dataset_json_version = "spaced_parse_instruction_easy_v8_train.json"
    parsed_response_list = json.load(open(os.path.join(os.getcwd(), "data", dataset_json_version)))
    args = proc_Config
    # dataset = prior.load_dataset("procthor-10k")
    dataset = load_dataset()
    dataset_train = dataset["train"]
    controller_env = ThorEnvCode(args, use_CloudRendering=True)
    house_id2dict = {}
    total_number = 0
    success_number = 0
    success_list = []
    fail_list = []
    base_save_file = "/data1/wzy/vision_dataset/test_open_find_object"
    for parsed_response_id, parsed_response_one in enumerate(tqdm(parsed_response_list)):
        house_id_one = parsed_response_one["house_id"]
        house_id2dict[house_id_one] = parsed_response_one
    # 开始进行遍历
    for house_id in tqdm(house_name_list):
        house_dict_one = house_id2dict[int(house_id)]
        house_one = dataset_train[int(house_id)]
        rgb_frame, depth_frame = controller_env.load_scene(house_name=house_one)
        output_list = house_dict_one["output"]
        for output_one_id, output_one in enumerate(output_list):
            perspective_camera_view(controller_env)
            target_object_name = output_one["arg"]
            candidate_position_list, candidate_target_obj_id_list = controller_env.get_candidate_by_name(target_object_name)

            for candidate_index, candidate_target_obj_id in enumerate(candidate_target_obj_id_list):
                perspective_camera_view(controller_env)
                target_position = candidate_position_list[candidate_index]
                nav_action, path = controller_env.parse_nav_action(target_position)
                is_arrive = teleport_nav_new(controller_env, path[-1])
                total_number += 1
                if is_arrive:
                    # 环视寻找物体
                    save_name = "house-" + str(house_id) + "_" + str(output_one_id) + "_" + candidate_target_obj_id.split("|")[0] + ".png"
                    is_find = controller_env.check_target_frame(candidate_target_obj_id)
                    if is_find:
                        # 保存图像
                        rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
                        # 获取bbox
                        bbox = controller_env.last_event.instance_detections2D[candidate_target_obj_id]

                        save_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        cv2.rectangle(save_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                            color=(0, 0, 255), thickness=2)
                        cv2.imwrite(os.path.join(base_save_file, save_name), save_rgb)
                        success_number += 1
                        success_list.append(save_name)
                    else:
                        # 启动打开物体脚本
                        nearest_objectId, obejct_data = get_nearest_recv_object(controller_env)
                        # 生成巡航坐标
                        object_position_dict = obejct_data["position"]
                        object_position_tuple = (object_position_dict["x"], object_position_dict["y"], object_position_dict["z"])
                        # nav_action, path = controller_env.parse_nav_action(object_position_tuple)
                        # 打开物体
                        open_action = dict(action="OpenObject", objectId=nearest_objectId, forceAction=True)
                        event = controller_env.step(open_action)
                        # 4点遍历巡航
                        surround_point_list = get_surround_point(object_position_tuple)
                        # 移动并环视寻找
                        for nav_point in surround_point_list:
                            perspective_camera_view(controller_env)
                            nav_action, path = controller_env.parse_nav_action(nav_point)
                            is_arrive = teleport_nav_new(controller_env, path[-1])
                            if is_arrive:
                                # 环视寻找物体
                                save_name = "house-" + str(house_id) + "_" + str(output_one_id) + "_" + candidate_target_obj_id.split("|")[0] + ".png"
                                is_find = controller_env.check_target_frame(candidate_target_obj_id)
                                if is_find:
                                    # 保存图像
                                    rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
                                    # 获取bbox
                                    bbox = controller_env.last_event.instance_detections2D[candidate_target_obj_id]

                                    save_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                                    cv2.rectangle(save_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                        color=(0, 0, 255), thickness=2)
                                    cv2.imwrite(os.path.join(base_save_file, save_name), save_rgb)
                                    success_number += 1
                                    success_list.append(save_name)
                                    break
                                else:
                                    fail_list.append(save_name)
            print("now success: ", success_number / total_number)

    # 输出结果
    print("success: ", success_number / total_number)
    result_dict = {}
    result_dict["success"] = success_list
    result_dict["fail"] = fail_list
    with open("/data1/wzy/vision_dataset/test_open_find_object.json", 'w') as f:
        json_str = json.dumps(result_dict, indent=2)
        f.write(json_str)
        f.write('\n')


if __name__ == "__main__":
    main()
