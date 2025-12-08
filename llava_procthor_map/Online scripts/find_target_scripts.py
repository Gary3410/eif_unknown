from utils.thor_env_code_v2 import ThorEnvCode
import prior
import pickle
from utils.procthor_config import Config as proc_Config
import json
import os
from tqdm import tqdm
import cv2
from prior import LazyJsonDataset
import gzip


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


def main():
    dataset_json_version = "spaced_parse_instruction_easy_v8_train.json"
    args = proc_Config
    # dataset = prior.load_dataset("procthor-10k")
    dataset = load_dataset()
    dataset_train = dataset["train"]
    controller_env = ThorEnvCode(args, use_CloudRendering=True)
    base_save_file = "/data1/wzy/vision_dataset/find_object_test"
    success_number = 0
    success_list = []
    fail_list = []
    total_number = 0
    # 读取base_file
    parsed_response_list = json.load(open(os.path.join(os.getcwd(), "data", dataset_json_version)))
    for parsed_response_id, parsed_response_one in enumerate(tqdm(parsed_response_list)):
        # 具体需要执行的动作序列 [action, arg, planning]
        output_list = parsed_response_one["output"]
        house_id = parsed_response_one["house_id"]
        house_one = dataset_train[house_id]
        rgb_frame, depth_frame = controller_env.load_scene(house_name=house_one)
        for output_one_id, output_one in enumerate(tqdm(output_list)):
            perspective_camera_view(controller_env)
            target_object_name = output_one["arg"]
            candidate_position_list, candidate_target_obj_id_list = controller_env.get_candidate_by_name(target_object_name)
            # 寻找每个物体
            for candidate_index, candidate_target_obj_id in enumerate(candidate_target_obj_id_list):
                perspective_camera_view(controller_env)
                target_position = candidate_position_list[candidate_index]
                nav_action, path = controller_env.parse_nav_action(target_position)
                target_nav_position = path[-1]
                # 先移动到目标位置
                is_arrive = teleport_nav_new(controller_env, path[-1])
                total_number += 1
                if is_arrive:
                    # 环视寻找物体
                    save_name = "house-" + str(house_id) + "_" + str(output_one_id) + ".png"
                    is_find = controller_env.check_target_frame(candidate_target_obj_id)
                    if is_find:
                        # 保存图像
                        rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
                        # save_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        # cv2.imwrite(os.path.join(base_save_file, "success", save_name), save_rgb)
                        success_number += 1
                        success_list.append(save_name)
                    else:
                        rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
                        # save_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        # cv2.imwrite(os.path.join(base_save_file, "fail", save_name), save_rgb)
                        fail_list.append(save_name)
                    break
            print("now success: ", success_number / total_number)

    # 输出结果
    print("success: ", success_number / total_number)
    result_dict = {}
    result_dict["success"] = success_list
    result_dict["fail"] = fail_list
    with open("/data1/wzy/vision_dataset/test_find_obejct.json", 'w') as f:
        json_str = json.dumps(result_dict, indent=2)
        f.write(json_str)
        f.write('\n')




if __name__ == "__main__":
    main()
