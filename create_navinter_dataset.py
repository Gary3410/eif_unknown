import gzip
import random

from prior import LazyJsonDataset

import json
from tqdm import tqdm
import blosc
import numpy as np
import prior

from utils.thor_env_code_v2 import ThorEnvCode
from utils.procthor_config import Config as proc_Config
import os
import cv2
import pickle


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


def teleport_nav_new(controller, target_position):
    is_arrive = False
    agent_position_dict = controller.last_event.metadata["agent"]["position"]
    target_nav_position = dict(x=target_position[0], y=agent_position_dict["y"], z=target_position[-1])
    for i in range(4):
        event = controller.step(action="Teleport", position=target_nav_position)
        if controller.last_event.metadata["lastActionSuccess"]:
            is_arrive = True
            break
        else:
            event = controller.step(action="RotateRight", forceAction=True)
    return is_arrive


def get_pickupable_object(controller):
    event = controller.last_event
    pickupable_object_list = []
    for obj_ind in range(len(event.metadata["objects"])):
        obj_name = event.metadata["objects"][obj_ind]["objectId"].split("|")[0]
        pickupable = event.metadata["objects"][obj_ind]["pickupable"]
        obj_dict_ind = event.metadata["objects"][obj_ind]["objectId"]
        object_position_dict = event.metadata["objects"][obj_ind]["position"]
        object_position_tuple = (object_position_dict["x"], object_position_dict["z"])
        if pickupable:
            dict_one = dict(object_id=obj_dict_ind,
                        object_name=obj_name,
                        position=object_position_tuple)
            pickupable_object_list.append(dict_one)
    return pickupable_object_list


def get_receptacle_object(controller):
    event = controller.last_event
    target_object_list = ["Fridge", "Microwave"]
    receptacle_object_list = []
    for obj_ind in range(len(event.metadata["objects"])):
        obj_name = event.metadata["objects"][obj_ind]["objectId"].split("|")[0]
        receptacle = event.metadata["objects"][obj_ind]["receptacle"]
        obj_dict_ind = event.metadata["objects"][obj_ind]["objectId"]
        object_position_dict = event.metadata["objects"][obj_ind]["position"]
        object_position_tuple = (object_position_dict["x"], object_position_dict["z"])
        if receptacle and obj_name in target_object_list:
            dict_one = dict(object_id=obj_dict_ind,
                object_name=obj_name,
                position=object_position_tuple)
            receptacle_object_list.append(dict_one)
    return receptacle_object_list


def get_surround_point(object_position):
    surround_point_list = []
    bias_mat = np.asarray([[0.5, 0],
                           [0, 0.5],
                           [-0.5, 0],
                           [0, -0.5]])
    object_position = np.asarray(object_position)
    for i in range(4):
        surround_point_list.append(object_position + bias_mat[i])

    response_dict = {"surround_point_list": surround_point_list}
    return response_dict


def save_step_image(input_dict):
    # 读取base配置 ------------------------------
    rgb_save_base_path = input_dict["rgb_save_base_path"]
    mask_save_base_path = input_dict["mask_save_base_path"]
    info_dict_save_base_path = input_dict["info_dict_save_base_path"]
    depth_save_base_path = input_dict["depth_save_base_path"]

    controller = input_dict["controller"]
    house_id = input_dict["house_id"]

    nav_point_list = input_dict["nav_point_list"]
    agent_position = input_dict["target_nav_point"]
    object_index = input_dict["object_index"]
    nav_index = input_dict["nav_index"]
    target_name = input_dict["target_name"]
    is_find = input_dict["is_find"]
    # -----------------------------------------

    last_event = controller.last_event
    # rgb images
    rgb_image = last_event.cv2img  # BGR
    # depth 不用预处理
    depth_image = last_event.depth_frame
    # mask
    instance_masks_dict = last_event.instance_masks
    # bbox [x_min, y_min, x_max, y_max]
    bbox_dict = last_event.instance_detections2D

    # 采集pose信息
    fov = last_event.metadata["fov"]
    cameraHorizon = last_event.metadata["agent"]["cameraHorizon"]
    camera_world_xyz = list(last_event.metadata["agent"]["position"].values())
    camera_world_xyz[1] = camera_world_xyz[1] + 0.675
    camera_world_xyz = np.asarray(camera_world_xyz)
    rotation = last_event.metadata["agent"]["rotation"]['y']
    info_dict = dict(fov=fov, cameraHorizon=cameraHorizon, camera_world_xyz=camera_world_xyz.tolist(), rotation=rotation)
    info_dict["camera_height"] = 1.576

    mask_list = []
    label_dict = {}
    # 对齐label与mask {key:id, bbox:[x1, y1, x2, y2], class_name:str}
    for object_Id, mask_one in instance_masks_dict.items():
        mask_one = np.asarray(mask_one).astype(np.float32)
        if "wall" in object_Id.lower():
            class_name = "wall"
        elif "floor" in object_Id.lower():
            class_name = "floor"
        else:
            class_name = object_Id.split("|")[0]
        if object_Id in bbox_dict.keys():
            bbox_one = np.asarray(bbox_dict[object_Id])
        else:
            bbox_one = np.asarray([0, 0, 0, 0])

        mask_list.append(mask_one[None, :, :])
        label_dict[len(mask_list) - 1] = dict(bbox=bbox_one.tolist(), class_name=class_name)

    mask_list = np.concatenate(mask_list, axis=0)
    assert len(label_dict.keys()) == mask_list.shape[0]
    info_dict["label_info"] = label_dict
    info_dict["house_id"] = house_id
    # add keys
    info_dict["nav_point_list"] = nav_point_list
    info_dict["agent_position"] = agent_position
    info_dict["target_name"] = target_name
    info_dict["is_find"] = is_find

    # 进行保存
    save_base_name = "house_" + str(house_id) + "_" + str(object_index) + "_" + str(nav_index)
    rgb_save_path_one = os.path.join(rgb_save_base_path, save_base_name + ".png")
    depth_save_path_one = os.path.join(depth_save_base_path, save_base_name + ".npz")
    mask_save_path_one = os.path.join(mask_save_base_path, save_base_name + ".npz")
    info_dict_save_path_one = os.path.join(info_dict_save_base_path, save_base_name + ".pkl")

    cv2.imwrite(rgb_save_path_one, rgb_image)
    np.savez_compressed(depth_save_path_one, depth_image=depth_image)
    np.savez_compressed(mask_save_path_one, mask=mask_list)
    with open(info_dict_save_path_one, 'wb') as f:
        pickle.dump(info_dict, f)


def main():
    # 加载配置文件
    total_cat2idx_path = "./utils/total_cat2idx.json"
    total_cat2idx = json.load(open(total_cat2idx_path))
    # 增加wall
    total_cat2idx["wall"] = 95
    total_cat2idx["floor"] = 96

    # 配置参数
    args = proc_Config
    # dataset = prior.load_dataset("procthor-10k")
    dataset = load_dataset()
    dataset_train = dataset["train"]
    controller_env = ThorEnvCode(args, use_CloudRendering=True)

    # 设定保存路径
    base_save_path = "./vision_dataset/nav_inter_dataset"
    rgb_save_base_path = os.path.join(base_save_path, "rgb")
    mask_save_base_path = os.path.join(base_save_path, "mask")
    info_dict_save_base_path = os.path.join(base_save_path, "info")
    depth_save_base_path = os.path.join(base_save_path, "depth")

    if not os.path.exists(rgb_save_base_path):
        os.makedirs(rgb_save_base_path)
    if not os.path.exists(mask_save_base_path):
        os.makedirs(mask_save_base_path)
    if not os.path.exists(info_dict_save_base_path):
        os.makedirs(info_dict_save_base_path)
    if not os.path.exists(depth_save_base_path):
        os.makedirs(depth_save_base_path)

    save_object_number = 0
    save_recep_number = 0
    for house_index, house_one in enumerate(tqdm(dataset_train)):
        # 开始遍历所有场景
        if save_object_number > 20000 and save_recep_number > 3000:
        # if save_object_number > 1 and save_recep_number >= 1:
            break

        try:
            rgb_frame, depth_frame = controller_env.load_scene(house_name=house_one)
        except:
            continue

        # 获取可抓取的小物体
        pickupable_object_list = get_pickupable_object(controller_env)
        # 获取微波炉与冰箱容器
        receptacle_object_list = get_receptacle_object(controller_env)
        object_index = 0

        # 开始遍历小物体
        if save_object_number < 20000:
            for pickup_object_one in tqdm(pickupable_object_list):
                target_position = pickup_object_one["position"]
                target_name = pickup_object_one["object_name"]
                target_object_id = pickup_object_one["object_id"]

                # 根据position生成4个候选点
                surround_point_list = get_surround_point(target_position)["surround_point_list"]
                # 移动并环视寻找
                for nav_index, nav_point in enumerate(surround_point_list):
                    is_label_choice = False
                    controller_env.perspective_camera_view()
                    nav_action, path = controller_env.parse_nav_action(nav_point)
                    is_arrive = teleport_nav_new(controller_env, path[-1])
                    is_find = controller_env.check_target_frame(target_object_id)  # 使用id进行检索
                    if is_find:
                        is_label_choice = True
                    # 进行保存
                    input_dict = {"rgb_save_base_path": rgb_save_base_path,
                                  "mask_save_base_path": mask_save_base_path,
                                  "info_dict_save_base_path": info_dict_save_base_path,
                                  "depth_save_base_path": depth_save_base_path,
                                  "controller": controller_env,
                                  "house_id": house_index,
                                  "nav_point_list": surround_point_list,
                                  "target_nav_point": nav_point,
                                  "object_index": object_index,
                                  "nav_index": nav_index,
                                  "target_name": target_name,
                                  "is_find": is_label_choice}
                    save_step_image(input_dict)
                object_index += 1
                save_object_number += 1
        if save_recep_number < 3000:
            # 开始遍历容器
            for receptacle_object_one in tqdm(receptacle_object_list):
                target_position = receptacle_object_one["position"]
                target_name = receptacle_object_one["object_name"]
                target_object_id = receptacle_object_one["object_id"]

                # 打开容器
                action = dict(action="OpenObject",
                    objectId=target_object_id,
                    forceAction=True)
                controller_env.step(action)

                # check其中是否有物体, 如果没有, 需要手动放置一下
                target_object_dict = next(obj for obj in controller_env.last_event.metadata["objects"]
                        if obj["objectId"] == target_object_id)
                receptacleObjectIds = target_object_dict["receptacleObjectIds"]
                if len(receptacleObjectIds) <= 0:
                    # 随机抓取物体放进去
                    pickup_object_id = random.choice(pickupable_object_list)["object_id"]
                    event = controller_env.step(
                        action="PickupObject",
                        objectId=pickup_object_id,
                        forceAction=True
                    )
                    # 放入容器中
                    action = dict(action="PutObject",
                        objectId=target_object_id,
                        forceAction=True,
                        placeStationary=True)
                    controller_env.step(action)
                receptacleObjectIds = target_object_dict["receptacleObjectIds"]
                if receptacleObjectIds is None:
                    continue
                if receptacleObjectIds is not None:
                    if len(receptacleObjectIds) <= 0:
                        continue
                # 根据position生成4个候选点
                surround_point_list = get_surround_point(target_position)["surround_point_list"]
                # 移动并环视寻找
                for nav_index, nav_point in enumerate(surround_point_list):
                    is_label_choice = False
                    controller_env.perspective_camera_view()
                    nav_action, path = controller_env.parse_nav_action(nav_point)
                    is_arrive = teleport_nav_new(controller_env, path[-1])
                    is_find = controller_env.check_target_frame(receptacleObjectIds[0])  # 使用id进行检索
                    if is_find:
                        is_label_choice = True
                    if not is_find:
                        # 需要提供容器视角
                        is_find = controller_env.check_target_frame(target_object_id)

                    # 进行保存
                    input_dict = {"rgb_save_base_path": rgb_save_base_path,
                                  "mask_save_base_path": mask_save_base_path,
                                  "info_dict_save_base_path": info_dict_save_base_path,
                                  "depth_save_base_path": depth_save_base_path,
                                  "controller": controller_env,
                                  "house_id": house_index,
                                  "nav_point_list": surround_point_list,
                                  "target_nav_point": nav_point,
                                  "object_index": object_index,
                                  "nav_index": nav_index,
                                  "target_name": target_name,
                                  "is_find": is_label_choice}
                    save_step_image(input_dict)
                object_index += 1
                save_recep_number += 1

if __name__ == "__main__":
    main()