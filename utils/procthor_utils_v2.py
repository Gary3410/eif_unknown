import os

import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Tuple
from collections import deque
import copy
from PIL import Image
import cv2
import math
from typing import Optional, Sequence, cast
import torch
# import skimage
from skimage import morphology
import random
from utils.procthor_utils import closest_grid_point, get_position_neighbors, shortest_path, get_rotation
from utils.parse_json_utils import get_frame_pointcloud
from copy import deepcopy


def get_frame_object_list(controller_env):
    visible_obj_id_list = []
    object_id2index = {}
    object_index2id = {}
    event = controller_env.last_event
    metadata = event.metadata
    object_metadata = metadata['objects']
    for obj_ind in range(len(object_metadata)):
        obj_data_one = object_metadata[obj_ind]
        obj_id = obj_data_one["objectId"]
        object_id2index[obj_id] = obj_ind
        object_index2id[obj_ind] = obj_id
    frame_object_dict = event.instance_masks
    for frame_object_id, frame_object_one in enumerate(list(frame_object_dict.keys())):
        if frame_object_one in object_id2index.keys():
            visible_obj_id_list.append(object_id2index[frame_object_one])

    frame_object_dict = {}
    for obj_ind in visible_obj_id_list:
        obj_Id = object_metadata[obj_ind]["objectId"]
        obj_name = object_metadata[obj_ind]["objectId"].split("|")[0]
        frame_object_dict[obj_Id] = obj_name

    return frame_object_dict


def get_frame_object_list_by_properties(controller_env, property=None):
    visible_obj_id_list = []
    object_id2index = {}
    object_index2id = {}
    event = controller_env.last_event
    metadata = event.metadata
    object_metadata = metadata['objects']
    for obj_ind in range(len(object_metadata)):
        obj_data_one = object_metadata[obj_ind]
        obj_id = obj_data_one["objectId"]
        object_id2index[obj_id] = obj_ind
        object_index2id[obj_ind] = obj_id
    frame_object_dict = event.instance_masks
    for frame_object_id, frame_object_one in enumerate(list(frame_object_dict.keys())):
        if frame_object_one in object_id2index.keys():
            visible_obj_id_list.append(object_id2index[frame_object_one])

    frame_object_dict = {}
    for obj_ind in visible_obj_id_list:
        obj_Id = object_metadata[obj_ind]["objectId"]
        obj_name = object_metadata[obj_ind]["objectId"].split("|")[0]
        if property is not None:
            if object_metadata[obj_ind]["objectId"][property]:
                frame_object_dict[obj_Id] = obj_name
        else:
            frame_object_dict[obj_Id] = obj_name

    return frame_object_dict


def get_frame_object_name_sp(controller_env, target_object_name):
    visible_obj_id_list = []
    object_id2index = {}
    object_index2id = {}
    event = controller_env.last_event
    metadata = event.metadata
    object_metadata = metadata['objects']
    for obj_ind in range(len(object_metadata)):
        obj_data_one = object_metadata[obj_ind]
        obj_id = obj_data_one["objectId"]
        object_id2index[obj_id] = obj_ind
        object_index2id[obj_ind] = obj_id
    frame_object_dict = event.instance_masks
    for frame_object_id, frame_object_one in enumerate(list(frame_object_dict.keys())):
        if frame_object_one in object_id2index.keys():
            visible_obj_id_list.append(object_id2index[frame_object_one])

    target_object_id_list = []
    for obj_ind in visible_obj_id_list:
        obj_Id = object_metadata[obj_ind]["objectId"]
        obj_name = object_metadata[obj_ind]["objectId"].split("|")[0]
        if obj_name == target_object_name:
            target_object_id_list.append(obj_Id)
    return target_object_id_list


def get_receptacle_object(event):
    receptacle_object_dict = {}
    for obj_ind in range(len(event.metadata["objects"])):
        obj_name = event.metadata["objects"][obj_ind]["objectId"].split("|")[0]
        receptacle = event.metadata["objects"][obj_ind]["receptacle"]
        obj_dict_ind = event.metadata["objects"][obj_ind]["objectId"]
        if receptacle:
            receptacle_object_dict[obj_dict_ind] = obj_name
    return receptacle_object_dict


def get_openable_object(event):
    openable_object_dict = {}
    for obj_ind in range(len(event.metadata["objects"])):
        obj_name = event.metadata["objects"][obj_ind]["objectId"].split("|")[0]
        openable = event.metadata["objects"][obj_ind]["openable"]
        obj_dict_ind = event.metadata["objects"][obj_ind]["objectId"]
        if openable:
            openable_object_dict[obj_dict_ind] = obj_name
    return openable_object_dict


def get_sliceable_object(event):
    sliceable_object_dict = {}
    for obj_ind in range(len(event.metadata["objects"])):
        obj_name = event.metadata["objects"][obj_ind]["objectId"].split("|")[0]
        sliceable = event.metadata["objects"][obj_ind]["sliceable"]
        obj_dict_ind = event.metadata["objects"][obj_ind]["objectId"]
        if sliceable:
            sliceable_object_dict[obj_dict_ind] = obj_name
    return sliceable_object_dict


def get_toggleable_object(event):
    toggleable_object_dict = {}
    for obj_ind in range(len(event.metadata["objects"])):
        obj_name = event.metadata["objects"][obj_ind]["objectId"].split("|")[0]
        toggleable = event.metadata["objects"][obj_ind]["toggleable"]
        obj_dict_ind = event.metadata["objects"][obj_ind]["objectId"]
        if toggleable:
            toggleable_object_dict[obj_dict_ind] = obj_name
    return toggleable_object_dict


def create_pickup_data(controller_env, target_object_name, fail_action=False):
    pickup_success = False
    find_object = True
    event = controller_env.last_event
    # 先获取frame可以看到的物体
    frame_object_dict = get_frame_object_list(controller_env)
    # 根据target_object_name生成需要交互的物体id
    pickup_object_list = []
    for obj_id, obj_name in frame_object_dict.items():
        if obj_name == target_object_name:
            pickup_object_list.append(obj_id)
    # 在执行pickup之前, 需要确保手中物体为空
    # drop_object_acion = dict(action="DropHandObject", forceAction=True)
    # event = controller_env.step(drop_object_acion)
    # 之后开始进行尝试抓取物体
    for pickup_object_one in pickup_object_list:
        pickup_action = dict(action="PickupObject",
            objectId=pickup_object_one, forceAction=False)
        event = controller_env.step(pickup_action)
        if event.metadata["lastActionSuccess"]:
            pickup_success = True
            break
    if len(pickup_object_list) > 0:
        if not pickup_success:
            pickup_action = dict(action="PickupObject",
                objectId=pickup_object_list[0], forceAction=True)
            event = controller_env.step(pickup_action)
            if event.metadata["lastActionSuccess"]:
                pickup_success = True
    else:
        # 当前视野没有可以抓取物体
        pickup_success = False
        find_object = False
    return pickup_success, find_object


def create_put_data(controller_env, target_object_name):
    put_success = False
    object_in_hand = True
    event = controller_env.last_event
    # 生成可以放置的物体
    receptacle_object_dict = get_receptacle_object(event)
    # 先获取frame可以看到的物体
    frame_object_dict = get_frame_object_list(controller_env)
    # 根据target_object_name生成需要交互的物体id
    put_object_list = []
    for obj_id, obj_name in frame_object_dict.items():
        if obj_name == target_object_name:
            put_object_list.append(obj_id)

    for put_object_one in put_object_list:
        # 生成交互动作
        put_action = dict(action="PutObject", objectId=put_object_one, forceAction=True,
            placeStationary=True)
        event = controller_env.step(put_action)
        if event.metadata["lastActionSuccess"]:
            put_success = True
            break
    # inventoryObjects
    objects_in_hand = event.metadata['inventoryObjects']
    if len(objects_in_hand) > 0 and len(receptacle_object_dict) > 0 and len(put_object_list) > 0:
        if not put_success:
            # 启动精确放置
            event = controller_env.step(
                action="GetSpawnCoordinatesAboveReceptacle",
                objectId=put_object_list[0],
                anywhere=False
            )
            # position_list = event.metadata["actionReturn"]
            # for i in range(len(position_list)):
            #     place_point_action = dict(action="PlaceObjectAtPoint", objectId=put_object_list[0],
            #     position=position_list[i])
            #     event = controller_env.step(place_point_action)
            #     if event.metadata["lastActionSuccess"]:
            #         put_success = True
            #         break
    else:
        # 扔掉物体
        put_success = False
        object_in_hand = False
    print("放置物体: ", put_success)
    if not put_success:
        event = controller_env.step(
            action="MoveHeldObjectAhead",
            moveMagnitude=0.1,
            forceVisible=False
        )
        drop_object_acion = dict(action="DropHandObject", forceAction=True)
        event = controller_env.step(drop_object_acion)
    return put_success, object_in_hand


def create_open_data(controller_env, target_object_name):
    open_success = False
    find_object = True
    event = controller_env.last_event
    frame_object_dict = get_frame_object_list(controller_env)
    open_object_list = []
    for obj_id, obj_name in frame_object_dict.items():
        if obj_name == target_object_name:
            open_object_list.append(obj_id)
    # 生成具体open action
    for open_obj_id in open_object_list:
        open_action = dict(action="OpenObject",
            objectId=open_obj_id, forceAction=True)
        event = controller_env.step(open_action)
        if event.metadata["lastActionSuccess"]:
            open_success = True
            break
    if len(open_object_list) > 0:
        if not open_success:
            pickup_action = dict(action="OpenObject",
                objectId=open_object_list[0], forceAction=True)
            event = controller_env.step(pickup_action)
            if event.metadata["lastActionSuccess"]:
                open_success = True
    else:
        # 当前视野没有可以抓取物体
        open_success = False
        find_object = False
    return open_success, find_object


def create_close_data(controller_env, target_object_name):
    close_success = False
    find_object = True
    event = controller_env.last_event
    frame_object_dict = get_frame_object_list(controller_env)
    close_object_list = []
    for obj_id, obj_name in frame_object_dict.items():
        if obj_name == target_object_name:
            close_object_list.append(obj_id)
    # 生成具体open action
    for close_obj_id in close_object_list:
        close_action = dict(action="CloseObject",
            objectId=close_obj_id, forceAction=True)
        event = controller_env.step(close_action)
        if event.metadata["lastActionSuccess"]:
            close_success = True
            break
    if len(close_object_list) > 0:
        if not close_success:
            close_action = dict(action="CloseObject",
                objectId=close_object_list[0], forceAction=True)
            event = controller_env.step(close_action)
            if event.metadata["lastActionSuccess"]:
                close_success = True
    else:
        # 当前视野没有可以抓取物体
        close_success = False
        find_object = False
    return close_success, find_object


def create_slice_data(controller_env, target_object_name):
    slice_success = False
    find_object = True
    event = controller_env.last_event
    frame_object_dict = get_frame_object_list(controller_env)
    slice_object_list = []
    for obj_id, obj_name in frame_object_dict.items():
        if obj_name == target_object_name:
            slice_object_list.append(obj_id)
    # 生成具体动作
    for slice_obj_id in slice_object_list:
        slice_action = dict(action="SliceObject",
            objectId=slice_obj_id, forceAction=True)
        event = controller_env.step(slice_action)
        if event.metadata["lastActionSuccess"]:
            slice_success = True
            break
    # 解析具体原因
    if len(slice_object_list) > 0:
        if not slice_success:
            slice_action = dict(action="SliceObject",
                objectId=slice_object_list[0], forceAction=True)
            event = controller_env.step(slice_action)
            if event.metadata["lastActionSuccess"]:
                slice_success = True
    else:
        # 当前视野没有可以抓取物体
        slice_success = False
        find_object = False
    return slice_success, find_object


def create_toggle_data(controller_env, target_object_name):
    toggle_success = False
    find_object = True
    frame_object_dict = get_frame_object_list(controller_env)
    toggle_object_list = []
    for obj_id, obj_name in frame_object_dict.items():
        if obj_name == target_object_name:
            toggle_object_list.append(obj_id)
    # 需要提前判断物体状态
    # 生成具体动作
    for toggle_object_one in toggle_object_list:
        toggle_off_action = dict(action="ToggleObjectOff",
            objectId=toggle_object_one, forceAction=True)
        toggle_on_action = dict(action="ToggleObjectOn",
            objectId=toggle_object_one, forceAction=True)
        is_Toggled = controller_env.object_dict[toggle_object_one]["isToggled"]
        if is_Toggled:
            event = controller_env.step(toggle_off_action)
        else:
            event = controller_env.step(toggle_on_action)
        if event.metadata["lastActionSuccess"]:
            toggle_success = True
            break
    if len(toggle_object_list) > 0:
        find_object = True
    else:
        find_object = False
        toggle_success = False
    return toggle_success, find_object


def create_clean_data(controller_env, target_object_name):
    clean_success = False
    find_object = True
    frame_object_dict = get_frame_object_list(controller_env)
    clean_object_list = []
    for obj_id, obj_name in frame_object_dict.items():
        if obj_name == target_object_name:
            clean_object_list.append(obj_id)
    for clean_object_one in clean_object_list:
        clean_action = dict(action="CleanObject", objectId=clean_object_one, forceAction=True)
        event = controller_env.step(clean_action)
        if event.metadata["lastActionSuccess"]:
            clean_success = True
            break
    if len(clean_object_list) > 0:
        find_object = True
    else:
        clean_success = False
        find_object = False
    return clean_success, find_object


def is_arrive_target_point(point, target_point):
    x0 = point["x"]
    z0 = point["z"]
    x1 = target_point[0]
    z1 = target_point[2]
    if math.isclose(float(x0), float(x1), rel_tol=1e-5) and math.isclose(float(z0), float(z1), rel_tol=1e-5):
        return True
    else:
        return False


def add_vln_information(output_one, controller, target_name, initial_position=None, args=None, nav_position_tuple=None):
    # 重新进行初始化场景
    output_dict_one_list = []
    save_frame_list = []
    save_sem_map_list = []
    target_id = None
    # 初始化随机位置
    if nav_position_tuple is None:
        nav_position = output_one["nav_position"]
        nav_position_tuple = (float(nav_position[1:-1].split(", ")[0]), 0.9, float(nav_position[1:-1].split(", ")[-1]))
    is_arrive = False
    old_action = output_one["action"]
    old_arg = output_one["arg"]
    nav_time = 0
    # 巡航的主体巡航
    while not is_arrive:
        output_dict_one = {}
        event = controller.step(action="GetReachablePositions")
        positions = event.metadata["actionReturn"]
        positions_tuple = [(p["x"], p["y"], p["z"]) for p in positions]
        neighbors = get_position_neighbors(positions_tuple)
        if initial_position is None:
            while True:
                initial_position = random.choice(positions)
                event = controller.step(action="Teleport", position=initial_position, rotation=90)
                initial_rotation = event.metadata['agent']['rotation']
                is_arrive = is_arrive_target_point(initial_position, nav_position_tuple)
                if not is_arrive:
                    break
        else:
            initial_position = controller.last_event.metadata['agent']['position']
            initial_rotation = controller.last_event.metadata['agent']['rotation']
        # dict --> tuple
        initial_position_tuple = (initial_position["x"], initial_position["y"], initial_position["z"])
        # 根据init_position与Nav_position_tuple生成具体路径
        print("起点: ", initial_position_tuple)
        print("终点: ", nav_position_tuple)
        path = shortest_path(initial_position_tuple, nav_position_tuple, neighbors, positions_tuple)
        print("路径长度", len(path))
        rotation = controller.last_event.metadata['agent']['rotation']
        start_rotation = rotation['y']

        select_path_dict = {}
        # 进行环视, 用来选取最远点
        for rotating_time in range(4):
            event = controller.step(action="RotateRight", forceAction=True)
            # 更新可见点云
            current_frame_point_cloud = get_frame_pointcloud(controller.last_event)
            current_frame_point_cloud = filter_point_clouds(current_frame_point_cloud)
            path_select_id_list = []
            distance_list = []
            current_rotation = int(controller.last_event.metadata['agent']['rotation']["y"])
            select_path_dict[current_rotation] = {}
            # 生成在frame中的路径点
            for path_id, path_one in enumerate(path):
                is_within_range, distance = check_isin_frame_workplace(current_frame_point_cloud, path_one, initial_position_tuple)
                if is_within_range:
                    path_select_id_list.append(path_id)
                    distance_list.append(distance)
            distance_list = np.asarray(distance_list)
            path_select_id_list = np.asarray(path_select_id_list)
            if len(distance_list) > 0:
                max_distance = np.max(distance_list)
                select_path_id = path_select_id_list[np.argmax(distance_list)]
                select_path_dict[current_rotation]["distance"] = max_distance
                select_path_dict[current_rotation]["path_id"] = select_path_id
            else:
                select_path_dict[current_rotation]["distance"] = 0
                # 直接走向下一个点
                select_path_dict[current_rotation]["path_id"] = 1
        # 根据环视结果, 选择具体的起始状态
        chose_rotation = initial_rotation["y"]
        chose_path_id = 1
        chose_distance = 0
        for rotation_one, path_dict in select_path_dict.items():
            distance = path_dict["distance"]
            path_id = path_dict["path_id"]
            if distance > chose_distance:
                chose_path_id = path_id
                chose_rotation = rotation_one
        # 移动到具体的选择状态
        event = controller.step(action="Teleport", position=initial_position, rotation=dict(x=0, y=chose_rotation, z=0))
        assert len(path) >= 2
        next_nav_position = path[chose_path_id]
        print("单步巡航位置", next_nav_position)
        # 单次巡航开始---------------------------------------------
        # 感知巡航前的信息
        # 看见物体list
        updated_object_list = update_object_input(controller.last_event)
        agent_start_position_dict = controller.last_event.metadata["agent"]["position"]
        agent_start_position_tuple = (agent_start_position_dict["x"], agent_start_position_dict["z"])

        # 看见巡航点list
        updated_position_list = update_position_set(controller.last_event, positions_tuple)
        current_frame = controller.last_event.frame
        rgb, depth_frame = controller.get_obs()
        object_point_cloud_dict, height_map_all = controller.get_instance_point(args)
        height_map_all[-1, :, :] = 1e-5

        # 地图合并
        if controller.init_sem_map is None:
            controller.init_sem_map = height_map_all.copy()
        new_sem_pred = np.maximum(controller.init_sem_map, height_map_all)
        controller.init_sem_map = new_sem_pred
        # 新的地图
        sem_map_process = new_sem_pred.argmax(0)

        # 移动到具体位置
        target_position = {"x": next_nav_position[0], "y": next_nav_position[1], "z": next_nav_position[2]}
        for i in range(4):
            event = controller.step(action="Teleport", position=target_position)
            if controller.last_event.metadata["lastActionSuccess"]:
                break
            else:
                event = controller.step(action="RotateRight", forceAction=True)
        # 环视寻找目标
        # find_object = find_object_rotation(controller, target_name)
        find_object, target_id = find_object_rotation_id(controller, target_name, target_id)
        print("是否找到物体", find_object)
        print("目标物体ID", target_id)
        # 增加如下信息
        if find_object:
            # 如果找到物体, 进行重新规划
            target_obj_Id_list = get_frame_object_name_sp(controller, target_name)
            # 因为是看到了, 直接获取第一个就行
            obj_metadata = controller.last_event.metadata["objects"]
            # target_object_position_dict = obj_metadata[target_obj_Id_list[0]]["position"]
            if target_id not in target_obj_Id_list:
                continue
            target_object_position_dict = get_object_position_by_Id(controller, target_id)
            target_object_position_tuple = (target_object_position_dict["x"], target_object_position_dict["y"], target_object_position_dict["z"])
            # 再次规划, 判断路径终点是否相同
            agent_position_dict = controller.last_event.metadata["agent"]["position"]
            agent_position_tuple = (agent_position_dict["x"], agent_position_dict["y"], agent_position_dict["z"])
            new_path = shortest_path(agent_position_tuple, target_object_position_tuple, neighbors, positions_tuple)
            new_nav_position_tuple = new_path[-1]
            new_nav_position_tuple_xy = (np.round(new_nav_position_tuple[0], 2), np.round(new_nav_position_tuple[2], 2))
            nav_position_tuple_xy = (np.round(nav_position_tuple[0], 2), np.round(nav_position_tuple[2], 2))
            print("以前巡航目标", nav_position_tuple_xy)
            print("更新巡航目标", new_nav_position_tuple_xy)
            if not nav_position_tuple_xy == new_nav_position_tuple_xy:
                nav_position_tuple = new_nav_position_tuple

        end_rotation = int(controller.last_event.metadata['agent']['rotation']["y"])

        # 单次巡航结束--------------------------------------------
        # 补充单次巡航的信息
        # str格式的保存-------------------------------------------------------
        output_dict_one["action"] = "GotoLocation"
        output_dict_one["arg"] = old_arg
        output_dict_one["expectation"] = " "
        output_dict_one["start_position"] = "(" + str(initial_position_tuple[0]) + ", " + str(
            initial_position_tuple[2]) + ")"
        output_dict_one["nav_position"] = "(" + str(next_nav_position[0]) + ", " + str(next_nav_position[2]) + ")"
        output_dict_one["start_rotation"] = str(int(start_rotation))
        output_dict_one["end_rotation"] = str(int(end_rotation))
        output_dict_one["Agent position"] = [np.round(agent_start_position_dict["x"], 2),
                                                np.round(agent_start_position_dict["z"], 2)]
        output_dict_one["input"] = json2str(updated_object_list)
        output_dict_one["input_shorter"] = json2str_shorter(updated_object_list, target_name)
        output_dict_one["position_set"] = list2str(updated_position_list)
        output_dict_one["position_set_shorter"] = list2str_shorter(updated_position_list, next_nav_position)
        # str格式的保存-------------------------------------------------------

        # list形式保存--------------------------------------------------------
        output_dict_one["nav_position_list"] = [next_nav_position[0], next_nav_position[2]]
        output_dict_one["start_position_list"] = [initial_position_tuple[0], initial_position_tuple[2]]
        output_dict_one["input_dict"] = updated_object_list
        output_dict_one["position_set_list"] = updated_position_list
        # list形式保存-------------------------------------------------------

        # 进一步保存当前场景的图片
        output_dict_one_list.append(output_dict_one)
        save_frame_list.append(current_frame)
        save_sem_map_list.append(sem_map_process)
        nav_time = nav_time + 1

        # 判断是否到达
        # target_position是目前巡航位置
        # nav_position_tuple是最终目标位置
        is_arrive = is_arrive_target_point(target_position, nav_position_tuple)
        if nav_time > 50:
            break
        # ********************new add*************************
        # 后续增加环视寻找目标

    return output_dict_one_list, save_frame_list, save_sem_map_list, nav_position_tuple


def add_vln_information_point_cloud(output_one, controller, target_name, initial_position=None, args=None, nav_position_tuple=None):
    # 重新进行初始化场景
    output_dict_one_list = []
    save_frame_list = []
    save_sem_map_list = []
    target_id = None
    # 初始化随机位置
    if nav_position_tuple is None:
        nav_position = output_one["nav_position"]
        nav_position_tuple = (float(nav_position[1:-1].split(", ")[0]), 0.9, float(nav_position[1:-1].split(", ")[-1]))
    is_arrive = False
    old_action = output_one["action"]
    old_arg = output_one["arg"]
    nav_time = 0
    # 巡航的主体巡航
    while not is_arrive:
        output_dict_one = {}
        event = controller.step(action="GetReachablePositions")
        positions = event.metadata["actionReturn"]
        positions_tuple = [(p["x"], p["y"], p["z"]) for p in positions]
        neighbors = get_position_neighbors(positions_tuple)
        if initial_position is None:
            while True:
                initial_position = random.choice(positions)
                event = controller.step(action="Teleport", position=initial_position, rotation=90)
                initial_rotation = event.metadata['agent']['rotation']
                is_arrive = is_arrive_target_point(initial_position, nav_position_tuple)
                if not is_arrive:
                    break
        else:
            initial_position = controller.last_event.metadata['agent']['position']
            initial_rotation = controller.last_event.metadata['agent']['rotation']
        # dict --> tuple
        initial_position_tuple = (initial_position["x"], initial_position["y"], initial_position["z"])
        # 根据init_position与Nav_position_tuple生成具体路径
        print("起点: ", initial_position_tuple)
        print("终点: ", nav_position_tuple)
        path = shortest_path(initial_position_tuple, nav_position_tuple, neighbors, positions_tuple)
        print("路径长度", len(path))
        rotation = controller.last_event.metadata['agent']['rotation']
        start_rotation = rotation['y']

        select_path_dict = {}
        # 进行环视, 用来选取最远点
        for rotating_time in range(4):
            event = controller.step(action="RotateRight", forceAction=True)
            # 更新可见点云
            current_frame_point_cloud = get_frame_pointcloud(controller.last_event)
            current_frame_point_cloud = filter_point_clouds(current_frame_point_cloud)
            path_select_id_list = []
            distance_list = []
            current_rotation = int(controller.last_event.metadata['agent']['rotation']["y"])
            select_path_dict[current_rotation] = {}
            # 生成在frame中的路径点
            for path_id, path_one in enumerate(path):
                is_within_range, distance = check_isin_frame_workplace(current_frame_point_cloud, path_one, initial_position_tuple)
                if is_within_range:
                    path_select_id_list.append(path_id)
                    distance_list.append(distance)
            distance_list = np.asarray(distance_list)
            path_select_id_list = np.asarray(path_select_id_list)
            if len(distance_list) > 0:
                max_distance = np.max(distance_list)
                select_path_id = path_select_id_list[np.argmax(distance_list)]
                select_path_dict[current_rotation]["distance"] = max_distance
                select_path_dict[current_rotation]["path_id"] = select_path_id
            else:
                select_path_dict[current_rotation]["distance"] = 0
                # 直接走向下一个点
                select_path_dict[current_rotation]["path_id"] = 1
        # 根据环视结果, 选择具体的起始状态
        chose_rotation = initial_rotation["y"]
        chose_path_id = 1
        chose_distance = 0
        for rotation_one, path_dict in select_path_dict.items():
            distance = path_dict["distance"]
            path_id = path_dict["path_id"]
            if distance > chose_distance:
                chose_path_id = path_id
                chose_rotation = rotation_one
        # 移动到具体的选择状态
        event = controller.step(action="Teleport", position=initial_position, rotation=dict(x=0, y=chose_rotation, z=0))
        assert len(path) >= 2
        next_nav_position = path[chose_path_id]
        print("单步巡航位置", next_nav_position)
        # 单次巡航开始---------------------------------------------
        # 感知巡航前的信息
        # 看见物体list
        updated_object_list = update_object_input(controller.last_event)
        agent_start_position_dict = controller.last_event.metadata["agent"]["position"]
        agent_start_position_tuple = (agent_start_position_dict["x"], agent_start_position_dict["z"])

        # 看见巡航点list
        updated_position_list = update_position_set(controller.last_event, positions_tuple)
        current_frame = controller.last_event.frame
        rgb, depth_frame = controller.get_obs()
        # object_point_cloud_dict, height_map_all = controller.get_instance_point(args)
        point_cloud_list = []
        for i in range(4):
            # [n, 4]
            object_point_cloud_one = controller.get_instance_point_only(args)
            object_point_cloud_one = down_sample_point_cloud(object_point_cloud_one)
            point_cloud_list.append(object_point_cloud_one)
            event = controller.step(action="RotateRight", forceAction=True)
        object_point_cloud_dict = np.concatenate(point_cloud_list, axis=0)

        # 地图合并
        if controller.init_sem_map is None:
            controller.init_sem_map = object_point_cloud_dict.copy()
        # new_sem_pred = np.maximum(controller.init_sem_map, height_map_all)
        else:
            new_sem_pred = np.concatenate((controller.init_sem_map, object_point_cloud_dict), axis=0)
            controller.init_sem_map = new_sem_pred
        # 新的地图
        # sem_map_process = controller.init_sem_map

        # 移动到具体位置
        target_position = {"x": next_nav_position[0], "y": next_nav_position[1], "z": next_nav_position[2]}
        for i in range(4):
            event = controller.step(action="Teleport", position=target_position)
            if controller.last_event.metadata["lastActionSuccess"]:
                break
            else:
                event = controller.step(action="RotateRight", forceAction=True)
        # 环视寻找目标
        # find_object = find_object_rotation(controller, target_name)
        find_object, target_id = find_object_rotation_id(controller, target_name, target_id)
        print("是否找到物体", find_object)
        print("目标物体ID", target_id)
        # 增加如下信息
        if find_object:
            # 如果找到物体, 进行重新规划
            target_obj_Id_list = get_frame_object_name_sp(controller, target_name)
            # 因为是看到了, 直接获取第一个就行
            obj_metadata = controller.last_event.metadata["objects"]
            # target_object_position_dict = obj_metadata[target_obj_Id_list[0]]["position"]
            if target_id not in target_obj_Id_list:
                continue
            target_object_position_dict = get_object_position_by_Id(controller, target_id)
            target_object_position_tuple = (target_object_position_dict["x"], target_object_position_dict["y"], target_object_position_dict["z"])
            # 再次规划, 判断路径终点是否相同
            agent_position_dict = controller.last_event.metadata["agent"]["position"]
            agent_position_tuple = (agent_position_dict["x"], agent_position_dict["y"], agent_position_dict["z"])
            new_path = shortest_path(agent_position_tuple, target_object_position_tuple, neighbors, positions_tuple)
            new_nav_position_tuple = new_path[-1]
            new_nav_position_tuple_xy = (np.round(new_nav_position_tuple[0], 2), np.round(new_nav_position_tuple[2], 2))
            nav_position_tuple_xy = (np.round(nav_position_tuple[0], 2), np.round(nav_position_tuple[2], 2))
            print("以前巡航目标", nav_position_tuple_xy)
            print("更新巡航目标", new_nav_position_tuple_xy)
            if not nav_position_tuple_xy == new_nav_position_tuple_xy:
                nav_position_tuple = new_nav_position_tuple

        end_rotation = int(controller.last_event.metadata['agent']['rotation']["y"])

        # 单次巡航结束--------------------------------------------
        # 补充单次巡航的信息
        # str格式的保存-------------------------------------------------------
        output_dict_one["action"] = "GotoLocation"
        output_dict_one["arg"] = old_arg
        output_dict_one["expectation"] = " "
        output_dict_one["start_position"] = "(" + str(initial_position_tuple[0]) + ", " + str(
            initial_position_tuple[2]) + ")"
        output_dict_one["nav_position"] = "(" + str(next_nav_position[0]) + ", " + str(next_nav_position[2]) + ")"
        output_dict_one["start_rotation"] = str(int(start_rotation))
        output_dict_one["end_rotation"] = str(int(end_rotation))
        output_dict_one["Agent position"] = [np.round(agent_start_position_dict["x"], 2),
                                                np.round(agent_start_position_dict["z"], 2)]
        output_dict_one["input"] = json2str(updated_object_list)
        output_dict_one["input_shorter"] = json2str_shorter(updated_object_list, target_name)
        output_dict_one["position_set"] = list2str(updated_position_list)
        output_dict_one["position_set_shorter"] = list2str_shorter(updated_position_list, next_nav_position)
        # str格式的保存-------------------------------------------------------

        # list形式保存--------------------------------------------------------
        output_dict_one["nav_position_list"] = [next_nav_position[0], next_nav_position[2]]
        output_dict_one["start_position_list"] = [initial_position_tuple[0], initial_position_tuple[2]]
        output_dict_one["input_dict"] = updated_object_list
        output_dict_one["position_set_list"] = updated_position_list
        # list形式保存-------------------------------------------------------

        # 进一步保存当前场景的图片
        output_dict_one_list.append(output_dict_one)
        save_frame_list.append(current_frame)
        save_sem_map_list.append(controller.init_sem_map)
        nav_time = nav_time + 1

        # 判断是否到达
        # target_position是目前巡航位置
        # nav_position_tuple是最终目标位置
        is_arrive = is_arrive_target_point(target_position, nav_position_tuple)
        if nav_time > 50:
            break
        # ********************new add*************************
        # 后续增加环视寻找目标

    return output_dict_one_list, save_frame_list, save_sem_map_list, nav_position_tuple

def filter_point_clouds(point_cloud):
    point_cloud = point_cloud[point_cloud[:, 0] > 0, :]
    point_cloud = point_cloud[point_cloud[:, 1] < 2.3, :]
    point_cloud = point_cloud[point_cloud[:, 2] > 0, :]
    return point_cloud


def check_isin_frame_workplace(point_cloud, point, initial_position_tuple):
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]
    # print("workplace:", (np.min(x), np.max(x), np.min(z), np.max(z)))
    # print(point)

    # 只用考虑x,z即可
    is_within_range = (point[0] >= np.min(x)) and (point[0] <= np.max(x)) and \
                      (point[2] >= np.min(z)) and (point[2] <= np.max(z))
    # print("is_within_range", is_within_range)
    # 判断距离
    x1 = initial_position_tuple[0]
    z1 = initial_position_tuple[2]
    init_point = np.asarray([x1, z1])
    end_point = np.asarray([point[0], point[2]])
    distance = np.linalg.norm(init_point - end_point)

    return is_within_range, distance


def find_object_rotation(controller, target_name):
    find_object = False
    for try_time in range(4):
        frame_object_dict = get_frame_object_list(controller)
        object_name_list = list(frame_object_dict.values())
        print("物体名称", object_name_list)
        if target_name in object_name_list:
            find_object = True
            break
        else:
            event = controller.step(action="RotateRight", forceAction=True)
    return find_object


def find_object_rotation_id(controller, target_name, target_id=None):
    find_object = False
    target_id_list = []
    for i in range(4):
        frame_object_dict = get_frame_object_list(controller)
        for obj_id_one, obj_name_one in frame_object_dict.items():
            if target_name == obj_name_one:
            # if target_name.lower() in obj_name_one.lower():
                target_id_list.append(obj_id_one)
        if target_id is not None:
            if target_id in target_id_list:
                find_object = True
                break
        else:
            if len(target_id_list) > 0:
                target_id = target_id_list[0]
                find_object = True
                break
        event = controller.step(action="RotateRight", forceAction=True)
        rgb, depth_frame = controller.get_obs()
    return find_object, target_id


def look_around_for_object(controller):
    object_name_list = []
    for try_time in range(4):
        frame_object_dict = get_frame_object_list(controller)
        object_name_list_one = list(frame_object_dict.values())
        object_name_list.extend(object_name_list_one)
        event = controller.step(action="RotateRight", forceAction=True)
    # 进行汇总
    object_name_list = list(set(object_name_list))
    # 转化为str
    object_name_list_str = "["
    for object_name_one in object_name_list:
        object_name_list_str = object_name_list_str + object_name_one + ", "
    object_name_list_str = object_name_list_str.strip(", ")
    object_name_list_str = object_name_list_str + "]"
    return object_name_list, object_name_list_str


def look_around_for_object_id(controller):
    object_name_list = []
    object_id_list = []
    for try_time in range(4):
        frame_object_dict = get_frame_object_list(controller)
        rgb_frame, depth_frame = controller.get_obs()
        object_name_list_one = list(frame_object_dict.values())
        object_id_list_one = list(frame_object_dict.keys())
        object_name_list.extend(object_name_list_one)
        object_id_list.extend(object_id_list_one)
        event = controller.step(action="RotateRight", forceAction=True)
        # 每走一步都需要增加额外的感知
    # 进行汇总
    object_name_list = list(set(object_name_list))
    object_id_list = list(set(object_id_list))
    # 转化为str
    object_name_list_str = "["
    for object_name_one in object_name_list:
        object_name_list_str = object_name_list_str + object_name_one + ", "
    object_name_list_str = object_name_list_str.strip(", ")
    object_name_list_str = object_name_list_str + "]"
    return object_name_list, object_id_list, object_name_list_str


def update_object_input(event):
    # 首先获取当前可以看到的物体
    visible_obj_id_list = []
    object_id2index = {}
    object_index2id = {}
    metadata = event.metadata
    object_metadata = metadata['objects']
    for obj_ind in range(len(object_metadata)):
        obj_data_one = object_metadata[obj_ind]
        obj_id = obj_data_one["objectId"]
        object_id2index[obj_id] = obj_ind
        object_index2id[obj_ind] = obj_id
    frame_object_dict = event.instance_masks
    for frame_object_id, frame_object_one in enumerate(list(frame_object_dict.keys())):
        if frame_object_one in object_id2index.keys():
            visible_obj_id_list.append(object_id2index[frame_object_one])
    # 根据可见物体保存list
    new_object_dict = {}
    for obj_ind in visible_obj_id_list:
        obj_Id = object_metadata[obj_ind]["objectId"]
        obj_name = object_metadata[obj_ind]["objectId"].split("|")[0]
        obj_pos = object_metadata[obj_ind]["position"]
        receptacle = object_metadata[obj_ind]["receptacle"]
        toggleable = object_metadata[obj_ind]["toggleable"]
        cookable = object_metadata[obj_ind]["cookable"]
        sliceable = object_metadata[obj_ind]["sliceable"]
        openable = object_metadata[obj_ind]["openable"]
        pickupable = object_metadata[obj_ind]["pickupable"]

        obj_bbox_size = object_metadata[obj_ind]["axisAlignedBoundingBox"]["size"]
        if obj_Id not in new_object_dict.keys():
            new_object_dict[obj_Id] = {}
            new_object_dict[obj_Id]["class_name"] = obj_name
            new_object_dict[obj_Id]["pos"] = (obj_pos["x"], obj_pos["z"], obj_bbox_size["x"], obj_bbox_size["z"])
            new_object_dict[obj_Id]["receptacle"] = receptacle
            new_object_dict[obj_Id]["toggleable"] = toggleable
            new_object_dict[obj_Id]["cookable"] = cookable
            new_object_dict[obj_Id]["sliceable"] = sliceable
            new_object_dict[obj_Id]["openable"] = openable
            new_object_dict[obj_Id]["pickupable"] = pickupable
        else:
            new_object_dict[obj_Id]["class_name"] = obj_name
            new_object_dict[obj_Id]["pos"] = (obj_pos["x"], obj_pos["z"], obj_bbox_size["x"], obj_bbox_size["z"])
            new_object_dict[obj_Id]["receptacle"] = receptacle
            new_object_dict[obj_Id]["toggleable"] = toggleable
            new_object_dict[obj_Id]["cookable"] = cookable
            new_object_dict[obj_Id]["sliceable"] = sliceable
            new_object_dict[obj_Id]["openable"] = openable
            new_object_dict[obj_Id]["pickupable"] = pickupable

    return new_object_dict


def update_position_set(event, positions_tuple):
    world_space_point_cloud = get_frame_pointcloud(event)
    initial_position = event.metadata['agent']['position']
    initial_position_tuple = (initial_position["x"], initial_position["y"], initial_position["z"])
    updated_position_list = []
    for positions_tuple_id, positions_tuple_one in enumerate(positions_tuple):
        is_within_range, _ = check_isin_frame_workplace(world_space_point_cloud, positions_tuple_one, initial_position_tuple)
        if is_within_range:
            updated_position_list.append(positions_tuple_one)
    return updated_position_list


def json2str(object_pos_dict):
    new_object_list = "["
    for obj_id, obj_dict_one in object_pos_dict.items():
        obj_name = obj_dict_one["class_name"]
        if "wall" in obj_name.lower() or "floor" in obj_name.lower() or "room" in obj_name.lower():
            continue
        obj_pos = obj_dict_one["pos"]
        obj_pos_str = "[" + str(np.round(obj_pos[0], 2)) + ", " + str(np.round(obj_pos[1], 2)) + "]"
        receptacle = obj_dict_one["receptacle"]
        toggleable = obj_dict_one["toggleable"]
        cookable = obj_dict_one["cookable"]
        sliceable = obj_dict_one["sliceable"]
        openable = obj_dict_one["openable"]
        pickupable = obj_dict_one["pickupable"]
        properties_str = "("
        if receptacle:
            properties_str = properties_str + "receptacle" + ", "
        if toggleable:
            properties_str = properties_str + "toggleable" + ", "
        if cookable:
            properties_str = properties_str + "cookable" + ", "
        if sliceable:
            properties_str = properties_str + "sliceable" + ", "
        if openable:
            properties_str = properties_str + "openable" + ", "
        if pickupable:
            properties_str = properties_str + "pickupable" + ", "
        properties_str = properties_str.strip(" ").strip(",")
        properties_str = properties_str + ")"
        # new_str = obj_name + ": " + obj_pos_str + ", "
        new_str = obj_name + properties_str + ": " + obj_pos_str + ", "
        new_object_list = new_object_list + new_str
    new_object_list = new_object_list.strip().strip(",") + "]"
    return new_object_list


def json2str_shorter(object_pos_dict, object_name=None):
    choice_number = random.randint(10, 16)
    if len(object_pos_dict) <= choice_number:
        choice_obj_dict = object_pos_dict
    else:
        target_obj_key_list = []
        for obj_id, obj_dict_one in object_pos_dict.items():
            obj_name = obj_dict_one["class_name"]
            if obj_name == object_name:
                target_obj_key_list.append(obj_id)
        choice_obj_dict_key = random.sample(list(object_pos_dict.keys()), choice_number)
        if object_name is not None:
            choice_obj_dict_key.extend(target_obj_key_list)
        random.shuffle(choice_obj_dict_key)
        choice_obj_dict = {}
        for choice_id_one in choice_obj_dict_key:
            choice_obj_dict[choice_id_one] = object_pos_dict[choice_id_one]

    new_object_list = "["
    for obj_id, obj_dict_one in choice_obj_dict.items():
        obj_name = obj_dict_one["class_name"]
        if "wall" in obj_name.lower() or "floor" in obj_name.lower() or "room" in obj_name.lower():
            continue
        obj_pos = obj_dict_one["pos"]
        obj_pos_str = "[" + str(np.round(obj_pos[0], 2)) + ", " + str(np.round(obj_pos[1], 2)) + "]"
        receptacle = obj_dict_one["receptacle"]
        toggleable = obj_dict_one["toggleable"]
        cookable = obj_dict_one["cookable"]
        sliceable = obj_dict_one["sliceable"]
        openable = obj_dict_one["openable"]
        pickupable = obj_dict_one["pickupable"]
        properties_str = "("
        if receptacle:
            properties_str = properties_str + "receptacle" + ", "
        if toggleable:
            properties_str = properties_str + "toggleable" + ", "
        if cookable:
            properties_str = properties_str + "cookable" + ", "
        if sliceable:
            properties_str = properties_str + "sliceable" + ", "
        if openable:
            properties_str = properties_str + "openable" + ", "
        if pickupable:
            properties_str = properties_str + "pickupable" + ", "
        properties_str = properties_str.strip(" ").strip(",")
        properties_str = properties_str + ")"
        # new_str = obj_name + ": " + obj_pos_str + ", "
        new_str = obj_name + properties_str + ": " + obj_pos_str + ", "
        new_object_list = new_object_list + new_str
    new_object_list = new_object_list.strip().strip(",") + "]"
    return new_object_list


def list2str(position_set_list):
    new_position_set = "["
    for position_set_id, position_set_one in enumerate(position_set_list):
        pos_str = "(" + str(np.round(position_set_one[0], 2)) + ", " + str(np.round(position_set_one[2], 2)) + ")"
        if position_set_id == len(position_set_list) - 1:
            new_str = pos_str
        else:
            new_str = pos_str + ", "
        new_position_set = new_position_set + new_str
    new_position_set = new_position_set.rstrip(" ").rstrip(",")
    new_position_set = new_position_set + "]"
    return new_position_set


def list2str_shorter(position_set_list, nav_position):
    choice_number = random.randint(10, 16)
    if len(position_set_list) <= choice_number:
        choice_position_list = position_set_list
    else:
        choice_position_list = random.sample(position_set_list, choice_number)
        choice_position_list.append(nav_position)
    random.shuffle(choice_position_list)
    new_position_set = "["
    for position_set_id, position_set_one in enumerate(choice_position_list):
        pos_str = "(" + str(np.round(position_set_one[0], 2)) + ", " + str(np.round(position_set_one[2], 2)) + ")"
        if position_set_id == len(position_set_list) - 1:
            new_str = pos_str
        else:
            new_str = pos_str + ", "
        new_position_set = new_position_set + new_str
    new_position_set = new_position_set.rstrip(" ").rstrip(",")
    new_position_set = new_position_set + "]"
    return new_position_set


def get_done_action(done_action_list):
    done_action_str = "\n"
    for done_action_one in done_action_list:
        done_action_str = done_action_str + done_action_one + "\n"
    done_action_str.rstrip("\n")
    return done_action_str


def get_planning_action(planning_action_list):
    planning_action_str = "\n"
    for planning_action_one_dict in planning_action_list:
        action_name = planning_action_one_dict["action"]
        arg = planning_action_one_dict["arg"]
        action_one_str = "action: " + action_name + ", " + "target object: " + arg
        planning_action_str = planning_action_str + action_one_str + "\n"
    planning_action_str.rstrip("\n")
    return planning_action_str


def reset_nav_position_tuple(controller, target_name, neighbors, positions_tuple):
    # 首先获取房间中类别名称相同的所有物体
    object_metadata = controller.last_event.metadata["objects"]
    target_object_ind_list = []
    for object_metadata_ind, object_metadata_one in enumerate(object_metadata):
        obj_id = object_metadata_one["objectId"]
        obj_name = obj_id.split("|")[0]
        if obj_name == target_name:
            target_object_ind_list.append(object_metadata_ind)
    # 再次判断

    return None


def get_object_position_by_Id(controller, target_Id):
    obj_metadata = controller.last_event.metadata["objects"]
    for obj_ind, obj_metadata_one in enumerate(obj_metadata):
        obj_Id = obj_metadata_one["objectId"]
        if obj_Id == target_Id:
            object_position_dict = obj_metadata_one["position"]
    return object_position_dict


def nav_openable_object(controller, target_Id):
    # 首先获取目标物体位置
    obj_metadata = controller.last_event.metadata["objects"]
    for obj_ind, obj_metadata_one in enumerate(obj_metadata):
        obj_Id = obj_metadata_one["objectId"]
        if obj_Id == target_Id:
            object_position_dict = obj_metadata_one["position"]
    object_position_tuple = (object_position_dict["x"], object_position_dict["z"])
    # 获取可以open物体
    openable_object_id_list = []
    openable_object_position_list = []
    for obj_ind, obj_metadata_one in enumerate.last_event.metadata["objects"]:
        openable = obj_metadata_one["openable"]
        isOpen = obj_metadata_one["isOpen"]
        obj_Id = obj_metadata_one["objectId"]
        object_position_dict_one = obj_metadata_one["position"]
        if openable and not isOpen:
            openable_object_id_list.append(obj_Id)
            object_position_tuple_one = (object_position_dict_one["x"], object_position_dict_one["z"])
            openable_object_position_list.append(object_position_tuple_one)
    # 找到最近open物体
    distance_list = []
    for position_tuple in openable_object_position_list:
        distance_one = np.sqrt((position_tuple[0] - object_position_tuple[0]) ** 2 + (position_tuple[1] - object_position_tuple[1]) ** 2)
        distance_list.append(distance_one)
    nearest_index = np.argmin(np.asarray(distance_list))
    openable_target_id = openable_object_id_list[nearest_index]
    return openable_target_id


def nav_surrounded_object(controller, target_Id):
    # 首先获取目标物体位置
    obj_metadata = controller.last_event.metadata["objects"]
    for obj_ind, obj_metadata_one in enumerate(obj_metadata):
        obj_Id = obj_metadata_one["objectId"]
        if obj_Id == target_Id:
            object_position_dict = obj_metadata_one["position"]

    neighbors = controller.neighbors
    positions_tuple = controller.positions_tuple

    object_position_tuple = (object_position_dict["x"], object_position_dict["y"], object_position_dict["z"])
    agent_position_dict = controller.last_event.metadata["agent"]["position"]
    agent_position_tuple = (agent_position_dict["x"], agent_position_dict["y"], agent_position_dict["z"])

    # 生成周围环绕点
    surrounded_position_list = []
    # First Point
    surrounded_position_one = np.asarray(object_position_tuple).copy()
    surrounded_position_one[0] = surrounded_position_one[0] - 0.5
    surrounded_position_list.append(surrounded_position_one)
    # Second Point
    surrounded_position_one = np.asarray(object_position_tuple).copy()
    surrounded_position_one[0] = surrounded_position_one[0] + 0.5
    surrounded_position_list.append(surrounded_position_one)
    # Third Point
    surrounded_position_one = np.asarray(object_position_tuple).copy()
    surrounded_position_one[2] = surrounded_position_one[2] + 0.5
    surrounded_position_list.append(surrounded_position_one)
    # Fourth Point
    surrounded_position_one = np.asarray(object_position_tuple).copy()
    surrounded_position_one[2] = surrounded_position_one[2] - 0.5
    surrounded_position_list.append(surrounded_position_one)
    nav_surrounded_position_list = []
    for surrounded_position_one in surrounded_position_list:
        surrounded_position_one_tuple = tuple(surrounded_position_one)
        new_path = shortest_path(agent_position_tuple, surrounded_position_one_tuple, neighbors, positions_tuple)
        new_nav_position_tuple = new_path[-1]
        nav_surrounded_position_list.append(new_nav_position_tuple)
    return nav_surrounded_position_list


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


def down_sample_point_cloud(point_cloud_list):

    num_points = point_cloud_list.shape[0]
    if num_points >= 10000:
        sampled_indices = np.random.choice(num_points, 10000, replace=False)
        sampled_point_cloud = point_cloud_list[sampled_indices]
    else:
        sampled_point_cloud = point_cloud_list

    return sampled_point_cloud






