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
# from utils.procthor_utils import closest_grid_point, get_position_neighbors, shortest_path, get_rotation

def generate_grid_points(x_range, y_range, step):
    x_start, x_end = x_range
    y_start, y_end = y_range
    x_values = np.arange(x_start, x_end + step, step)
    y_values = np.arange(y_start, y_end + step, step)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    return grid_points


def get_path_one_x_step(grid_points, start_point, end_point):
    x1 = end_point[0]
    z1 = end_point[1]
    x0 = start_point[0]
    z0 = start_point[1]
    path_point_xd = grid_points[grid_points[:, 0] == x1, :]
    path_point_zd = grid_points[grid_points[:, 1] == z1, :]
    if x0 != x1:
        if x0 < x1:
            path_point_zd = path_point_zd[path_point_zd[:, 0] >= x0, :]
            ind_zd = np.argmin(path_point_zd, axis=0)[0]
        else:
            path_point_zd = path_point_zd[path_point_zd[:, 0] <= x0, :]
            ind_zd = np.argmax(path_point_zd, axis=0)[0]
        path_point_zd_one = path_point_zd[ind_zd]
        if not (path_point_zd_one == end_point).all():
            return path_point_zd_one
    else:
        if z0 < z1:
            path_point_xd = path_point_xd[path_point_xd[:, 1] >= z0, :]
            ind_xd = np.argmin(path_point_xd, axis=0)[0]
        else:
            path_point_xd = path_point_xd[path_point_xd[:, 1] <= z0, :]
            ind_xd = np.argmax(path_point_xd, axis=0)[0]
        path_point_xd_one = path_point_xd[ind_xd]
        if not (path_point_xd_one == end_point).all():
            return path_point_xd_one
        else:
            return end_point

def get_path_one_x_step(grid_points, start_point, end_point):
    x1 = end_point[0]
    z1 = end_point[1]
    x0 = start_point[0]
    z0 = start_point[1]
    # path_point_xd = grid_points[grid_points[:, 0] == x1, :]
    path_point_zd = grid_points[grid_points[:, 1] == z1, :]
    if x0 < x1:
        path_point_zd = path_point_zd[path_point_zd[:, 0] >= x0, :]
        ind_zd = np.argmin(path_point_zd, axis=0)[0]
    else:
        path_point_zd = path_point_zd[path_point_zd[:, 0] <= x0, :]
        ind_zd = np.argmax(path_point_zd, axis=0)[0]
    path_point_zd_one = path_point_zd[ind_zd]
    if not (path_point_zd_one == end_point).all():
        return path_point_zd_one
    else:
        return end_point


def get_path_one_z_step(grid_points, start_point, end_point):
    x1 = end_point[0]
    z1 = end_point[1]
    x0 = start_point[0]
    z0 = start_point[1]
    path_point_xd = grid_points[grid_points[:, 0] == x1, :]
    # path_point_zd = grid_points[grid_points[:, 1] == z1, :]
    if z0 < z1:
        path_point_xd = path_point_xd[path_point_xd[:, 1] >= z0, :]
        ind_xd = np.argmin(path_point_xd, axis=0)[1]
    else:
        path_point_xd = path_point_xd[path_point_xd[:, 1] <= z0, :]
        ind_xd = np.argmax(path_point_xd, axis=0)[1]
    path_point_xd_one = path_point_xd[ind_xd]
    if not (path_point_xd_one == end_point).all():
        return path_point_xd_one
    else:
        return end_point


def get_position_neighbors(positions_tuple):
    grid_size = 0.25
    neighbors = dict()
    for position in positions_tuple:
        position_neighbors = set()
        for p in positions_tuple:
            if position != p and (
                (
                    abs(position[0] - p[0]) < 1.5 * grid_size
                    and abs(position[2] - p[2]) < 0.5 * grid_size
                )
                or (
                    abs(position[0] - p[0]) < 0.5 * grid_size
                    and abs(position[2] - p[2]) < 1.5 * grid_size
                )
            ):
                position_neighbors.add(p)
        neighbors[position] = position_neighbors

    return neighbors


def closest_grid_point(world_point: Tuple[float, float, float], positions_tuple) -> Tuple[float, float, float]:
    """Return the grid point that is closest to a world coordinate.

    Expects world_point=(x_pos, y_pos, z_pos). Note y_pos is ignored in the calculation.
    """
    def distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]):
        # ignore the y_pos
        return ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5

    min_dist = float("inf")
    closest_point = None
    assert len(positions_tuple) > 0
    for p in positions_tuple:
        dist = distance(p, world_point)
        if dist < min_dist:
            min_dist = dist
            closest_point = p

    return closest_point


def shortest_path(start: Tuple[float, float, float], end: Tuple[float, float, float], neighbors, positions_tuple):
    """Expects the start=(x_pos, y_pos, z_pos) and end=(x_pos, y_pos, z_pos).

    Note y_pos is ignored in the calculation.
    """
    start = closest_grid_point(start, positions_tuple)
    end = closest_grid_point(end, positions_tuple)
    # print(start, end)

    if start == end:
        return [start]

    q = deque()
    q.append([start])

    visited = set()

    while q:
        path = q.popleft()
        pos = path[-1]

        if pos in visited:
            continue

        visited.add(pos)
        for neighbor in neighbors[pos]:
            if neighbor == end:
                return path + [neighbor]
            q.append(path + [neighbor])

    raise Exception("Invalid state. Must be a bug!")


def get_rotation(x0, z0, x1, z1, start_rotation):
    # 先确定旋转方向
    if x0 < x1:
        end_roration = 90
    elif x0 > x1:
        end_roration = 270
    elif z0 < z1:
        end_roration = 0
    elif z0 > z1:
        end_roration = 180
    else:
        end_roration = start_rotation

    return end_roration


def get_action_list(path, rotation_y):
    action_list = []
    start_rotation = int(rotation_y)
    for i in range(len(path) - 1):
        action_one = []
        pos_one = path[i]
        next_pos_one = path[i+1]
        x0, z0 = pos_one[0], pos_one[2]
        x1, z1 = next_pos_one[0], next_pos_one[2]
        end_rotation = get_rotation(x0, z0, x1, z1, start_rotation)
        if start_rotation != end_rotation:
            if start_rotation < end_rotation:
                action_one.extend(["RotateRight"] * int((end_rotation - start_rotation) / 90))
            else:
                action_one.extend(["RotateLeft"] * int((start_rotation - end_rotation) / 90))
            start_rotation = end_rotation
        action_one.extend(["MoveAhead"])
        action_list.extend(action_one)
    return action_list


def get_top_down_frame(controller):
    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bound
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    # add the camera to the scene
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    top_down_frame = event.third_party_camera_frames[-1]
    return Image.fromarray(top_down_frame)


def save_frames_as_video(frames, video_name):
    # 获取第一帧的高度和宽度
    print("保存分辨率:", frames[0].shape)
    height, width, _ = frames[0].shape

    # 创建一个视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 可以根据需要更改编码器
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

    # 将每一帧图片写入视频中
    for frame in frames:
        frame = frame[:, :, ::-1]
        video.write(frame)

    # 释放资源
    video.release()


# visible': False, 'isInteractable': False
def get_interactable_obj(metadata):
    interactable_obj_id_list = []
    object_metadata = metadata['objects']
    for obj_ind in range(len(object_metadata)):
        obj_data_one = object_metadata[obj_ind]
        obj_id = obj_data_one["objectId"]
        isInteractable = obj_data_one["isInteractable"]
        if isInteractable:
            interactable_obj_id_list.append(obj_id)

    return interactable_obj_id_list


def get_visible_obj(metadata):
    visible_obj_id_list = []
    object_metadata = metadata['objects']
    for obj_ind in range(len(object_metadata)):
        obj_data_one = object_metadata[obj_ind]
        obj_id = obj_data_one["objectId"]
        visible = obj_data_one["visible"]
        if visible:
            visible_obj_id_list.append(obj_id)

    return visible_obj_id_list


def check_visible(metadata, check_obj_id):
    visible_obj_id_list = []
    object_metadata = metadata['objects']
    for obj_ind in range(len(object_metadata)):
        obj_data_one = object_metadata[obj_ind]
        obj_id = obj_data_one["objectId"]
        visible = obj_data_one["visible"]
        if visible:
            visible_obj_id_list.append(obj_id)
    if check_obj_id in visible_obj_id_list:
        return True
    else:
        return False


def cpu_only_depth_frame_to_camera_space_xyz(
    depth_frame: np.ndarray, mask: Optional[np.ndarray], fov: float = 90
):
    """"""
    assert (
        len(depth_frame.shape) == 2 and depth_frame.shape[0] == depth_frame.shape[1]
    ), f"depth has shape {depth_frame.shape}, we only support (N, N) shapes for now."

    resolution = depth_frame.shape[0]
    if mask is None:
        mask = np.ones(depth_frame.shape, dtype=bool)

    # pixel centers
    camera_space_yx_offsets = (
        np.stack(np.where(mask))
        + 0.5  # Offset by 0.5 so that we are in the middle of the pixel
    )

    # Subtract center
    camera_space_yx_offsets -= resolution / 2.0

    # Make "up" in y be positive
    camera_space_yx_offsets[0, :] *= -1

    # Put points on the clipping plane
    camera_space_yx_offsets *= (2.0 / resolution) * math.tan((fov / 2) / 180 * math.pi)

    camera_space_xyz = np.concatenate(
        [
            camera_space_yx_offsets[1:, :],  # This is x
            camera_space_yx_offsets[:1, :],  # This is y
            np.ones_like(camera_space_yx_offsets[:1, :]),
        ],
        axis=0,
    )
    return camera_space_xyz * depth_frame[mask][None, :]

def cpu_only_camera_space_xyz_to_world_xyz(
    camera_space_xyzs: np.ndarray,
    camera_world_xyz: np.ndarray,
    rotation: float,
    horizon: float,
):
    # Adapted from https://github.com/devendrachaplot/Neural-SLAM.

    # view_position = 3, world_points = 3 x N
    # NOTE: camera_position is not equal to agent_position!!

    # First compute the transformation that points undergo
    # due to the camera's horizon
    psi = -horizon * np.pi / 180
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    # fmt: off
    horizon_transform = np.array(
        [
            [1, 0, 0], # unchanged
            [0, cos_psi, sin_psi],
            [0, -sin_psi, cos_psi,],
        ],
        np.float64,
    )
    # fmt: on

    # Next compute the transformation that points undergo
    # due to the agent's rotation about the y-axis
    phi = -rotation * np.pi / 180
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    # fmt: off
    rotation_transform = np.array(
        [
            [cos_phi, 0, -sin_phi],
            [0, 1, 0], # unchanged
            [sin_phi, 0, cos_phi],],
        np.float64,
    )
    # fmt: on

    # Apply the above transformations
    view_points = (rotation_transform @ horizon_transform) @ camera_space_xyzs

    # Translate the points w.r.t. the camera's position in world space.
    world_points = view_points + camera_world_xyz[:, None]
    return world_points


def get_toggleable_object(event):
    toggleable_object_dict = {}
    for obj_ind in range(len(event.metadata["objects"])):
        obj_name = event.metadata["objects"][obj_ind]["objectId"].split("|")[0]
        toggleable = event.metadata["objects"][obj_ind]["toggleable"]
        obj_dict_ind = event.metadata["objects"][obj_ind]["objectId"]
        if toggleable:
            toggleable_object_dict[obj_dict_ind] = obj_name
    return toggleable_object_dict


def get_receptacle_object(event):
    receptacle_object_dict = {}
    for obj_ind in range(len(event.metadata["objects"])):
        obj_name = event.metadata["objects"][obj_ind]["objectId"].split("|")[0]
        receptacle = event.metadata["objects"][obj_ind]["receptacle"]
        obj_dict_ind = event.metadata["objects"][obj_ind]["objectId"]
        if receptacle:
            receptacle_object_dict[obj_dict_ind] = obj_name
    return receptacle_object_dict


def get_cookable_object(event):
    cookable_object_dict = {}
    for obj_ind in range(len(event.metadata["objects"])):
        obj_name = event.metadata["objects"][obj_ind]["objectId"].split("|")[0]
        cookable = event.metadata["objects"][obj_ind]["cookable"]
        obj_dict_ind = event.metadata["objects"][obj_ind]["objectId"]
        if cookable:
            cookable_object_dict[obj_dict_ind] = obj_name
    return cookable_object_dict


def get_sliceable_object(event):
    sliceable_object_dict = {}
    for obj_ind in range(len(event.metadata["objects"])):
        obj_name = event.metadata["objects"][obj_ind]["objectId"].split("|")[0]
        sliceable = event.metadata["objects"][obj_ind]["sliceable"]
        obj_dict_ind = event.metadata["objects"][obj_ind]["objectId"]
        if sliceable:
            sliceable_object_dict[obj_dict_ind] = obj_name
    return sliceable_object_dict


def get_openable_object(event):
    openable_object_dict = {}
    for obj_ind in range(len(event.metadata["objects"])):
        obj_name = event.metadata["objects"][obj_ind]["objectId"].split("|")[0]
        openable = event.metadata["objects"][obj_ind]["openable"]
        obj_dict_ind = event.metadata["objects"][obj_ind]["objectId"]
        if openable:
            openable_object_dict[obj_dict_ind] = obj_name
    return openable_object_dict


def get_pickupable_object(event):
    pickupable_object_dict = {}
    for obj_ind in range(len(event.metadata["objects"])):
        obj_name = event.metadata["objects"][obj_ind]["objectId"].split("|")[0]
        pickupable = event.metadata["objects"][obj_ind]["pickupable"]
        obj_dict_ind = event.metadata["objects"][obj_ind]["objectId"]
        if pickupable:
            pickupable_object_dict[obj_dict_ind] = obj_name
    return pickupable_object_dict


def move_to_object(controller_env, object_id):
    object_event = controller_env.last_event.metadata["objects"]
    object_dict = controller_env.object_dict
    obj_position = object_dict[object_id]["position"]
    close_point = closest_grid_point((float(obj_position["x"]), 0.1, float(obj_position["z"])), controller_env.position_tuple)
    # 巡航到这里
    nav_position = dict(x=close_point[0], y=close_point[1], z=close_point[2])
    event = controller_env.step(action="Teleport", position=nav_position, rotation=dict(x=0, y=90, z=0))
    return event


def check_target_in_frame(event, target_name):
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

    new_object_list = []
    for obj_ind in visible_obj_id_list:
        obj_Id = object_metadata[obj_ind]["objectId"]
        obj_name = object_metadata[obj_ind]["objectId"].split("|")[0]
        obj_pos = object_metadata[obj_ind]["position"]
        obj_bbox_size = object_metadata[obj_ind]["axisAlignedBoundingBox"]["size"]
        new_object_list.append(obj_name)

    if target_name in new_object_list:
        return True
    else:
        return False


def get_frame_object(event):
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

    new_object_list = []
    for obj_ind in visible_obj_id_list:
        obj_Id = object_metadata[obj_ind]["objectId"]
        obj_name = object_metadata[obj_ind]["objectId"].split("|")[0]
        obj_pos = object_metadata[obj_ind]["position"]
        obj_bbox_size = object_metadata[obj_ind]["axisAlignedBoundingBox"]["size"]
        # 增加一个筛选
        if "wall" in obj_name.lower() or "floor" in obj_name.lower() or "room" in obj_name.lower():
            continue
        else:
            new_object_list.append(obj_name)
    return new_object_list


def get_obs(event):
    rgb = event.frame.copy()  # shape (h, w, 3)
    h = rgb.shape[0]
    w = rgb.shape[1]
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb


def reset_object(controller_env, old_obj_dict):
    object_poses = [
        {
            "position": o["position"],
            "rotation": o["rotation"],
            "objectName": o["name"],
        }
        for o in old_obj_dict.values()
        if o["pickupable"] or o["moveable"]
    ]
    event = controller_env.step(action="SetObjectPoses", objectPoses=object_poses)
    return event


def get_approximate_success(prev_rgb, frame):
    prev_rgb = np.asarray(prev_rgb)
    frame = np.asarray(frame)
    wheres = np.where(prev_rgb != frame)
    wheres_ar = np.zeros(prev_rgb.shape)
    wheres_ar[wheres] = 1
    wheres_ar = np.sum(wheres_ar, axis=2).astype(bool)
    # connected_regions = skimage.morphology.label(wheres_ar, connectivity=2)
    connected_regions = morphology.label(wheres_ar, connectivity=2)
    unique_labels = [i for i in range(1, np.max(connected_regions)+1)]
    max_area = -1
    for lab in unique_labels:
        wheres_lab = np.where(connected_regions == lab)
        max_area = max(len(wheres_lab[0]), max_area)
    if max_area > 50:
        success = True
    else:
        success = False
    return success


def create_pickup_data(controller_env, house_id, rgb_save_path, fail_action=False):
    create_number = 0
    event = controller_env.last_event
    pickupable_object_dict = get_pickupable_object(event)
    pickupable_object_name_list = list(pickupable_object_dict.values())
    old_obj_dict = controller_env.object_dict
    for obj_ind in pickupable_object_dict.keys():
        event = move_to_object(controller_env, obj_ind)
        obj_name = pickupable_object_dict[obj_ind]
        test_time = 0
        while True:
            is_inframe = check_target_in_frame(controller_env.last_event, obj_name)
            if is_inframe or test_time >= 4:
                break
            else:
                action = dict(action="RotateLeft", degrees="90",
                    forceAction=True)
                event = controller_env.step(action)
                test_time = test_time + 1
        is_inframe = check_target_in_frame(controller_env.last_event, obj_name)
        if not is_inframe:
            continue
        # 开始进行具体执行操作
        before_image = get_obs(controller_env.last_event)
        if fail_action:
            fail_test_time = 0
            frame_object_list = get_frame_object(controller_env.last_event)
            for frame_object_one in frame_object_list:
                if frame_object_one != obj_name and frame_object_one in pickupable_object_name_list:
                    print("fail success change")
                    obj_name = frame_object_one
                    break
        # 生成交互动作
        pickup_action = dict(action="PickupObject",
            objectId=obj_ind, forceAction=True)
        event = controller_env.step(pickup_action)
        if event.metadata["lastActionSuccess"]:
            after_image = get_obs(event)
            # 生成保存名字
            action_str = "{" + "action: PickupObject" + ", " + "target: " + obj_name + "}" + "_" + str(create_number)
            before_save_name = "house_" + str(house_id) + "-" + action_str + "_0" + ".png"
            after_save_name = "house_" + str(house_id) + "-" + action_str + "_1" + ".png"
            before_save_path = os.path.join(rgb_save_path, before_save_name)
            after_save_path = os.path.join(rgb_save_path, after_save_name)
            cv2.imwrite(before_save_path, before_image)
            cv2.imwrite(after_save_path, after_image)
            create_number = create_number + 1
            drop_object_acion = dict(action="DropHandObject", forceAction=True)
            event = controller_env.step(drop_object_acion)
            # reset物体交互场景
            event = reset_object(controller_env, old_obj_dict)
        else:
            continue
    return create_number


def create_put_data(controller_env, house_id, rgb_save_path, fail_action=False):
    create_number = 0
    event = controller_env.last_event
    pickupable_object_dict = get_pickupable_object(event)
    receptacle_object_dict = get_receptacle_object(event)
    receptacle_object_name_list = list(receptacle_object_dict.values())
    if len(receptacle_object_dict.keys()) == 0:
        print("没有容器")
        return create_number
    old_obj_dict = controller_env.object_dict
    # 直接抓取物体, 后续执行其他任务
    for obj_ind in pickupable_object_dict.keys():
        # for inventory_object_dict in event.metadata["inventoryObjects"]:
        # for inventory_object_dict in pickupable_object_dict.keys():
        # 从pickupable_object_dict随机选一个
        # keys = list(pickupable_object_dict.keys())
        keys = list(receptacle_object_dict.keys())
        random_times = 0
        while True:
            random_key = random.choice(keys)
            if obj_ind != random_key or random_times >= 2:
                break
            else:
                random_times = random_times + 1
        if random_key == obj_ind:
            continue
        # inventory_object_dict = pickupable_object_dict[random_key]
        recept_object_id = random_key
        # event = move_to_object(controller_env, obj_ind)
        # 选定放置物体
        # inventory_object_id = inventory_object_dict["objectId"]
        # 移动到指定位置上
        event = move_to_object(controller_env, recept_object_id)
        obj_name = receptacle_object_dict[recept_object_id]
        test_time = 0
        while True:
            is_inframe = check_target_in_frame(controller_env.last_event, obj_name)
            if is_inframe or test_time >= 4:
                break
            else:
                action = dict(action="RotateLeft", degrees="90",
                    forceAction=True)
                event = controller_env.step(action)
                test_time = test_time + 1
        is_inframe = check_target_in_frame(controller_env.last_event, obj_name)
        if not is_inframe:
            print("找不到物体")
            continue
        # 先强行抓取一个物体
        pickup_action = dict(action="PickupObject",
            objectId=obj_ind, forceAction=True)
        event = controller_env.step(pickup_action)
        # 开始进行具体执行操作
        before_image = get_obs(controller_env.last_event)
        if fail_action:
            fail_test_time = 0
            while True:
                fail_object_name = random.choice(receptacle_object_name_list)
                if fail_object_name != obj_name or fail_test_time >= 10:
                    obj_name = fail_object_name
                    break
                else:
                    fail_test_time = fail_test_time + 1

        # 生成交互动作
        put_action = dict(action="PutObject", objectId=recept_object_id, forceAction=True,
            placeStationary=True)
        # 生成放置位置
        # put_x = controller_env.object_dict[obj_ind]["position"]["x"]
        # put_y = controller_env.object_dict[obj_ind]["position"]["z"]
        # put_action = dict(action="PutObject", x=put_x, y=put_y, forceAction=True,
        #     placeStationary=True, putNearXY=True)
        event = controller_env.step(put_action)
        if event.metadata["lastActionSuccess"]:
            after_image = get_obs(event)
            # 生成保存名字
            action_str = "{" + "action: PutObject" + ", " + "target: " + obj_name + "}" + "_" + str(create_number)
            before_save_name = "house_" + str(house_id) + "-" + action_str + "_0" + ".png"
            after_save_name = "house_" + str(house_id) + "-" + action_str + "_1" + ".png"
            before_save_path = os.path.join(rgb_save_path, before_save_name)
            after_save_path = os.path.join(rgb_save_path, after_save_name)
            cv2.imwrite(before_save_path, before_image)
            cv2.imwrite(after_save_path, after_image)
            create_number = create_number + 1
            # reset物体交互场景
            event = reset_object(controller_env, old_obj_dict)
        else:
            # 没有放下, 手上的物体就需要丢掉
            print("put 失败")
            drop_object_acion = dict(action="DropHandObject", forceAction=True)
            event = controller_env.step(drop_object_acion)
            event = reset_object(controller_env, old_obj_dict)
            continue
    return create_number


def create_open_data(controller_env, house_id, rgb_save_path, fail_action=False):
    create_number = 0
    event = controller_env.last_event
    openable_object_dict = get_openable_object(event)
    openable_object_name_list = list(openable_object_dict.values())
    for obj_ind in openable_object_dict.keys():
        # 移动到物体附近
        event = move_to_object(controller_env, obj_ind)
        obj_name = openable_object_dict[obj_ind]
        test_time = 0
        while True:
            is_inframe = check_target_in_frame(controller_env.last_event, obj_name)
            if is_inframe or test_time >= 4:
                break
            else:
                action = dict(action="RotateLeft", degrees="90",
                    forceAction=True)
                event = controller_env.step(action)
                test_time = test_time + 1
        is_inframe = check_target_in_frame(controller_env.last_event, obj_name)
        if not is_inframe:
            continue

        # 增加初始状态判断
        is_open = controller_env.object_dict[obj_ind]["isOpen"]
        if is_open:
            close_action = dict(action="CloseObject",
                objectId=obj_ind,
                forceAction=True)
            event = controller_env.step(close_action)
            if not event.metadata["lastActionSuccess"]:
                continue

        # 开始进行具体执行操作
        before_image = get_obs(controller_env.last_event)
        if fail_action:
            fail_test_time = 0
            while True:
                fail_object_name = random.choice(openable_object_name_list)
                if fail_object_name != obj_name or fail_test_time >= 10:
                    obj_name = fail_object_name
                    break
                else:
                    fail_test_time = fail_test_time + 1
        # 生成交互动作
        pickup_action = dict(action="OpenObject",
            objectId=obj_ind, forceAction=True)
        event = controller_env.step(pickup_action)
        if event.metadata["lastActionSuccess"]:
            after_image = get_obs(event)
            success = get_approximate_success(before_image, after_image)
            if not success:
                continue
            # 生成保存名字
            action_str = "{" + "action: OpenObject" + ", " + "target: " + obj_name + "}" + "_" + str(create_number)
            before_save_name = "house_" + str(house_id) + "-" + action_str + "_0" + ".png"
            after_save_name = "house_" + str(house_id) + "-" + action_str + "_1" + ".png"
            before_save_path = os.path.join(rgb_save_path, before_save_name)
            after_save_path = os.path.join(rgb_save_path, after_save_name)
            cv2.imwrite(before_save_path, before_image)
            cv2.imwrite(after_save_path, after_image)
            create_number = create_number + 1
        else:
            continue
    return create_number


def create_close_data(controller_env, house_id, rgb_save_path, fail_action=False):
    create_number = 0
    event = controller_env.last_event
    openable_object_dict = get_openable_object(event)
    openable_object_name_list = list(openable_object_dict.keys())
    for obj_ind in openable_object_dict.keys():
        # 移动到物体附近
        event = move_to_object(controller_env, obj_ind)
        obj_name = openable_object_dict[obj_ind]
        test_time = 0
        while True:
            is_inframe = check_target_in_frame(controller_env.last_event, obj_name)
            if is_inframe or test_time >= 4:
                break
            else:
                action = dict(action="RotateLeft", degrees="90",
                    forceAction=True)
                event = controller_env.step(action)
                test_time = test_time + 1
        is_inframe = check_target_in_frame(controller_env.last_event, obj_name)
        if not is_inframe:
            continue

        # 增加初始状态判断
        is_open = controller_env.object_dict[obj_ind]["isOpen"]
        if not is_open:
            open_action = dict(action="OpenObject",
                objectId=obj_ind,
                forceAction=True)
            event = controller_env.step(open_action)
            if not event.metadata["lastActionSuccess"]:
                continue

        # 开始进行具体执行操作
        before_image = get_obs(controller_env.last_event)
        if fail_action:
            fail_test_time = 0
            while True:
                fail_object_name = random.choice(openable_object_name_list)
                if fail_object_name != obj_name or fail_test_time >= 10:
                    obj_name = fail_object_name
                    break
                else:
                    fail_test_time = fail_test_time + 1
        # 生成交互动作
        pickup_action = dict(action="CloseObject",
            objectId=obj_ind, forceAction=True)
        event = controller_env.step(pickup_action)
        if event.metadata["lastActionSuccess"]:
            after_image = get_obs(event)
            success = get_approximate_success(before_image, after_image)
            if not success:
                continue
            # 生成保存名字
            action_str = "{" + "action: CloseObject" + ", " + "target: " + obj_name + "}" + "_" + str(create_number)
            before_save_name = "house_" + str(house_id) + "-" + action_str + "_0" + ".png"
            after_save_name = "house_" + str(house_id) + "-" + action_str + "_1" + ".png"
            before_save_path = os.path.join(rgb_save_path, before_save_name)
            after_save_path = os.path.join(rgb_save_path, after_save_name)
            cv2.imwrite(before_save_path, before_image)
            cv2.imwrite(after_save_path, after_image)
            create_number = create_number + 1
        else:
            continue
    return create_number


def create_slice_data(controller_env, house_id, rgb_save_path, fail_action=False):
    create_number = 0
    event = controller_env.last_event
    sliceable_object_dict = get_sliceable_object(event)
    sliceable_object_name_list = list(sliceable_object_dict.keys())
    old_obj_dict = controller_env.object_dict
    # 在刚开始抓起Knife, ButterKnife
    grasp_knife = False
    for obj_dict in controller_env.object_dict.values():
        obj_name = obj_dict["objectId"].split("|")[0]
        if "Knife" in obj_name:
            obj_id = obj_dict["objectId"]
            event = move_to_object(controller_env, obj_id)
            pickup_action = dict(action="PickupObject",
                objectId=obj_id, forceAction=True)
            grasp_knife = True
            break
    if not grasp_knife:
        return create_number
    else:
        for obj_ind in sliceable_object_dict.keys():
            # 移动到指定的区域
            event = move_to_object(controller_env, obj_ind)
            obj_name = sliceable_object_dict[obj_ind]
            test_time = 0
            while True:
                is_inframe = check_target_in_frame(controller_env.last_event, obj_name)
                if is_inframe or test_time >= 4:
                    break
                else:
                    action = dict(action="RotateLeft", degrees="90",
                        forceAction=True)
                    event = controller_env.step(action)
                    test_time = test_time + 1
            is_inframe = check_target_in_frame(controller_env.last_event, obj_name)
            if not is_inframe:
                drop_object_acion = dict(action="DropHandObject", forceAction=True)
                event = controller_env.step(drop_object_acion)
                event = reset_object(controller_env, old_obj_dict)
                continue
            # 开始进行具体执行操作
            before_image = get_obs(controller_env.last_event)
            if fail_action:
                fail_test_time = 0
                while True:
                    fail_object_name = random.choice(sliceable_object_name_list)
                    if fail_object_name != obj_name or fail_test_time >= 10:
                        obj_name = fail_object_name
                        break
                    else:
                        fail_test_time = fail_test_time + 1
            # 生成交互动作
            pickup_action = dict(action="SliceObject",
                objectId=obj_ind, forceAction=True)
            event = controller_env.step(pickup_action)
            if event.metadata["lastActionSuccess"]:
                after_image = get_obs(event)
                # 生成保存名字
                action_str = "{" + "action: SliceObject" + ", " + "target: " + obj_name + "}" + "_" + str(create_number)
                before_save_name = "house_" + str(house_id) + "-" + action_str + "_0" + ".png"
                after_save_name = "house_" + str(house_id) + "-" + action_str + "_1" + ".png"
                before_save_path = os.path.join(rgb_save_path, before_save_name)
                after_save_path = os.path.join(rgb_save_path, after_save_name)
                cv2.imwrite(before_save_path, before_image)
                cv2.imwrite(after_save_path, after_image)
                create_number = create_number + 1
            else:
                # 没有放下, 手上的物体就需要丢掉
                drop_object_acion = dict(action="DropHandObject", forceAction=True)
                event = controller_env.step(drop_object_acion)
                event = reset_object(controller_env, old_obj_dict)
                continue
        return create_number


def create_toggle_data(controller_env, house_id, rgb_save_path, fail_action=False):
    create_number = 0
    event = controller_env.last_event
    toggleable_object_dict = get_toggleable_object(event)
    toggleable_object_name_list = list(toggleable_object_dict.values())
    for obj_ind in toggleable_object_dict.keys():
        # 移动到指定的区域
        event = move_to_object(controller_env, obj_ind)
        obj_name = toggleable_object_dict[obj_ind]
        test_time = 0
        while True:
            is_inframe = check_target_in_frame(controller_env.last_event, obj_name)
            if is_inframe or test_time >= 4:
                break
            else:
                action = dict(action="RotateLeft", degrees="90",
                    forceAction=True)
                event = controller_env.step(action)
                test_time = test_time + 1
        is_inframe = check_target_in_frame(controller_env.last_event, obj_name)
        if not is_inframe:
            continue
        # 开始进行具体执行操作
        before_image = get_obs(controller_env.last_event)
        if fail_action:
            fail_test_time = 0
            while True:
                fail_object_name = random.choice(toggleable_object_name_list)
                if fail_object_name != obj_name or fail_test_time >= 10:
                    obj_name = fail_object_name
                    break
                else:
                    fail_test_time = fail_test_time + 1
        # 生成交互动作 toggle_off
        toggle_off_action = dict(action="ToggleObjectOff",
            objectId=obj_ind, forceAction=True)
        toggle_on_action = dict(action="ToggleObjectOn",
            objectId=obj_ind, forceAction=True)
        # 判断物体状态
        is_Toggled = controller_env.object_dict[obj_ind]["isToggled"]
        if is_Toggled:
            event = controller_env.step(toggle_off_action)
        else:
            event = controller_env.step(toggle_on_action)

        if event.metadata["lastActionSuccess"]:
            after_image = get_obs(event)
            success = get_approximate_success(before_image, after_image)
            if not success:
                continue
            # 生成保存名字
            action_str = "{" + "action: ToggleObject" + ", " + "target: " + obj_name + "}" + "_" + str(create_number)
            before_save_name = "house_" + str(house_id) + "-" + action_str + "_0" + ".png"
            after_save_name = "house_" + str(house_id) + "-" + action_str + "_1" + ".png"
            before_save_path = os.path.join(rgb_save_path, before_save_name)
            after_save_path = os.path.join(rgb_save_path, after_save_name)
            cv2.imwrite(before_save_path, before_image)
            cv2.imwrite(after_save_path, after_image)
            create_number = create_number + 1
        else:
            continue
    return create_number


def get_prompt_type(prompt_type=None):
    if prompt_type == "s1":
        human_instruction_prompt = "Based on the input instruction, agent position, object list and done actions, please output the current action to be performed and the subsequent planning. Specifically, the input information means the following:" + "\n" + \
                                   "Instructions represent the requirements you need to handle." + "\n" + "Agent position represents where you are, expressed in the form of coordinates, represented as (x1, y1). These values represent the center point." + "\n" + \
                                   "Object list represents the objects that can be interacted with, represented as class name(properties): [x1, y1], class name represents the name of the object, (properties) represents the purpose of the object, and [x1, y1] represents the position of the center of the object." + "\n" + \
                                   "Done actions represents the high-level actions that have been completed." + "\n" + "Each high-level action is formatted as a dictionary, denoted as {action: action name, arg: object name}." + "\n" + \
                                   "In addition, the output current action and subsequent planning actions are high level actions."
    elif prompt_type == "s2":
        human_instruction_prompt = "Based on the input execute action, agent position, object list and the reachable points list, please output the specific single-step action to be performed, the target position and the target object. Specifically, the input information means the following:" + "\n" + \
                                   "Execute action represents the high-level action that is currently being executed." + "\n" + \
                                   "Object list represents the objects that can be interacted with, represented as class name(properties): [x1, y1], class name represents the name of the object, (properties) represents the purpose of the object, and [x1, y1] represents the position of the center of the object." + "\n" + \
                                   "Agent position represents where you are, expressed in the form of coordinates, represented as (x1, y1). These values represent the center point." + "\n" + \
                                   "Reachable position list represents the points you can reach, with each point represented as (x1, y1)." + "\n" + \
                                   "Specific single-step action represents a low-level action that you need to perform, in the form of a dictionary as {action: action name, target position: (x1, y1), target object: object name}." + "\n" + \
                                   "Each high-level action is formatted as a dictionary, denoted as {action: action name, arg: object name}."
    elif prompt_type == "s3":
        human_instruction_prompt = "Based on the execute action, agent position, object list and the subsequent planning actions, please output whether the current execut action is complete (yes or no). Specifically, the input information means the following:" + "\n" + \
                                   "Execute action represents the high-level action that is currently being executed." + "\n" + \
                                   "Agent position represents where you are, expressed in the form of coordinates, represented as (x1, y1). These values represent the center point." + "\n" + \
                                   "Object list represents the objects that can be interacted with, represented as class name(properties): [x1, y1], class name represents the name of the object, (properties) represents the purpose of the object, and [x1, y1] represents the position of the center of the object." + "\n" + \
                                   "Each high-level action is formatted as a dictionary, denoted as {action: action name, arg: object name}."
    else:
        human_instruction_prompt = " "

    return human_instruction_prompt


def get_prompt_type_v2(prompt_type=None):
    if prompt_type == "s1":
        human_instruction_prompt = "Based on the input instruction, agent position, object list and done actions, please output the current action to be performed and the subsequent planning. Specifically, the input information means the following:" + "\n" + \
                                   "Instructions represent the requirements you need to handle." + "\n" + "Agent position represents where you are, expressed in the form of coordinates, represented as (x1, y1). These values represent the center point." + "\n" + \
                                   "Object list represents the objects that can be interacted with, represented as class name(properties): [x1, y1], class name represents the name of the object, (properties) represents the purpose of the object, and [x1, y1] represents the position of the center of the object." + "\n" + \
                                   "Done actions represents the high-level actions that have been completed." + "\n" + "Please perform the following high-level action and write it in the format: action: action name, target object: object name" + "\n" + \
                                   "In addition, the output current action and subsequent planning actions are high level actions."
    elif prompt_type == "s2":
        human_instruction_prompt = "Based on the input execute action, agent position, object list and the reachable points list, please output the specific single-step action to be performed, the target position and the target object. Specifically, the input information means the following:" + "\n" + \
                                   "Execute action represents the high-level action that is currently being executed." + "\n" + \
                                   "Object list represents the objects that can be interacted with, represented as class name(properties): [x1, y1], class name represents the name of the object, (properties) represents the purpose of the object, and [x1, y1] represents the position of the center of the object." + "\n" + \
                                   "Agent position represents where you are, expressed in the form of coordinates, represented as (x1, y1). These values represent the center point." + "\n" + \
                                   "Reachable position list represents the points you can reach, with each point represented as (x1, y1)." + "\n" + \
                                   "Specific single-step action represents a low-level action that you need to perform, write it in the format: action: action name, target position: (x1, y1), target object: object name." + "\n" + \
                                   "Please perform the following high-level action and write it in the format: action: action name, target object: object name."
    elif prompt_type == "s3":
        human_instruction_prompt = "Based on the execute action, agent position, object list and the subsequent planning actions, please output whether the current execut action is complete (yes or no). Specifically, the input information means the following:" + "\n" + \
                                   "Execute action represents the high-level action that is currently being executed." + "\n" + \
                                   "Agent position represents where you are, expressed in the form of coordinates, represented as (x1, y1). These values represent the center point." + "\n" + \
                                   "Object list represents the objects that can be interacted with, represented as class name(properties): [x1, y1], class name represents the name of the object, (properties) represents the purpose of the object, and [x1, y1] represents the position of the center of the object." + "\n" + \
                                   "Please perform the following high-level action and write it in the format: action: action name, arg: object name."
    else:
        human_instruction_prompt = " "

    return human_instruction_prompt


