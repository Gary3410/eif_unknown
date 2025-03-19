import os.path
import time
import prior
import json
from tqdm import tqdm
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import numpy as np
import random
import cv2
from utils.procthor_utils import closest_grid_point, get_position_neighbors, shortest_path, get_rotation
from utils.procthor_utils import get_action_list

"""
重点复写高层交互指令的接口
"""

def teleport_nav(controller, position, rotation):
    """
    :param controller:
    :param position: dict(x=, y=, z=)
    :param rotation: dict(x=0, y=, z=0)
    :return: event
    """
    event = controller.step(action="Teleport", position=position, rotation=rotation)
    return event


def step_nav(controller, start_position, end_position):
    """
    :param controller:
    :param start_position: dict(x=, y=, z=)
    :param end_position: dict(x=, y=, z=)
    :param rotation: dict(x=0, y=, z=0)
    :return: event
    """
    event = controller.step(action="GetReachablePositions")
    positions = event.metadata["actionReturn"]
    positions_tuple = [(p["x"], p["y"], p["z"]) for p in positions]
    neighbors = get_position_neighbors(positions_tuple)
    end_position_tuple = (end_position["x"], end_position["y"], end_position["z"])
    initial_position = controller.last_event.metadata['agent']['position']
    initial_rotation = controller.last_event.metadata['agent']['rotation']
    start_position_tuple = (initial_position["x"], initial_position["y"], initial_position["z"])
    path = shortest_path(start_position_tuple, end_position_tuple, neighbors, positions_tuple)
    # 进行具体巡航策略
    start_rotation = initial_rotation['y']
    action_list = get_action_list(path, rotation_y=start_rotation)
    for action_one in action_list:
        event = controller.step(action=action_one)
    return event


def smooth_nav(controller, start_position, end_position):
    frame_list = []
    event = controller.step(action="GetReachablePositions")
    positions = event.metadata["actionReturn"]
    positions_tuple = [(p["x"], p["y"], p["z"]) for p in positions]
    neighbors = get_position_neighbors(positions_tuple)
    end_position_tuple = (end_position["x"], end_position["y"], end_position["z"])
    initial_position = controller.last_event.metadata['agent']['position']
    initial_rotation = controller.last_event.metadata['agent']['rotation']
    start_position_tuple = (initial_position["x"], initial_position["y"], initial_position["z"])
    path = shortest_path(start_position_tuple, end_position_tuple, neighbors, positions_tuple)
    # 进行具体巡航策略
    start_rotation = initial_rotation['y']
    action_list = get_action_list(path, rotation_y=start_rotation)
    for action_one in action_list:
        if "Rotate" in action_one:
            frames = [
                controller.step(action=action_one, degrees=5).frame
                for _ in range(90 // 5)
            ]
        else:
            frames = [
                controller.step(action=action_one).frame
            ]
        # time.sleep(1)
        frame_list.extend(frames)
    return event, frame_list


def smooth_rotation(controller, end_rotation):
    action_list = []
    start_rotation = controller.last_event["agent"]["rotation"]["y"]
    if start_rotation != end_rotation:
        if start_rotation < end_rotation:
            action_list.extend(["RotateRight"] * int((end_rotation - start_rotation) / 90))
        else:
            action_list.extend(["RotateLeft"] * int((start_rotation - end_rotation) / 90))
    if len(action_list) == 0:
        frames = []
    for action_one in action_list:
        if "Rotate" in action_one:
            frames = [
                controller.step(action=action_one, degrees=5).frame
                for _ in range(90 // 5)
            ]
    return frames


