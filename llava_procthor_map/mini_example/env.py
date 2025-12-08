import math

import cv2
import copy
import sys
import os
import json
from collections import Counter, OrderedDict
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import matplotlib
import numpy as np
import random
from typing import Tuple
from collections import deque
from typing import Optional, Sequence, cast

import torch
from torchvision import transforms
import torchvision
from tqdm import tqdm

# 配置环境参数
AGENT_STEP_SIZE = 0.25
RECORD_SMOOTHING_FACTOR = 1
CAMERA_HEIGHT_OFFSET = 0.75
VISIBILITY_DISTANCE = 25

class ThorEnvCode(Controller):
    # action enum
    class Action(Enum):
        PASS = 0,
        GOTO = 1,
        PICK = 2,
        PUT = 3,
        OPEN = 4,
        CLOSE = 5,
        TOGGLE = 6,
        HEAT = 7,
        CLEAN = 8,
        COOL = 9,
        SLICE = 10,
        INVENTORY = 11,
        EXAMINE = 12,
        LOOK = 13

    def __init__(self, args,
                 quality='Ultra',
                 habitat_config=None, use_CloudRendering=False):
        # super().__init__(quality=quality, renderImage=True, renderDepthImage=True)
        # 先进行CloudRendering启动
        if args.alfred_scene:
            # IMAGE_WIDTH = args.env_frame_width_alfred
            image_width = args.env_frame_width_alfred
            # IMAGE_HEIGHT = args.env_frame_height_alfred
            image_height = args.env_frame_height_alfred
        else:
            # IMAGE_WIDTH = args.env_frame_width
            image_width = args.env_frame_width
            # IMAGE_HEIGHT = args.env_frame_height
            image_height = args.env_frame_height
        if use_CloudRendering:
            super().__init__(quality=quality, renderImage=True, renderDepthImage=True, platform=CloudRendering,
                width=image_width, height=image_height, fieldOfView=90)
        else:
            super().__init__(quality=quality, renderImage=True, renderDepthImage=True,
                width=image_width, height=image_height, fieldOfView=90)
        self.view_angle = None
        self.steps_taken = None
        self.errs = None
        self.actions = None
        self.args = args
        self.receptacles = {}

        if self.args.vis_demo:
            event = self.step(dict(action='Initialize', gridSize=AGENT_STEP_SIZE / RECORD_SMOOTHING_FACTOR,
                cameraY=CAMERA_HEIGHT_OFFSET,
                renderImage=True,
                renderDepthImage=True,
                renderClassImage=True,
                renderObjectImage=True,
                visibility_distance=VISIBILITY_DISTANCE,
                makeAgentsVisible=True))

        else:
            event = self.step(dict(action='Initialize', gridSize=AGENT_STEP_SIZE / RECORD_SMOOTHING_FACTOR,
                cameraY=CAMERA_HEIGHT_OFFSET,
                renderImage=True,
                renderDepthImage=True,
                renderClassImage=True,
                renderObjectImage=True,
                visibility_distance=VISIBILITY_DISTANCE,
                makeAgentsVisible=False))

    def load_scene(self, house_name, args=None):
        """
        :param house_name: procthor中house_name
        :param args: 模型具体参数
        :return: 返回初始观察
        """
        # scene setup
        self.view_angle = 30
        self.steps_taken = 0
        self.errs = []
        self.actions = [dict(action="LookDown_15", forceAction=True)]
        # 初始化
        # super().reset(house_name)
        self.reset(house_name)
        # 之前保存的东西需要去掉

        # 巡航可视化展示
        self.nav_path = None
        self.seen_position_list = []
        self.init_sem_map = None
        if self.args.vis_demo:
            event = super().step(dict(
                action='Initialize',
                gridSize=AGENT_STEP_SIZE / RECORD_SMOOTHING_FACTOR,
                cameraY=CAMERA_HEIGHT_OFFSET,
                renderImage=True,
                renderDepthImage=True,
                renderClassImage=True,
                renderObjectImage=True,
                visibility_distance=VISIBILITY_DISTANCE,
                makeAgentsVisible=True,
            ))
        else:
            event = super().step(dict(
                action='Initialize',
                gridSize=AGENT_STEP_SIZE / RECORD_SMOOTHING_FACTOR,
                cameraY=CAMERA_HEIGHT_OFFSET,
                renderImage=True,
                renderDepthImage=True,
                renderClassImage=True,
                renderObjectImage=True,
                visibility_distance=VISIBILITY_DISTANCE,
                makeAgentsVisible=False,
            ))
        self.camera_horizon = self.last_event.metadata['agent']['cameraHorizon']
        # internal position

        object_data = event.metadata["objects"]
        object_dict = {}
        for object_one in object_data:
            object_id = object_one["objectId"]
            object_dict[object_id] = object_one
            # 增加容器初始化
            if object_one["receptacle"]:
                self.receptacles[object_id] = object_one

        self.object_dict = object_dict

        objects = self.last_event.metadata["objects"]
        rgb_frame, depth_frame, _, _ = self.get_obs()

        return rgb_frame, depth_frame

    def get_obs(self):
        # RGB图像
        rgb = self.last_event.frame.copy()  # shape (h, w, 3)
        h = rgb.shape[0]
        w = rgb.shape[1]
        self.image_width = w
        self.image_height = h
        # 深度图像
        depth_frame = self.last_event.depth_frame.copy()
        mask_list = []
        label_dict = {}
        # mask
        instance_masks_dict = self.last_event.instance_masks
        # bbox [x_min, y_min, x_max, y_max]
        bbox_dict = self.last_event.instance_detections2D
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
            label_dict[len(mask_list) - 1] = dict(bbox=bbox_one, class_name=class_name)

        mask_list = np.concatenate(mask_list, axis=0)
        assert len(label_dict.keys()) == mask_list.shape[0]

        # 采集pose信息
        fov = self.last_event.metadata["fov"]
        cameraHorizon = self.last_event.metadata["agent"]["cameraHorizon"]
        camera_world_xyz = list(self.last_event.metadata["agent"]["position"].values())
        robot_position = np.asarray(list(self.last_event.metadata["agent"]["position"].values()))
        camera_world_xyz[1] = camera_world_xyz[1] + 0.675
        camera_world_xyz = np.asarray(camera_world_xyz)
        rotation = self.last_event.metadata["agent"]["rotation"]['y']
        info_dict = dict(fov=fov, cameraHorizon=cameraHorizon, camera_world_xyz=camera_world_xyz, rotation=rotation)
        info_dict["camera_height"] = 1.576
        info_dict["robot_position"] = robot_position

        return rgb, depth_frame, mask_list, info_dict

    def to_thor_api_exec(self, action, object_id="", smooth_nav=False, force=True):
        action_received = copy.deepcopy(action)
        self.action_received = action_received
        self.last_action = action_received
        if force:
            forceAction = True
        else:
            forceAction = False

        if "End" in action or "Stop" in action:
            self.stopped = True
            event = self.last_event
        elif "RotateLeft" in action:
            if smooth_nav:
                action = dict(action="RotateLeft", degrees="5",
                    forceAction=True)
            else:
                action = dict(action="RotateLeft", degrees="90",
                    forceAction=True)
            # obs, rew, done, info = self.step(action, smooth_nav=smooth_nav)
            event = self.step(action)
        elif "RotateRight" in action:
            if smooth_nav:
                action = dict(action="RotateRight", degrees="5",
                    forceAction=True)
            else:
                action = dict(action="RotateRight", degrees="90",
                    forceAction=True)
            # obs, rew, done, info = self.step(action, smooth_nav=smooth_nav)
            event = self.step(action)
        elif "MoveAhead" in action:
            action = dict(action="MoveAhead",
                forceAction=True)
            # obs, rew, done, info = self.step(action, smooth_nav=smooth_nav)
            event = self.step(action)


