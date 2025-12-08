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

from utils.procthor_utils import closest_grid_point, get_position_neighbors, shortest_path, get_rotation, get_action_list
from utils.procthor_utils import cpu_only_depth_frame_to_camera_space_xyz, cpu_only_camera_space_xyz_to_world_xyz
# import envs.utils.pose as pu
import torch
from torchvision import transforms
import torchvision
from tqdm import tqdm
import utils.pose as pu

from enum import Enum
from models.segmentation.segmentation_helper import SemgnetationHelper
from models.segmentation.segmentation_helper_procthor import SemgnetationHelperProcThor
from models.Detic.segmentation_helper_procthor_detic import SemgnetationHelperProcThorDetic
from models.depth.alfred_perception_models import AlfredSegmentationAndDepthModel
from collections import Counter, OrderedDict

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

        # 增加巡航设定
        self.seen_object_dict = []
        self.seen_object_dict_all = {}
        self.position_set = []
        self.updated_position_list = []
        self.seen_object_id2type = {}
        self.done_action_list = []
        self.position_tuple = None
        self.neighbors = None
        self.stopped = False
        self.object_dict = None
        self.info = {}
        self.last_inter_object_id = None
        # 巡航可视化展示
        self.nav_path = None
        self.seen_position_list = []
        self.init_sem_map = None
        # 加载处理过程
        # self.total_cat2idx = json.load(open("utils/total_cat2idx.json"))
        if args.alfred_scene:
            self.total_cat2idx = json.load(open(args.total_cat2idx_alfred_path))
        else:
            if args.use_sem_seg:
                self.total_cat2idx = json.load(open(args.total_cat2idx_procthor_path))
            else:
                self.total_cat2idx = json.load(open(args.total_cat2idx_procthor_path))
                # 增加额外的信息
                self.total_cat2idx["wall"] = 95
                self.total_cat2idx["floor"] = 96

        # add 信息
        self.frontiers_distance_threshold = args.frontiers_distance_threshold
        self.picked_up = False
        self.picked_up_cat = None
        self.picked_up_mask = None
        self.seen_object_name_list_all = []
        self.interactive_object_id = None
        self.api_fail_times = 0
        # 用于替代的物体
        self.cat_equate_dict = {}
        self.err_message = None
        self.frame_object_name_list = []

        # 增加llm_planner设定
        self.feedback = None
        self.receptacles = {}
        self.curr_recep = None
        self.exec_action_list = []

        self.res = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize((args.frame_height, args.frame_width),
                                           interpolation=transforms.InterpolationMode.NEAREST)])
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

        # internal states
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()
        # self.args = args

        # --------add learned_model--------------
        # FILM Mask-RCNN and depth
        if args.use_sem_seg:
            if args.alfred_scene:
                self.seg = SemgnetationHelper(self)
            else:
                # self.seg = SemgnetationHelperProcThor(self)
                if args.use_detic:
                    self.seg = SemgnetationHelperProcThorDetic(self)
                else:
                    self.seg = SemgnetationHelperProcThor(self)
            self.init_depth_model(args)

        print("ThorEnv started.")

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

        self.seen_object_dict = []
        self.seen_object_dict_all = {}
        self.position_set = []
        self.updated_position_list = []
        self.seen_object_id2type = {}
        self.done_action_list = []
        self.position_tuple = None
        self.neighbors = None
        self.stopped = False
        self.object_dict = None
        self.info = {}
        self.seen_object_name_list_all = []
        self.interactive_object_id = None
        self.exec_action_list = []
        self.last_inter_object_id = None

        self.picked_up = False
        self.picked_up_cat = None
        self.picked_up_mask = None

        self.last_sim_location = self.get_sim_location()
        self.o = 0.0
        self.o_behind = 0.0
        self.api_fail_times = 0

        self.reset_states()

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
        # 初始化整体场景可到达点
        event = self.step(action="GetReachablePositions")
        positions = event.metadata["actionReturn"]
        self.position_tuple = [(p["x"], p["y"], p["z"]) for p in positions]
        self.neighbors = get_position_neighbors(self.position_tuple)
        # 初始化场景agent位置
        # initial_position = random.choice(positions)
        # event = self.step(action="Teleport", position=initial_position, rotation=dict(x=0, y=90, z=0))
        # initial_rotation = event.metadata['agent']['rotation']
        # 初始化物体list
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

    def reset_states(self):
        # 重新初始化状态
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

    def reset_arrival_position(self):
        event = self.step(action="GetReachablePositions")
        positions = event.metadata["actionReturn"]
        self.position_tuple = [(p["x"], p["y"], p["z"]) for p in positions]
        self.neighbors = get_position_neighbors(self.position_tuple)

    def reset_object_dict(self):
        object_data = self.last_event.metadata["objects"]
        object_dict = {}
        receptacles = {}
        for object_one in object_data:
            object_id = object_one["objectId"]
            object_dict[object_id] = object_one
            if object_one["receptacle"]:
                receptacles[object_id] = object_one
        self.object_dict = object_dict
        self.receptacles = receptacles

    def get_obs(self):
        # RGB图像
        rgb = self.last_event.frame.copy()  # shape (h, w, 3)
        h = rgb.shape[0]
        w = rgb.shape[1]
        self.image_width = w
        self.image_height = h
        # 深度图像
        depth_frame = self.last_event.depth_frame.copy()
        if self.args.use_learned_depth or self.args.use_sem_seg:
            # [nc, 300, 300]
            sem_seg_pred, depth_frame = self._preprocess_obs(rgb, depth_frame)
            if self.args.alfred_scene:
                mask_list, label_dict = self.post_process_seg_result_alfred()
            else:
                mask_list, label_dict = self.post_process_seg_result_procthor()
        else:
            # 增加可到达点与可见物体
            mask_list = []
            label_dict = {}
            self.seen_object_dict = self.update_object_input()
            for obj_id, obj_dict_one in self.seen_object_dict.items():
                obj_class_name = obj_dict_one["class_name"]
                if obj_id not in self.seen_object_dict_all.keys():
                    self.seen_object_dict_all[obj_id] = obj_class_name
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

        # add 交互物体
        info_dict["interactive_object"] = None
        info_dict["add_robot_mask"] = False

        info_dict["label_info"] = label_dict
        # 增加可视化信息
        if self.args.use_sem_seg:
            info_dict["vis_mask"] = self.seg.visualize_sem()

        dx, dy, do = self.get_pose_change()
        info_dict['sensor_pose'] = [dx, dy, do]

        self.reset_object_dict()

        return rgb, depth_frame, mask_list, info_dict

    def execAction(self, action, target_arg, target_position=None, target_object_id=None, mask_px_sample=1, smooth_nav=False, force=True):
        """
        :param action: 具体的执行动作 list, [goto] 重点是应对巡航的一些具体移动指令
        :return:
        """
        goal_success = False
        self.last_action_ogn = action
        all_ids = []
        instance_segs = np.array(self.last_event.instance_segmentation_frame)
        color_to_object_id = self.last_event.color_to_object_id
        if self.args.use_sem_seg:
            interact_mask = self.seg.sem_seg_get_instance_mask_from_obj_type(target_arg)
        else:
            interact_mask = self.get_instance_mask_from_obj_type(object_name=target_arg)
        if interact_mask is not None:
            nz_rows, nz_cols = np.nonzero(interact_mask)
            instance_counter = Counter()
            for i in range(0, len(nz_rows), mask_px_sample):
                x, y = nz_rows[i], nz_cols[i]
                instance = tuple(instance_segs[x, y])
                instance_counter[instance] += 1

            iou_scores = {}
            for color_id, intersection_count in instance_counter.most_common():
                union_count = np.sum(np.logical_or(np.all(instance_segs == color_id, axis=2), interact_mask.astype(bool)))
                iou_scores[color_id] = intersection_count / float(union_count)
            iou_sorted_instance_ids = list(OrderedDict(sorted(iou_scores.items(), key=lambda x: x[1], reverse=True)))

            # get the most common object ids ignoring the object-in-hand
            inv_obj = self.last_event.metadata['inventoryObjects'][0]['objectId'] \
                if len(self.last_event.metadata['inventoryObjects']) > 0 else None
            all_ids = [color_to_object_id[color_id] for color_id in iou_sorted_instance_ids
                       if color_id in color_to_object_id and color_to_object_id[color_id] != inv_obj]

            # print instance_ids
            instance_ids = [inst_id for inst_id in all_ids if inst_id is not None]
            instance_ids = self.prune_by_any_interaction(instance_ids)

            # add process
            if not self.args.use_sem_seg:
                pass
            else:
                instance_ids_new = self.prune_by_any_interaction(instance_ids)
                for instance_id in instance_ids:
                    if 'Sink' in instance_id:
                        instance_ids_new.append(instance_id)
            # 没有找到物体
            if len(instance_ids) == 0:
                if not ("Rotate" in action or "MoveAhead" in action or "Look" in action):
                    err = "Bad interact mask. Couldn't locate target object"
                    print("Went through bad interaction mask")
                    success = False
                    # rgb = self.last_event.frame.copy()  # shape (h, w, 3)
                    # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    # depth = self.last_event.depth_frame.copy()
                    # return state, 0, False, self.info, False, None, "", err, None
                    rgb, depth_frame, mask_list, info_dict = self.get_obs()
                    return rgb, depth_frame, mask_list, info_dict
                else:
                    target_instance_id = ""
            if len(instance_ids) != 0:
                target_instance_id = instance_ids[0]
        else:
            target_instance_id = ""
        if "SinkBasin" == target_arg:
            target_instance_id = self.get_nearest_object_by_type(target_arg)
        # 如果进入函数, 证明已经找到物体
        if target_instance_id == "":
            target_instance_id = self.get_nearest_object_by_type(target_arg)
        # 尝试进行交互
        # 解析巡航动作
        if "GotoLocation" in action and target_position is not None:
            action = self.parse_nav_action(target_position)
        try:
            # obs, rew, done, infos, event, api_action = self.to_thor_api_exec(action, target_instance_id, smooth_nav)
            if target_object_id is not None:
                target_instance_id = target_object_id
            print("交互物体: ", target_instance_id)
            rgb, depth_frame, mask_list, info_dict = self.to_thor_api_exec(action, target_instance_id, smooth_nav, force)
        except Exception as err:
            success = False
            # rgb = self.last_event.frame.copy()  # shape (h, w, 3)
            # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # depth = self.last_event.depth_frame.copy()
            rgb, depth_frame, mask_list, info_dict = self.get_obs()
            return rgb, depth_frame, mask_list, info_dict
        # 判断操作是否在模拟器中可行
        if not self.last_event.metadata['lastActionSuccess']:
            success = False
            if len(self.last_event.metadata['errorMessage']) > 0:
                print("action that causes below error is ", action)
                self.err_message = self.last_event.metadata["errorMessage"]
            return rgb, depth_frame, mask_list, info_dict
        success = True
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
        elif "CookObject" in action:
            action = dict(action="CookObject", objectId=object_id,
                forceAction=True)
            # obs, rew, done, info = self.step(action, smooth_nav=smooth_nav)
            event = self.step(action)
        elif "LookUp" in action:
            # if abs(self.event.metadata['agent']['cameraHorizon']-0) <5:
            if abs(self.camera_horizon - 0) < 5:
                action = dict(action="LookUp_0",
                    forceAction=True)
            else:
                action = dict(action=action,
                    forceAction=True)
            # obs, rew, done, info = self.step(action, smooth_nav=smooth_nav)
            event = self.step(action)
        elif "LookDown" in action:
            # if abs(self.event.metadata['agent']['cameraHorizon'] - 90) <5:
            if abs(self.camera_horizon - 90) < 5:
                action = dict(action="LookDown_0",
                    forceAction=True)
            else:
                action = dict(action=action,
                    forceAction=True)
            # obs, rew, done, info = self.step(action, smooth_nav=smooth_nav)
            event = self.step(action)
        elif "Clean0bject" in action:
            action = dict(action="CleanObject", objectId=object_id,
                forceAction=True)
            event = self.step(action)
        elif "Heatobject" in action or "Coolobject" in action:
            action = dict(action="SetRoomTempDecayTimeForType", objectType=object_id.split("|")[0],
                TimeUntilRoomTemp=10000)
            event = self.step(action)
        elif "OpenObject" in action:
            action = dict(action="OpenObject",
                objectId=object_id,
                moveMagnitude=1.0, forceAction=forceAction)
            # obs, rew, done, info = self.step(action)
            event = self.step(action)

        elif "CloseObject" in action:
            action = dict(action="CloseObject",
                objectId=object_id,
                forceAction=forceAction)
            # obs, rew, done, info = self.step(action)
            event = self.step(action)
        elif "PickupObject" in action:
            action = dict(action="PickupObject",
                objectId=object_id, forceAction=forceAction)
            # obs, rew, done, info = self.step(action)
            event = self.step(action)

            # add information
            self.picked_up = True
            self.picked_up_cat = self.total_cat2idx[object_id.split("|")[0]]
            object_name = object_id.split("|")[0]
            if self.args.use_sem_seg:
                self.picked_up_mask = self.seg.sem_seg_get_instance_mask_from_obj_type_largest_only(object_name)
            else:
                self.picked_up_mask = self.seg.get_instance_mask_from_obj_type_largest(object_name)

        elif "PutObject" in action:
            inventory_object_id = self.last_event.metadata['inventoryObjects'][0]['objectId']
            # new_inventory_object_id = self.last_event.metadata['inventoryObjects'][0]['objectId']
            action = dict(action="PutObject",
                objectId=object_id,
                forceAction=True,
                placeStationary=True)
            event = self.step(action)
            # obs, rew, done, info = self.step(action)
            # for i in range(4):
            #     action_turn = dict(action="RotateRight", degrees="90",
            #         forceAction=True)
            #     event = self.step(action_turn)
            #     event = self.step(action)
            #     if self.last_event.metadata["lastActionSuccess"]:
            #         break
            # objects_in_hand = self.last_event.metadata['inventoryObjects']
            # if len(objects_in_hand) > 0:
            #     # 扔掉手里物体
            #     event = self.step(
            #         action="MoveHeldObjectAhead",
            #         moveMagnitude=0.2,
            #         forceVisible=True
            #     )
            #     drop_object_acion = dict(action="DropHandObject", forceAction=True)
            #     event = self.step(drop_object_acion)

            # add information
            self.picked_up = False
            self.picked_up_cat = None
            self.picked_up_mask = None
            action = dict(action="PutObject", objectId=inventory_object_id, forceAction=True, placeStationary=True, receptacleObjectId=object_id)
        elif "ToggleObjectOn" in action:
            action = dict(action="ToggleObjectOn",
                objectId=object_id, forceAction=forceAction)
            # obs, rew, done, info = self.step(action)
            event = self.step(action)
        elif "ToggleObjectOff" in action:
            action = dict(action="ToggleObjectOff",
                objectId=object_id, forceAction=forceAction)
            # obs, rew, done, info = self.step(action)
            event = self.step(action)
        elif "SliceObject" in action:
            # check if agent is holding knife in hand
            inventory_objects = self.last_event.metadata['inventoryObjects']
            if len(inventory_objects) == 0 or 'Knife' not in inventory_objects[0]['objectType']:
                raise Exception("Agent should be holding a knife before slicing.")
            action = dict(action="SliceObject",
                objectId=object_id, forceAction=forceAction)
            # obs, rew, done, info = self.step(action)
            event = self.step(action)
        elif "DisableObject" in action:
            action = dict(action="DisableObject",
                objectId=object_id)
            event = self.step(action)
        elif "EnableObject" in action:
            action = dict(action="EnableObject",
                objectId=object_id)
            event = self.step(action)
        else:
            raise Exception("Invalid action. Conversion to THOR API failed! (action='" + str(action) + "')")
        if isinstance(action, dict):
            if action["action"] not in ["DisableObject", "EnableObject"]:
                self.exec_action_list.append(action)
        if self.last_event.metadata["errorMessage"] != '':
            self.api_fail_times += 1
        rgb, depth_frame, mask_list, info_dict = self.get_obs()
        done = self.get_done()
        self.interactive_object_id = object_id
        # 更新状态
        # clean object
        # sink_basin = self.get_obj_of_type_closest_to_obj_easy('SinkBasin', object_id)
        sink_basin = self.get_obj_of_type_closest_to_obj_easy(ref_object_id=object_id, object_type='SinkBasin')
        if sink_basin is None:
            cleaned_object_ids = None
        else:
            cleaned_object_ids = sink_basin['receptacleObjectIds']
        if cleaned_object_ids is None:
            cleaned_object_ids = []
        # self.cleaned_objects = self.cleaned_objects | set(cleaned_object_ids) if cleaned_object_ids is not None else set()
        self.cleaned_objects = self.cleaned_objects | set(cleaned_object_ids)
        # print("清理物体:", self.cleaned_objects)
        # heat object
        microwave = self.get_objects_of_type('Microwave', self.last_event.metadata)
        if len(microwave) > 0:
            heated_object_ids = microwave[0]['receptacleObjectIds']
        else:
            heated_object_ids = None
        # self.heated_objects = self.heated_objects | set(heated_object_ids) if heated_object_ids is not None else set()
        if heated_object_ids is None:
            heated_object_ids = []
        self.heated_objects = self.heated_objects | set(heated_object_ids)
        # cool object
        fridge = self.get_objects_of_type('Fridge', self.last_event.metadata)
        if len(fridge) > 0:
            cooled_object_ids = fridge[0]['receptacleObjectIds']
        else:
            cooled_object_ids = None
        if cooled_object_ids is None:
            cooled_object_ids = []
        # self.cooled_objects = self.cooled_objects | set(cooled_object_ids) if cooled_object_ids is not None else set()
        self.cooled_objects = self.cooled_objects | set(cooled_object_ids)
        # self.last_inter_object_id = object_id

        return rgb, depth_frame, mask_list, info_dict

    def get_object(self, object_id, metadata):
        for obj in metadata['objects']:
            if obj['objectId'] == object_id:
                return obj
        return None

    def get_objects_of_type(self, object_type, metadata):
        return [obj for obj in metadata['objects'] if obj['objectType'] == object_type]

    def get_obj_of_type_closest_to_obj(self, ref_object_id, object_type):
        metadata = self.last_event.metadata
        objs_of_type = [obj for obj in metadata['objects'] if obj['objectType'] == object_type and obj['visible']]
        ref_obj = self.get_object(ref_object_id, metadata)
        closest_objs_of_type = sorted(objs_of_type, key=lambda o: np.linalg.norm(np.array([o['position']['x'],
                                                                                           o['position']['y'],
                                                                                           o['position']['z']]) - \
                                                                                 np.array([ref_obj['position']['x'],
                                                                                           ref_obj['position']['y'],
                                                                                           ref_obj['position']['z']])))
        # if len(closest_objs_of_type) == 0:
        #     raise Exception("No closest %s found!" % (ref_obj))
        if len(closest_objs_of_type) == 0:
            return None
        else:
            return closest_objs_of_type[0]  # retrun the first closest visible object

    def get_obj_of_type_closest_to_obj_easy(self, ref_object_id, object_type):
        metadata = self.last_event.metadata
        objs_of_type = [obj for obj in metadata['objects'] if obj['objectType'] == object_type]
        ref_obj = self.get_object(ref_object_id, metadata)
        if ref_obj is None:
            return None
        closest_objs_of_type = sorted(objs_of_type, key=lambda o: np.linalg.norm(np.array([o['position']['x'],
                                                                                           o['position']['y'],
                                                                                           o['position']['z']]) - \
                                                                                 np.array([ref_obj['position']['x'],
                                                                                           ref_obj['position']['y'],
                                                                                           ref_obj['position']['z']])))
        # if len(closest_objs_of_type) == 0:
        #     raise Exception("No closest %s found!" % (ref_obj))
        if len(closest_objs_of_type) == 0:
            return None
        else:
            return closest_objs_of_type[0]  # retrun the first closest visible object


    def get_done(self):
        # if self.info['time'] >= self.args.max_episode_length - 1:
        #     done = True
        if self.stopped:
            done = True
        else:
            done = False
        return done

    def get_instance_mask_from_obj_type(self, object_name, target_coord=None):
        scores, masks = [-10], [np.zeros((self.image_height, self.image_width))]
        for k, v in self.last_event.instance_masks.items():
            if "sink" in k.lower():
                if "Basin" not in k:
                    category = k.split("|")[0]
                else:
                    category = "SinkBasin"
            else:
                category = k.split('|')[0]
            category_last = k.split('|')[-1]
            if category == object_name:
                masks.append(v)
                score = 0 if (target_coord is None) else self.maskDistance(v, target_coord)
                scores.append(score)
        # 选取第一个或者最接近的物体作为目标
        mask = masks[np.argmax(scores)].astype(np.float_)
        if np.sum(mask) == 0:
            mask = None
        return mask

    def maskDistance(self, mask, ref_pt):
        cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ref_xy = (ref_pt[1], ref_pt[0])
        dist = cv2.pointPolygonTest(cnt[0], ref_xy, True)  # positive: inside the mask, negative: outside
        return dist

    def prune_by_any_interaction(self, instances_ids):
        '''
        ignores any object that is not interactable in anyway
        '''
        pruned_instance_ids = []
        for obj in self.last_event.metadata['objects']:
            obj_id = obj['objectId']
            if obj_id in instances_ids:
                if obj['pickupable'] or obj['receptacle'] or obj['openable'] or obj['toggleable'] or obj['sliceable']:
                    pruned_instance_ids.append(obj_id)

        ordered_instance_ids = [id for id in instances_ids if id in pruned_instance_ids]
        return ordered_instance_ids

    def update_object_input(self):
        visible_obj_id_list = []
        object_id2index = {}
        object_index2id = {}
        metadata = self.last_event.metadata
        object_metadata = metadata['objects']
        for obj_ind in range(len(object_metadata)):
            obj_data_one = object_metadata[obj_ind]
            obj_id = obj_data_one["objectId"]
            object_id2index[obj_id] = obj_ind
            object_index2id[obj_ind] = obj_id
        frame_object_dict = self.last_event.instance_masks
        for frame_object_id, frame_object_one in enumerate(list(frame_object_dict.keys())):
            if frame_object_one in object_id2index.keys():
                visible_obj_id_list.append(object_id2index[frame_object_one])

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
                # 增加具体物体属性
                new_object_dict[obj_Id]["receptacle"] = receptacle
                new_object_dict[obj_Id]["toggleable"] = toggleable
                new_object_dict[obj_Id]["cookable"] = cookable
                new_object_dict[obj_Id]["sliceable"] = sliceable
                new_object_dict[obj_Id]["openable"] = openable
                new_object_dict[obj_Id]["pickupable"] = pickupable
            else:
                new_object_dict[obj_Id]["class_name"] = obj_name
                new_object_dict[obj_Id]["pos"] = (obj_pos["x"], obj_pos["z"], obj_bbox_size["x"], obj_bbox_size["z"])
                # 增加具体物体属性
                new_object_dict[obj_Id]["receptacle"] = receptacle
                new_object_dict[obj_Id]["toggleable"] = toggleable
                new_object_dict[obj_Id]["cookable"] = cookable
                new_object_dict[obj_Id]["sliceable"] = sliceable
                new_object_dict[obj_Id]["openable"] = openable
                new_object_dict[obj_Id]["pickupable"] = pickupable

        # self.seen_object_dict = new_object_dict
        return new_object_dict

    def update_position_set(self):
        world_space_point_cloud = self.get_frame_pointcloud()
        # 进行点云降噪
        world_space_point_cloud = self.filter_point_clouds(world_space_point_cloud)
        initial_position = self.last_event.metadata['agent']['position']
        initial_position_tuple = (initial_position["x"], initial_position["y"], initial_position["z"])
        updated_position_list = []
        seen_updated_position_list = []
        for positions_tuple_id, positions_tuple_one in enumerate(self.position_tuple):
            is_within_range = self.check_isin_frame_workplace(world_space_point_cloud, positions_tuple_one, initial_position_tuple)
            if is_within_range:
                updated_position_list.append(positions_tuple_one)
            # is_within_range = self.check_isin_frame(world_space_point_cloud, positions_tuple_one, initial_position_tuple)
            # if is_within_range:
            #     seen_updated_position_list.append(positions_tuple_one)
        # self.updated_position_list = updated_position_list
        return updated_position_list, seen_updated_position_list
        # return None

    def get_frame_pointcloud(self):
        fov = self.last_event.metadata["fov"]
        cameraHorizon = self.last_event.metadata["agent"]["cameraHorizon"]
        camera_world_xyz = list(self.last_event.metadata["agent"]["position"].values())
        camera_world_xyz[1] = camera_world_xyz[1] + 0.675
        camera_world_xyz = np.asarray(camera_world_xyz)
        rotation = self.last_event.metadata["agent"]["rotation"]['y']
        depth_image = self.last_event.depth_frame
        depth_image = np.asarray(depth_image)
        camera_space_point_cloud = cpu_only_depth_frame_to_camera_space_xyz(depth_image, mask=None, fov=fov)
        world_space_point_cloud = cpu_only_camera_space_xyz_to_world_xyz(camera_space_point_cloud,
            camera_world_xyz, rotation, cameraHorizon)
        # world_space_point_cloud_xyz = world_space_point_cloud[:, :3]
        return world_space_point_cloud.T

    def filter_point_clouds(self, point_cloud):
        point_cloud = point_cloud[point_cloud[:, 0] > 0, :]
        point_cloud = point_cloud[point_cloud[:, 1] < 2.3, :]
        point_cloud = point_cloud[point_cloud[:, 2] > 0, :]
        return point_cloud

    def parse_nav_action(self, target_position):
        agent_start_position = self.last_event.metadata['agent']['position']
        agent_start_position_tuple = (agent_start_position["x"], agent_start_position["y"], agent_start_position["z"])
        target_position_tuple = (target_position[0], agent_start_position["y"], target_position[1])
        path = shortest_path(agent_start_position_tuple, target_position_tuple, self.neighbors, self.position_tuple)
        self.nav_path = path
        rotation = self.last_event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        action_list = get_action_list(path, rotation_y=start_rotation)
        # 增加最终停止符号
        action_list.extend(["Stop"])
        return action_list, path

    def parse_obj2obj_path(self, start_position_tuple, end_position_tuple):
        path = shortest_path(start_position_tuple, end_position_tuple, self.neighbors, self.position_tuple)
        self.nav_path = path
        rotation = self.last_event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        action_list = get_action_list(path, rotation_y=start_rotation)
        # 增加最终停止符号
        action_list.extend(["Stop"])
        return action_list, path

    def object_dict2str_llama_s1_only(self, target_name=""):
        new_object_list = "["
        # object_pos_dict = self.seen_object_dict
        object_pos_dict = self.seen_object_dict_all
        seen_object_list = []
        for obj_key_id, object_key in enumerate(list(object_pos_dict.keys())):
            obj_name = object_pos_dict[object_key]

            if "wall" in obj_name.lower() or "floor" in obj_name.lower() or "room" in obj_name.lower():
                continue
            seen_object_list.append(obj_name)
        seen_object_list = list(set(seen_object_list))
        for seen_obj_index, seen_obj_name in enumerate(seen_object_list):
            new_str = seen_obj_name + ", "
            new_object_list = new_object_list + new_str
        new_object_list = new_object_list.strip().strip(",") + "]"
        return new_object_list

    def object_dict2str_llama_s1_only_alfred(self, object_name_list=None):
        new_object_list = "["
        if object_name_list is None:
            seen_object_name_list_all = self.seen_object_name_list_all
        else:
            seen_object_name_list_all = object_name_list
        for seen_object_one in seen_object_name_list_all:
            new_str = seen_object_one + ", "
            new_object_list = new_object_list + new_str
        new_object_list = new_object_list.strip().strip(",") + "]"
        return new_object_list

    def object_dict2str_llama_s2_only(self, target_name=""):
        new_object_list = "["
        object_pos_dict = self.seen_object_dict
        for obj_key_id, object_key in enumerate(list(object_pos_dict.keys())):
            object_dict_one = object_pos_dict[object_key]
            obj_name = object_dict_one["class_name"]
            if "wall" in obj_name.lower() or "floor" in obj_name.lower() or "room" in obj_name.lower():
                continue
            obj_pos = object_dict_one["pos"]
            obj_pos_str = "(" + str(np.round(obj_pos[0], 2)) + ", " + str(np.round(obj_pos[1], 2)) + ")"
            new_str = obj_name + ": " + obj_pos_str + ", "
            new_object_list = new_object_list + new_str
        new_object_list = new_object_list.strip().strip(",") + "]"
        return new_object_list

    def look_around(self):
        action = dict(action="RotateLeft", degrees="90",
            forceAction=True)
        # obs, rew, done, info = self.step(action, smooth_nav=smooth_nav)
        event = self.step(action)

    def closest_grid_point(self, world_point: Tuple[float, float, float], positions_tuple) -> Tuple[float, float, float]:
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

    def shortest_path(self, start: Tuple[float, float, float], end: Tuple[float, float, float], neighbors, positions_tuple):
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

    def get_rotation(self, x0, z0, x1, z1, start_rotation):
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

    def get_targets(self, task_label_dict):
        '''
        returns a dictionary of all targets for the task
        '''
        targets = {
            'object': task_label_dict['object_target'],
            'parent': task_label_dict['parent_target'],
            'toggle': task_label_dict['toggle_target'],
            'mrecep': task_label_dict['mrecep_target'],
            "sliced": task_label_dict["object_sliced"]
        }

        return targets

    # def get_objects_with_name_and_prop(self, name, prop):
    #     metadata = self.last_event.metadata
    #     return [obj for obj in metadata['objects']
    #             if name in obj['objectId'] and obj[prop]]

    def get_objects_with_name_and_prop(self, name, prop):
        metadata = self.last_event.metadata
        if ',' in name:
            name = [obj_name.strip() for obj_name in name.split(',')]
            return [obj for obj in metadata['objects'] for i in range(len(name))
                    if name[i] in obj['objectId'] and obj[prop]]
        else:
            return [obj for obj in metadata['objects']
                    if name in obj['objectId'] and obj[prop]]

    def check_task(self, task_type=None, task_label_dict=None):
        success = False
        if task_type is None:
            return success
        if task_type == "look_at_obj_in_light":
            ts = 2
            s = 0
            targets = self.get_targets(task_label_dict)
            toggleables = self.get_objects_with_name_and_prop(targets['toggle'], 'toggleable')
            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
            inventory_objects = self.last_event.metadata['inventoryObjects']

            # check if the right object is in hand
            if len(inventory_objects) > 0 and inventory_objects[0]['objectId'] in [p['objectId'] for p in pickupables]:
                s += 1
            # check if the lamp is visible and turned on
            # if np.any([t['isToggled'] and t['visible'] for t in toggleables]):
            #     s += 1
            if np.any([t['isToggled'] for t in toggleables]):
                s += 1

            if targets["sliced"]:
                s_sa = 0
                ts += 1
                # check if some object was sliced
                objs_sliced = [t['objectId'] for t in pickupables if t["isSliced"]]
                if len(objs_sliced) > 0:
                    s_sa += 1
                for obj_id in objs_sliced:
                    if obj_id.split("|")[0] in str(inventory_objects):
                        s_sa += 1
                if s_sa >= 2:
                    s += 1

        elif task_type == "pick_and_place_simple":
            ts = 1
            s = 0

            targets = self.get_targets(task_label_dict)
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
            obj_in_place = [p['objectId'] for p in pickupables for r in receptacles
                            if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]

            if len(obj_in_place) > 0:
                s += 1

            if targets["sliced"]:
                s_sa = 0
                ts += 1
                # check if some object was sliced
                objs_sliced = [t['objectId'] for t in pickupables if t["isSliced"]]
                if len(objs_sliced) > 0:
                    s_sa += 1
                for obj_id in objs_sliced:
                    if obj_id.split("|")[0] in str(obj_in_place):
                        s_sa += 1
                if s_sa >= 2:
                    s += 1

        elif task_type == "pick_and_place_with_movable_recep":
            ts = 3
            s = 0

            targets = self.get_targets(task_label_dict)
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
            movables = self.get_objects_with_name_and_prop(targets['mrecep'], 'pickupable')

            pickup_in_place = [p for p in pickupables for m in movables
                               if 'receptacleObjectIds' in p and m['receptacleObjectIds'] is not None
                               and p['objectId'] in m['receptacleObjectIds']]
            movable_in_place = [m for m in movables for r in receptacles
                                if 'receptacleObjectIds' in r and r['receptacleObjectIds'] is not None
                                and m['objectId'] in r['receptacleObjectIds']]
            # check if the object is in the final receptacle
            if len(pickup_in_place) > 0:
                s += 1
            # check if the movable receptacle is in the final receptacle
            if len(movable_in_place) > 0:
                s += 1
            # check if both the object and movable receptacle stack is in the final receptacle
            if np.any([np.any([p['objectId'] in m['receptacleObjectIds'] for p in pickupables]) and
                       np.any([r['objectId'] in m['parentReceptacles'] for r in receptacles]) for m in movables
                       if m['parentReceptacles'] is not None and m['receptacleObjectIds'] is not None]):
                s += 1

            if targets["sliced"]:
                s_sa = 0
                ts += 1
                # check if some object was sliced
                objs_sliced = [t['objectId'] for t in pickupables if t["isSliced"]]
                if len(objs_sliced) > 0:
                    s_sa += 1
                for obj_id in objs_sliced:
                    if obj_id.split("|")[0] in str(pickup_in_place):
                        s_sa += 1
                if s_sa >= 2:
                    s += 1

        elif task_type == "pick_clean_then_place_in_recep":
            ts = 3
            s = 0

            targets = self.get_targets(task_label_dict)
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')

            objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                             if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
            objs_cleaned = [p['objectId'] for p in pickupables if p['objectId'] in self.cleaned_objects]

            # check if object is in the receptacle
            if len(objs_in_place) > 0:
                s += 1
            # check if some object was cleaned
            if len(objs_cleaned) > 0:
                s += 1
            # check if the object is both in the receptacle and clean
            if np.any([obj_id in objs_cleaned for obj_id in objs_in_place]):
                s += 1

            if targets["sliced"]:
                s_sa = 0
                ts += 1
                # check if some object was sliced
                objs_sliced = [t['objectId'] for t in pickupables if t["isSliced"]]
                if len(objs_sliced) > 0:
                    s_sa += 1
                for obj_id in objs_sliced:
                    if obj_id.split("|")[0] in str(objs_in_place) and str(objs_cleaned):
                        s_sa += 1
                if s_sa >= 2:
                    s += 1

        elif task_type == "pick_cool_then_place_in_recep":
            ts = 3
            s = 0

            targets = self.get_targets(task_label_dict)
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
            objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                             if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
            objs_cooled = [p['objectId'] for p in pickupables if p['objectId'] in self.cooled_objects]

            # check if object is in the receptacle
            if len(objs_in_place) > 0:
                s += 1
            # check if some object was cooled
            if len(objs_cooled) > 0:
                s += 1
            # check if the object is both in the receptacle and cold
            if np.any([obj_id in objs_cooled for obj_id in objs_in_place]):
                s += 1

            if targets["sliced"]:
                s_sa = 0
                ts += 1
                # check if some object was sliced
                objs_sliced = [t['objectId'] for t in pickupables if t["isSliced"]]
                if len(objs_sliced) > 0:
                    s_sa += 1
                for obj_id in objs_sliced:
                    if obj_id.split("|")[0] in str(objs_in_place) and str(objs_cooled):
                        s_sa += 1
                if s_sa >= 2:
                    s += 1

        elif task_type == "pick_heat_then_place_in_recep":
            ts = 3
            s = 0

            targets = self.get_targets(task_label_dict)
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')

            objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                             if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
            objs_heated = [p['objectId'] for p in pickupables if p['objectId'] in self.heated_objects]

            # check if object is in the receptacle
            if len(objs_in_place) > 0:
                s += 1
            # check if some object was heated
            if len(objs_heated) > 0:
                s += 1
            # check if the object is both in the receptacle and hot
            if np.any([obj_id in objs_heated for obj_id in objs_in_place]):
                s += 1

            if targets["sliced"]:
                s_sa = 0
                ts += 1
                # check if some object was sliced
                objs_sliced = [t['objectId'] for t in pickupables if t["isSliced"]]
                if len(objs_sliced) > 0:
                    s_sa += 1
                for obj_id in objs_sliced:
                    if obj_id.split("|")[0] in str(objs_in_place) and str(objs_heated):
                        s_sa += 1
                if s_sa >= 2:
                    s += 1

        elif task_type == "pick_two_obj_and_place":
            ts = 2
            s = 0

            targets = self.get_targets(task_label_dict)
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
            if "," in targets["object"]:
                targets["object"] = targets["object"].split(",")
                pickupables_1 = self.get_objects_with_name_and_prop(targets['object'][0].strip(), 'pickupable')
                pickupables_2 = self.get_objects_with_name_and_prop(targets['object'][1].strip(), 'pickupable')

                obj_in_place_1 = [p['objectId'] for p in pickupables_1 for r in receptacles
                                  if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
                obj_in_place_2 = [p['objectId'] for p in pickupables_2 for r in receptacles
                                  if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]

                if np.any([np.any([p['objectId'] in r['receptacleObjectIds']
                                   for r in receptacles if r['receptacleObjectIds'] is not None])
                           for p in pickupables_1]):
                    s += 1

                if np.any([np.any([p['objectId'] in r['receptacleObjectIds']
                                   for r in receptacles if r['receptacleObjectIds'] is not None])
                           for p in pickupables_2]):
                    s += 1

                if targets["sliced"]:
                    s_sa_1 = 0
                    s_sa_2 = 0
                    ts += 2
                    # check if some object was sliced
                    objs_sliced = [t['objectId'] for t in pickupables_1 if t["isSliced"]]
                    if len(objs_sliced) > 0:
                        s_sa_1 += 1
                    for obj_id in objs_sliced:
                        if obj_id.split("|")[0] in str(obj_in_place_1):
                            s_sa_1 += 1
                    objs_sliced = [t['objectId'] for t in pickupables_2 if t["isSliced"]]
                    if len(objs_sliced) > 0:
                        s_sa_2 += 1
                    for obj_id in objs_sliced:
                        if obj_id.split("|")[0] in str(obj_in_place_2):
                            s_sa_2 += 1
                    if s_sa_1 >= 2:
                        s += 1
                    if s_sa_2 >= 2:
                        s += 1
            else:
                pickupables = self.get_objects_with_name_and_prop(targets['object'].strip(), 'pickupable')
                obj_in_place = [p['objectId'] for p in pickupables for r in receptacles
                                if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
                if len(receptacles) < 1:
                    s += 0
                else:
                    s += min(np.max([sum([1 if r['receptacleObjectIds'] is not None
                                               and p['objectId'] in r['receptacleObjectIds'] else 0
                                          for p in pickupables])
                                     for r in receptacles]), 2)

                if targets["sliced"]:
                    s_sa_1 = 0
                    s_sa_2 = 0
                    ts += 2
                    objs_sliced = [t['objectId'] for t in pickupables if t["isSliced"]]
                    s_sa_1 += min(len([p for p in pickupables if p["isSliced"]]), 2)
                    for obj_id in objs_sliced:
                        if obj_id.split("|")[0] in str(obj_in_place):
                            s_sa_2 += 1
                    s += min(min(s_sa_1, s_sa_2), 2)

        else:
            return success

        if ts == s:
            success = True

        response_dict = {"success": success,
                         "ts": ts,
                         "s": s}

        return response_dict

    def get_candidate_by_name(self, target_name, action_name=None):
        # 增加每个动作的候选
        action_properties_dict = {
            "PickupObject": "pickupable",
            "OpenObject": "openable",
            "CookObject": "cookable",
            "CloseObject": "openable",
            "SliceObject": "sliceable",
            "PutObject": "receptacle",
            "ToggleObjectOn": "toggleable",
            "ToggleObjectOff": "toggleable"
        }

        # 对sink进行额外处理
        if "sink" in target_name.lower():
            target_name = "SinkBasin"

        target_object_property = None
        if action_name is not None:
            if action_name in action_properties_dict.keys():
                target_object_property = action_properties_dict[action_name]

        candidate_position_list = []
        candidate_target_obj_id_list = []
        object_dict = self.object_dict
        for obj_id, obj_dict_one in object_dict.items():
            if "sink" in target_name.lower():
                obj_name_one = obj_dict_one["objectType"]
            else:
                obj_name_one = obj_dict_one["objectId"].split("|")[0]
            position_dict_one = obj_dict_one["position"]
            position_tuple = (position_dict_one["x"], position_dict_one["z"])
            # 必须要相同
            if obj_name_one == target_name:
                # if target_name.lower() in obj_name_one.lower():
                if target_object_property is None:
                    candidate_position_list.append(position_tuple)
                    candidate_target_obj_id_list.append(obj_id)
                else:
                    if obj_dict_one[target_object_property]:
                        candidate_position_list.append(position_tuple)
                        candidate_target_obj_id_list.append(obj_id)

        return candidate_position_list, candidate_target_obj_id_list

    def get_candidate_id_by_object_id(self, candidate_target_obj_id_list, used_object_id=None):
        # 根据生成的候选, 判断所观察的frame中是否有物体
        candidate_index_list = []
        object_id_list = list(self.seen_object_dict_all.keys())
        for candidate_index, candidate_id in enumerate(candidate_target_obj_id_list):
            if candidate_id in object_id_list:
                if used_object_id is not None:
                    if candidate_id not in used_object_id:
                        candidate_index_list.append(candidate_index)
                else:
                    candidate_index_list.append(candidate_index)

        return candidate_index_list

    def seen_object_name(self):
        object_name_list = []
        for k, v in self.seen_object_dict_all.items():
            object_name_list.append(k.split("|")[0])
        object_name_list = list(set(object_name_list))
        return object_name_list

    def seen_object_name_alfred(self):
        return self.seen_object_name_list_all

    def open_door(self):
        # openable
        object_metadata = self.last_event.metadata["objects"]
        for obj_ind in range(len(object_metadata)):
            obj_name = object_metadata[obj_ind]["objectId"].split("|")[0]
            obj_id = object_metadata[obj_ind]["objectId"]
            if "door" in obj_name.lower():
                open_able = object_metadata[obj_ind]["openable"]
                if open_able:
                    open_action = dict(action="OpenObject", objectId=obj_id, forceAction=True)
                    event = self.step(open_action)

    def check_target_frame(self, target_object_id):

        is_find = False
        look_action_list = [{"action": "LookUp", "degrees": 30},
                            {"action": "LookDown", "degrees": 60},
                            {"action": "LookUp", "degrees": 30}]
        for i in range(4):
            is_find_in_frame = self.check_isin_frame(target_object_id)
            rgb, depth_frame, mask_list, info_dict = self.get_obs()
            if is_find_in_frame:
                is_find = True
                break
            for action_id, action_dict_one in enumerate(look_action_list):
                action_name = action_dict_one["action"]
                degrees = action_dict_one["degrees"]
                event = self.step(action=action_name, degrees=degrees)
                # metadata = controller.last_event.metadata
                is_find_in_frame = self.check_isin_frame(target_object_id)
                rgb, depth_frame, mask_list, info_dict = self.get_obs()
                if is_find_in_frame:
                    is_find = True
                    break
            if is_find:
                break
            # 增加转圈
            event = self.step(action="RotateRight", forceAction=True)
        return is_find

    def check_target_frame_alfred(self, target_name):
        is_find = False
        if self.args.alfred_scene:
            look_action_list = []
        else:
            look_action_list = [{"action": "LookUp", "degrees": 30},
                                {"action": "LookDown", "degrees": 60},
                                {"action": "LookUp", "degrees": 30}]
        for i in range(4):
            rgb, depth_frame, mask_list, info_dict = self.get_obs()
            is_find_in_frame = self.check_isin_frame_alfred(target_name)
            if is_find_in_frame:
                is_find = True
                break
            for action_id, action_dict_one in enumerate(look_action_list):
                action_name = action_dict_one["action"]
                degrees = action_dict_one["degrees"]
                event = self.step(action=action_name, degrees=degrees)
                # add action info
                self.exec_action_list.append(dict(action=action_name, degrees=degrees, forceAction=True))
                rgb, depth_frame, mask_list, info_dict = self.get_obs()
                # metadata = controller.last_event.metadata
                is_find_in_frame = self.check_isin_frame_alfred(target_name)
                if is_find_in_frame:
                    is_find = True
                    break
            if is_find:
                break
            # 增加转圈
            event = self.step(action="RotateRight", forceAction=True)
            self.exec_action_list.append(dict(action="RotateRight", degrees=90, forceAction=True))
        return is_find



    def get_frame_object_list(self):
        visible_obj_id_list = []
        object_id2index = {}
        object_index2id = {}
        event = self.last_event
        metadata = event.metadata
        object_metadata = metadata['objects']
        for obj_ind in range(len(object_metadata)):
            obj_data_one = object_metadata[obj_ind]
            obj_id = obj_data_one["objectId"]
            object_id2index[obj_id] = obj_ind
            object_index2id[obj_ind] = obj_id
        # frame_object_dict = event.instance_masks
        frame_object_dict = event.instance_detections2D
        for frame_object_id, frame_object_one in enumerate(list(frame_object_dict.keys())):
            if frame_object_one in object_id2index.keys():
                # add check bbox
                bbox_one = frame_object_dict[frame_object_one]
                center_x = (bbox_one[0] + bbox_one[2]) / 2
                center_y = (bbox_one[1] + bbox_one[3]) / 2
                if 50 <= center_x <= self.image_width-50 and 50 <= center_y <= self.image_height-50:
                    visible_obj_id_list.append(object_id2index[frame_object_one])

        frame_object_dict = {}
        for obj_ind in visible_obj_id_list:
            obj_Id = object_metadata[obj_ind]["objectId"]
            obj_name = object_metadata[obj_ind]["objectId"].split("|")[0]
            frame_object_dict[obj_Id] = obj_name

        return frame_object_dict

    def check_isin_frame(self, target_object_id):
        frame_object_dict = self.get_frame_object_list()
        object_id_list = list(frame_object_dict.keys())
        if target_object_id in object_id_list:
            return True
        else:
            return False

    def check_isin_frame_alfred(self, target_name):
        # 直接启动obs
        # _, _, _, _ = self.get_obs()
        # seen_object_name_list_all = self.seen_object_name_list_all
        frame_object_name_list = self.frame_object_name_list
        if target_name in frame_object_name_list:
            return True
        else:
            return False

    def check_action_success_before_execute(self, action_name, target_object_id):
        planning_action_success = False
        if target_object_id is None:
            planning_action_success = True
            return planning_action_success

        object_dict = self.object_dict
        agent_position_dict = self.last_event.metadata["agent"]["position"]
        agent_position_tuple = (agent_position_dict["x"], agent_position_dict["y"], agent_position_dict["z"])
        target_position_dict = object_dict[target_object_id]["position"]
        target_position_tuple = (target_position_dict["x"], target_position_dict["y"], target_position_dict["z"])
        if "PickupObject" in action_name:
            # check距离
            distance = np.sqrt((target_position_tuple[0] - agent_position_tuple[0]) ** 2 +
                               (target_position_tuple[2] - agent_position_tuple[2]) ** 2)
            if distance < 2:
                # if controller_env.last_event.metadata["lastActionSuccess"]:
                planning_action_success = True
        elif "PutObject" in action_name:
            # 首先判断手上有没有物体
            inventory_object_list = self.last_event.metadata['inventoryObjects']
            if len(inventory_object_list):
                planning_action_success = True
        else:
            planning_action_success = True
        return planning_action_success

    def check_is_near_target(self, target_object_id):
        is_near = False
        if target_object_id is None:
            is_near = True
            return is_near

        object_dict = self.object_dict
        agent_position_dict = self.last_event.metadata["agent"]["position"]
        agent_position_tuple = (agent_position_dict["x"], agent_position_dict["y"], agent_position_dict["z"])
        target_position_dict = object_dict[target_object_id]["position"]
        target_position_tuple = (target_position_dict["x"], target_position_dict["y"], target_position_dict["z"])

        distance = np.sqrt((target_position_tuple[0] - agent_position_tuple[0]) ** 2 +
                           (target_position_tuple[2] - agent_position_tuple[2]) ** 2)
        if distance < 2:
            # if controller_env.last_event.metadata["lastActionSuccess"]:
            is_near = True
        return is_near

    def goto_location_nav(self, target_nav_position, sem_map=None):
        nav_action, path = self.parse_nav_action(target_nav_position)
        path_distance = (len(path) - 1) * 0.25
        for nav_action_id, action_one in enumerate(tqdm(nav_action)):
            rgb, depth_frame, mask_list, info_dict = self.to_thor_api_exec([action_one])
            if sem_map is not None:
                if nav_action_id % 5 == 0:
                    _, _ = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update", only_seg=False)
                else:
                    _, _ = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update", only_seg=True)
        return path_distance

    def goto_location_nav_teleport(self, target_nav_position):
        nav_action, path = self.parse_nav_action(target_nav_position)
        is_arrive = self.teleport_nav_new(path[-1])
        return is_arrive

    def prepare_object(self, action, target_object_id):
        # 主要针对Pickup与Put
        is_prepare = False
        if "PickupObject" in [action]:
            self.step(
                action="DisableObject",
                objectId=target_object_id
            )
            is_prepare = True
        elif "PutObject" in [action]:
            self.step(
                action="EnableObject",
                objectId=target_object_id
            )
            is_prepare = True
        else:
            pass
        rgb, depth_frame, mask_list, info_dict = self.get_obs()
        return rgb, depth_frame, mask_list, info_dict, is_prepare

    def get_llama_input(self, target_name=""):
        # object_name_str = self.object_dict2str_llama(target_name)
        object_name_str = self.object_dict2str_llama_s2_only(target_name)
        position_str, position_list = self.position_list2str()
        agent_position_dict = self.last_event.metadata["agent"]["position"]
        agent_position_str = "(" + str(np.round(agent_position_dict["x"], 2)) + ", " + str(np.round(agent_position_dict["z"], 2)) + ")"

        return object_name_str, position_str, agent_position_str, position_list

    def position_list2str(self, distance_constraint=False):
        # 获取可以看到物体的所有位置
        position_set_list = []
        for obj in self.last_event.metadata["objects"]:
            if obj["objectId"] in self.seen_object_dict_all.keys():
                position_one = obj["position"]
                obj_position_tuple = (position_one["x"], position_one["y"], position_one["z"])
                obj_nearest_point = self.closest_grid_point(obj_position_tuple, self.position_tuple)
                position_set_list.append(obj_nearest_point)
        random.shuffle(position_set_list)
        position_list = []
        new_position_set = "["
        agent_position_dict = self.last_event.metadata["agent"]["position"]
        agent_position_tuple = (agent_position_dict["x"], agent_position_dict["y"], agent_position_dict["z"])
        for position_set_id, position_set_one in enumerate(position_set_list):
            # 计算与agent的距离, 争取在一个圈边上
            # distance = math.sqrt((position_set_one[0] - agent_position_tuple[0]) ** 2 + (position_set_one[2] - agent_position_tuple[2]) ** 2)
            # print(distance)
            # if math.isclose(distance, 1.45, rel_tol=0.3):
            pos_str = "(" + str(np.round(position_set_one[0], 2)) + ", " + str(np.round(
                position_set_one[2], 2)) + ")"
            new_str = pos_str + ", "
            new_position_set = new_position_set + new_str
            position_list.append((np.round(position_set_one[0], 2), np.round(position_set_one[2], 2)))
            if position_set_id >= 17:
                break
        new_position_set = new_position_set.strip(", ") + "]"
        return new_position_set, position_list

    def llm_planner_step(self, action_str, sem_map):
        event = None
        self.feedback = "Nothing happens."
        try:
            cmd = self.parse_command(action_str)
            if cmd['action'] == self.Action.GOTO:
                target = cmd['tar']
                candidate_position_list, candidate_target_obj_id_list = self.get_candidate_by_name(target)
                candidate_index_list = self.get_candidate_id_by_object_id(candidate_target_obj_id_list)
                if len(candidate_index_list) < 1:
                    return self.feedback
                select_candidate_index = candidate_index_list[0]
                target_nav_position = candidate_position_list[select_candidate_index]
                target_object_id = candidate_target_obj_id_list[select_candidate_index]
                path_distance = self.goto_location_nav(target_nav_position, sem_map)
                if target_object_id not in self.receptacles.keys():
                    return self.feedback
                # feedback conditions
                curr_recep = self.receptacles[target_object_id]
                self.curr_recep = curr_recep["objectType"]
                loc_id = list(self.receptacles.keys()).index(target_object_id)
                loc_feedback = "You arrive at loc %s. " % loc_id
                state_feedback = "The {} is {}. ".format(curr_recep["objectType"], "open" if curr_recep["isOpen"] else "closed") if \
                curr_recep["openable"] else ""
                loc_state_feedback = loc_feedback + state_feedback
                self.feedback = loc_state_feedback + self.feedback if "closed" not in state_feedback else loc_state_feedback
            elif cmd['action'] == self.Action.PICK:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                object = self.get_object_of_name(obj)
                if object is not None:
                    event = self.step({'action': "PickupObject",
                                           'objectId': object["objectId"],
                                           'forceAction': True})
                    rgb, depth_frame, mask_list, info_dict = self.get_obs()
                    self.feedback = "You pick up the %s from the %s." % (obj, tar)
            elif cmd['action'] == self.Action.PUT:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                recep = self.get_object_of_name(tar, self.receptacles)
                if recep is not None:
                    event = self.step({'action': "PutObject",
                                           'objectId': self.last_event.metadata['inventoryObjects'][0]['objectId'],
                                           'receptacleObjectId': recep["objectId"],
                                           'forceAction': True})
                    if event.metadata['lastActionSuccess']:
                        self.feedback = "You put the %s %s the %s." % (obj, rel, tar)
                    rgb, depth_frame, mask_list, info_dict = self.get_obs()
            elif cmd['action'] == self.Action.OPEN:
                target = cmd['tar']
                recep = self.get_object_of_name(target, self.receptacles)
                if recep is not None:
                    event = self.step({'action': "OpenObject",
                                           'objectId': recep["objectId"],
                                           'forceAction': True})
                    action_feedback = "You open the %s. The %s is open. " % (target, target)
                    self.feedback = action_feedback + self.feedback.replace("On the %s" % target, "In it")
                    rgb, depth_frame, mask_list, info_dict = self.get_obs()
            elif cmd['action'] == self.Action.CLOSE:
                target = cmd['tar']
                recep = self.get_object_of_name(target, self.receptacles)
                if recep is not None:
                    event = self.step({'action': "CloseObject",
                                           'objectId': recep["objectId"],
                                           'forceAction': True})
                    self.feedback = "You close the %s." % target
                    rgb, depth_frame, mask_list, info_dict = self.get_obs()
            elif cmd['action'] == self.Action.TOGGLE:
                target = cmd['tar']
                obj = self.get_object_of_name(target)
                if obj is not None:
                    event = self.step({'action': "ToggleObjectOn",
                                           'objectId': obj['objectId'],
                                           'forceAction': True})
                    self.feedback = "You turn on the %s." % target
            elif cmd['action'] == self.Action.HEAT:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                obj_id = self.last_event.metadata['inventoryObjects'][0]['objectId']
                recep = self.get_object_of_name(tar, self.receptacles)
                if recep is not None:
                    # open the microwave, heat the object, take the object, close the microwave
                    events = []
                    events.append(self.step({'action': 'OpenObject', 'objectId': recep['objectId'],
                                                 'forceAction': True}))
                    events.append(self.step({'action': 'PutObject', 'objectId': obj_id,
                                                 'receptacleObjectId': recep['objectId'], 'forceAction': True}))
                    events.append(self.step({'action': 'CloseObject', 'objectId': recep['objectId'],
                                                 'forceAction': True}))
                    events.append(self.step({'action': 'ToggleObjectOn', 'objectId': recep['objectId'],
                                                 'forceAction': True}))
                    events.append(self.step({'action': 'Pass'}))
                    events.append(self.step({'action': 'ToggleObjectOff', 'objectId': recep['objectId'],
                                                 'forceAction': True}))
                    events.append(self.step({'action': 'OpenObject', 'objectId': recep['objectId'],
                                                 'forceAction': True}))
                    events.append(self.step({'action': 'PickupObject', 'objectId': obj_id, 'forceAction': True}))
                    events.append(self.step({'action': 'CloseObject', 'objectId': recep['objectId'],
                                                 'forceAction': True}))

                    if all(e.metadata['lastActionSuccess'] for e in events) and self.curr_recep == tar:
                        self.feedback = "You heat the %s using the %s." % (obj, tar)
            elif cmd['action'] == self.Action.CLEAN:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                object = self.last_event.metadata['inventoryObjects'][0]
                sink = self.get_obj_cls_from_metadata('BathtubBasin' if "bathtubbasin" in tar else "SinkBasin")
                faucet = self.get_obj_cls_from_metadata('Faucet')
                if sink is not None and faucet is not None:
                    # put the object in the sink, turn on the faucet, turn off the faucet, pickup the object
                    events = []
                    events.append(self.step({'action': 'PutObject', 'objectId': object['objectId'],
                                                 'receptacleObjectId': sink['objectId'], 'forceAction': True}))
                    events.append(self.step({'action': 'ToggleObjectOn', 'objectId': faucet['objectId'],
                                                 'forceAction': True}))
                    events.append(self.step({'action': 'Pass'}))
                    events.append(self.step({'action': 'ToggleObjectOff', 'objectId': faucet['objectId'],
                                                 'forceAction': True}))
                    events.append(self.step({'action': 'PickupObject', 'objectId': object['objectId'],
                                                 'forceAction': True}))

                    if all(e.metadata['lastActionSuccess'] for e in events) and self.curr_recep == tar:
                        self.feedback = "You clean the %s using the %s." % (obj, tar)

            elif cmd['action'] == self.Action.COOL:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                object = self.last_event.metadata['inventoryObjects'][0]
                fridge = self.get_obj_cls_from_metadata('Fridge')

                # open the fridge, put the object inside, close the fridge, open the fridge, pickup the object
                events = []
                events.append(self.step({'action': 'OpenObject', 'objectId': fridge['objectId'],
                                             'forceAction': True}))
                events.append(self.step({'action': 'PutObject', 'objectId': object['objectId'],
                                             'receptacleObjectId': fridge['objectId'], 'forceAction': True}))
                events.append(self.step({'action': 'CloseObject', 'objectId': fridge['objectId'],
                                             'forceAction': True}))
                events.append(self.step({'action': 'Pass'}))
                events.append(self.step({'action': 'OpenObject', 'objectId': fridge['objectId'],
                                             'forceAction': True}))
                events.append(self.step({'action': 'PickupObject', 'objectId': object['objectId'],
                                             'forceAction': True}))
                events.append(self.step({'action': 'CloseObject', 'objectId': fridge['objectId'],
                                             'forceAction': True}))

                if all(e.metadata['lastActionSuccess'] for e in events) and self.curr_recep == tar:
                    self.feedback = "You cool the %s using the %s." % (obj, tar)
            elif cmd['action'] == self.Action.SLICE:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                object = self.get_object_of_name(obj)
                inventory_objects = self.last_event.metadata['inventoryObjects']
                if 'Knife' in inventory_objects[0]['objectType']:
                    event = self.step({'action': "SliceObject",
                                           'objectId': object['objectId']})
                    self.feedback = "You slice %s with the %s" % (obj, tar)

            elif cmd['action'] == self.Action.INVENTORY:
                if len(self.last_event.metadata['inventoryObjects']) > 0:
                    self.feedback = "You are carrying: a %s" % (self.last_event.metadata['inventoryObjects'][0])
                else:
                    self.feedback = "You are not carrying anything."

            elif cmd['action'] == self.Action.EXAMINE:
                target = cmd['tar']
                receptacle = self.get_object_of_name(target, self.receptacles)
                object = self.get_object_of_name(target)

                # if receptacle:
                #     self.visible_objects, self.feedback = self.print_frame(receptacle, self.curr_loc)
                #     self.frame_desc = str(self.feedback)
                # elif object:
                #     self.feedback = self.print_object(object)

            elif cmd['action'] == self.Action.LOOK:
                if self.curr_recep == "nothing":
                    self.feedback = "You are in the middle of a room. Looking quickly around you, you see nothing."
                else:
                    self.feedback = "You are facing the %s. Next to it, you see nothing." % self.curr_recep
        except:
            pass

        if event and not event.metadata['lastActionSuccess']:
            self.feedback = "Nothing happens."
        return self.feedback

    def get_object_of_name(self, name, object_dict=None):
        if object_dict is None:
            vis_objs = [obj for obj in self.last_event.metadata['objects'] if obj['visible']]
        else:
            vis_objs = [obj for obj in object_dict if obj['visible']]
        for obj in vis_objs:
            if obj["objectType"] == name:
                return obj
        return None

    def get_obj_cls_from_metadata(self, name):
        objs = [obj for obj in self.last_event.metadata['objects'] if obj['visible'] and name in obj['objectType']]
        return objs[0] if len(objs) > 0 else None

    def parse_command(self, action_str):
        def get_triplet(astr, key):
            astr = astr.replace(key, "").split()
            obj, tar = astr[0], astr[1]
            return obj, "rel", tar

        action_str = str(action_str).strip()
        if "Navigation " in action_str:
            tar = action_str.replace("Navigation ", "")
            return {'action': self.Action.GOTO, 'tar': tar}
        elif "PickupObject " in action_str:
            obj, rel, tar = get_triplet(action_str, "PickupObject ")
            return {'action': self.Action.PICK, 'obj': obj, 'rel': rel, 'tar': tar}
        elif "PutObject " in action_str:
            obj, rel, tar = get_triplet(action_str, "PutObject ")
            return {'action': self.Action.PUT, 'obj': obj, 'rel': rel, 'tar': tar}
        elif "OpenObject " in action_str:
            tar = action_str.replace("OpenObject ", "")
            return {'action': self.Action.OPEN, 'tar': tar}
        elif "CloseObject " in action_str:
            tar = action_str.replace("CloseObject ", "")
            return {'action': self.Action.CLOSE, 'tar': tar}
        elif "ToggleObjectOn " in action_str:
            tar = action_str.replace("ToggleObjectOn ", "")
            return {'action': self.Action.TOGGLE, 'tar': tar}
        elif "ToggleObjectOff " in action_str:
            tar = action_str.replace("ToggleObjectOff ", "")
            return {'action': self.Action.TOGGLE, 'tar': tar}
        elif "SliceObject " in action_str:
            obj, rel, tar = get_triplet(action_str, "SliceObject ")
            return {'action': self.Action.SLICE, 'obj': obj, 'rel': rel, 'tar': tar}
            # Below is not used by LLM-Planner
        elif "HeatObject " in action_str:
            obj, rel, tar = get_triplet(action_str, "HeatObject ")
            return {'action': self.Action.HEAT, 'obj': obj, 'rel': rel, 'tar': tar}
        elif "CoolObject " in action_str:
            obj, rel, tar = get_triplet(action_str, "CoolObject ")
            return {'action': self.Action.COOL, 'obj': obj, 'rel': rel, 'tar': tar}
        elif "CleanObject " in action_str:
            obj, rel, tar = get_triplet(action_str, "CleanObject ")
            return {'action': self.Action.CLEAN, 'obj': obj, 'rel': rel, 'tar': tar}
        elif "Inventory" in action_str:
            return {'action': self.Action.INVENTORY}
        elif "ExamineObject " in action_str:
            tar = action_str.replace("ExamineObject ", "")
            return {'action': self.Action.EXAMINE, 'tar': tar}
        elif "LookObject" in action_str:
            return {'action': self.Action.LOOK}
        else:
            return {'action': self.Action.PASS}

    def perspective_camera_view(self, reset_view_angle=0.0):
        # 默认视角为0.0
        normal_view_angle = reset_view_angle
        # 先获取当前视角的角度
        cameraHorizon = self.last_event.metadata["agent"]["cameraHorizon"]
        if int(cameraHorizon) - normal_view_angle < 0:
            # 需要进行look_down
            event = self.step(action="LookDown", degrees=normal_view_angle - int(cameraHorizon), forceAction=True)
            # 加入action_info
            action_info = dict(action="LookDown", degrees=normal_view_angle - int(cameraHorizon), forceAction=True)
            self.exec_action_list.append(action_info)
        elif int(cameraHorizon) - normal_view_angle > 0:
            # 需要进行look_up
            event = self.step(action="LookUp", degrees=int(cameraHorizon) - normal_view_angle, forceAction=True)
            action_info = dict(action="LookUp", degrees=int(cameraHorizon) - normal_view_angle, forceAction=True)
            self.exec_action_list.append(action_info)
        else:
            normal_view_angle = reset_view_angle

    def perspective_robot_angle(self, reset_view_angle):
        current_angle = self.last_event.metadata["agent"]["rotation"]["y"]
        for _ in range(4):
            if abs(current_angle - reset_view_angle) < 2:
                break
            else:
                event = self.step(action="RotateRight",  forceAction=True)
                self.exec_action_list.append(dict(action="RotateRight", degrees=90, forceAction=True))

    def teleport_nav_new(self, position_tuple):
        is_arrive = False
        target_position = {"x": position_tuple[0], "y": position_tuple[1], "z": position_tuple[2]}
        for i in range(4):
            event = self.step(action="Teleport", position=target_position)
            if self.last_event.metadata["lastActionSuccess"]:
                is_arrive = True
                break
            else:
                event = self.step(action="RotateRight", forceAction=True)
        return is_arrive

    def check_target_is_recept(self, target_id):
        # recept_object_list = [r['receptacleObjectIds']]
        recept_object_list = []
        for object_dict in self.last_event.metadata["objects"]:
            if object_dict["receptacle"] and object_dict["openable"] and not object_dict["isOpen"]:
                recept_object_one = object_dict["receptacleObjectIds"]
                if recept_object_one is not None:
                    recept_object_list.extend(recept_object_one)
        if target_id in recept_object_list:
            return True
        else:
            return False

    def get_pose_change(self):
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do

    def get_sim_location(self):
        y = -self.last_event.metadata['agent']['position']['x']
        x = self.last_event.metadata['agent']['position']['z']
        o = np.deg2rad(-self.last_event.metadata['agent']['rotation']['y'])
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def check_for_down_surround(self, frontiers_label_dict_list, sem_map):
        need_surround = False
        # 获取机器人位置
        robot_position = self.last_event.metadata["agent"]["position"]
        robot_position_tuple = (robot_position["x"], robot_position["y"], robot_position["z"])
        # 解析边界list
        # frontiers_position_list = []
        for frontiers_label_dict_one in frontiers_label_dict_list:
            centroid = frontiers_label_dict_one["centroid"]
            # 计算nav_point world
            target_nav_position = sem_map.pixel2world_point(centroid)
            # distance = np.sqrt(
            #     (robot_position_tuple[0] - target_nav_position[0]) ** 2 + \
            #     (robot_position_tuple[-1] - target_nav_position[-1]) ** 2
            # )
            nav_action, path = self.parse_nav_action(target_nav_position)
            distance = (len(path) - 1) * 0.25
            if distance < self.frontiers_distance_threshold:
                need_surround = True
                break
        return need_surround

    def down_surround(self, sem_map):
        # 进行低头
        event = self.step(action="LookDown", degrees=30)
        # 开始环视
        for _ in range(4):
            # 增加转圈
            rgb, depth_frame, mask_list, info_dict = self.get_obs()
            info_dict["add_robot_mask"] = True
            info_dict["region_size"] = 0.20
            global_sem_map, global_sem_feature_map = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
            event = self.step(action="RotateRight", forceAction=True)
        # 调整视角
        self.perspective_camera_view()

    def init_depth_model(self, args):
        if args.use_learned_depth:
            self.depth_gpu = torch.device("cuda:" + str(args.depth_gpu) if args.cuda else "cpu")
            model_path = 'valts/model-2000-best_silog_10.13741'  # 45 degrees only model
            self.depth_pred_model = AlfredSegmentationAndDepthModel()
            state_dict = torch.load('models/depth/depth_models/' + model_path, map_location=self.depth_gpu)['model']

            new_checkpoint = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_checkpoint[name] = v

            state_dict = new_checkpoint
            del new_checkpoint

            self.depth_pred_model.load_state_dict(state_dict)
            self.depth_pred_model.eval()
            self.depth_pred_model.to(device=self.depth_gpu)

            model_path = 'valts0/model-102500-best_silog_17.00430'
            self.depth_pred_model_0 = AlfredSegmentationAndDepthModel()
            state_dict = torch.load('models/depth/depth_models/' + model_path, map_location=self.depth_gpu)['model']

            new_checkpoint = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_checkpoint[name] = v

            state_dict = new_checkpoint
            del new_checkpoint

            self.depth_pred_model_0.load_state_dict(state_dict)
            self.depth_pred_model_0.eval()
            self.depth_pred_model_0.to(device=self.depth_gpu)

            print("depth initialized")

    def _preprocess_obs(self, rgb, depth):
        sem_seg_pred = self.seg.get_sem_pred(rgb.astype(np.uint8))  # (300, 300, num_cat)
        if self.args.use_learned_depth:
            include_mask = np.sum(sem_seg_pred, axis=2).astype(bool).astype(float)
            include_mask = np.expand_dims(np.expand_dims(include_mask, 0), 0)
            include_mask = torch.tensor(include_mask).to(self.depth_gpu)

            depth = self.depth_pred_later(include_mask)

            depth = self._preprocess_depth(depth, self.args.min_depth, self.args.max_depth)
        return sem_seg_pred, depth

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1  # shape (h,w)

        if self.picked_up:
            mask_err_below = depth < 0.5
            if not (self.picked_up_mask is None):
                mask_picked_up = self.picked_up_mask == 1
                depth[mask_picked_up] = 100.0
        else:
            mask_err_below = (depth < 0.0)
        depth[mask_err_below] = 100.0
        # depth[depth > 3] = 0

        # 以m为单位
        # depth = depth * 100
        return depth

    def depth_pred_later(self, sem_seg_pred):
        rgb = cv2.cvtColor(self.last_event.frame.copy(), cv2.COLOR_RGB2BGR)  # shape (h, w, 3)
        rgb_image = torch.from_numpy(rgb).permute((2, 0, 1)).unsqueeze(0).half() / 255

        if abs(self.last_event.metadata["agent"]["cameraHorizon"] - 0) < 5:
            _, pred_depth = self.depth_pred_model_0.predict(rgb_image.to(device=self.depth_gpu).float())

        else:
            _, pred_depth = self.depth_pred_model.predict(rgb_image.to(device=self.depth_gpu).float())

        if abs(self.last_event.metadata["agent"]["cameraHorizon"] - 0) < 5:
            include_mask_prop = self.args.valts_trustworthy_obj_prop0
        else:
            include_mask_prop = self.args.valts_trustworthy_obj_prop
        depth_img = pred_depth.get_trustworthy_depth(max_conf_int_width_prop=self.args.valts_trustworthy_prop, include_mask=sem_seg_pred, include_mask_prop=include_mask_prop)  # default is 1.0
        depth_img = depth_img.squeeze().detach().cpu().numpy()
        self.learned_depth_frame = pred_depth.depth_pred.detach().cpu().numpy()
        self.learned_depth_frame = self.learned_depth_frame.reshape((50, 300, 300))
        self.learned_depth_frame = 5 * 1 / 50 * np.argmax(self.learned_depth_frame, axis=0)  # Now shape is (300,300)
        del pred_depth
        depth = depth_img

        depth = np.expand_dims(depth, 2)
        return depth

    def post_process_seg_result_alfred(self):
        # 读取seg的object name
        object_name_list = []
        segmented_dict = self.seg.segmented_dict
        segmented_dict_small = segmented_dict["small"]
        segmented_dict_large = segmented_dict["large"]
        pred_box_small = segmented_dict_small["boxes"].numpy()
        pred_class_small = segmented_dict_small["classes"].numpy()
        pred_mask_small = segmented_dict_small["masks"]
        pred_box_large = segmented_dict_large["boxes"].numpy()
        pred_class_large = segmented_dict_large["classes"].numpy()
        pred_mask_large = segmented_dict_large["masks"]
        small_object_name_list = []
        large_object_name_list = []
        for pred_class_small_one in pred_class_small:
            small_object_name_list.append(self.seg.small_idx2small_object[pred_class_small_one])
        for pred_class_large_one in pred_class_large:
            large_object_name_list.append(self.seg.large_idx2large_object[pred_class_large_one])
        object_name_list.extend(small_object_name_list)
        object_name_list.extend(large_object_name_list)
        object_name_list = list(set(object_name_list))
        self.frame_object_name_list = object_name_list
        self.seen_object_name_list_all.extend(object_name_list)
        self.seen_object_name_list_all = list(set(self.seen_object_name_list_all))

        # 整理为info_dict
        mask_list = []
        label_dict = {}
        for index, pred_class_small_one in enumerate(pred_class_small):
            bbox_one = pred_box_small[index]
            mask_one = pred_mask_small[index]
            class_name = self.seg.small_idx2small_object[pred_class_small_one]
            mask_list.append(mask_one[None, :, :])
            label_dict[len(mask_list) - 1] = dict(bbox=bbox_one, class_name=class_name)

        for index, pred_class_large_one in enumerate(pred_class_large):
            bbox_one = pred_box_large[index]
            mask_one = pred_mask_large[index]
            class_name = self.seg.large_idx2large_object[pred_class_large_one]
            mask_list.append(mask_one[None, :, :])
            label_dict[len(mask_list) - 1] = dict(bbox=bbox_one, class_name=class_name)

        assert len(mask_list) == len(label_dict.keys())
        if len(mask_list) > 0:
            mask_list = np.concatenate(mask_list, axis=0)
        return mask_list, label_dict

    def post_process_seg_result_procthor(self):
        segmented_dict = self.seg.segmented_dict
        pred_box = segmented_dict["boxes"]
        if not isinstance(pred_box, np.ndarray):
            pred_box = pred_box.numpy()
        pred_class = segmented_dict["classes"]
        if not isinstance(pred_class, np.ndarray):
            pred_class = pred_class.numpy()
        pred_mask = segmented_dict["masks"]
        object_name_list = []
        for pred_class_one in pred_class:
            object_name_list.append(self.seg.idx2object[pred_class_one])
        object_name_list = list(set(object_name_list))
        self.frame_object_name_list = object_name_list
        self.seen_object_name_list_all.extend(object_name_list)
        self.seen_object_name_list_all = list(set(self.seen_object_name_list_all))

        # 整理为info_dict
        mask_list = []
        label_dict = {}
        for index, pred_class_one in enumerate(pred_class):
            bbox_one = pred_box[index]
            mask_one = pred_mask[index]
            class_name = self.seg.idx2object[pred_class_one]
            mask_list.append(mask_one[None, :, :])
            label_dict[len(mask_list) - 1] = dict(bbox=bbox_one, class_name=class_name)

        assert len(mask_list) == len(label_dict.keys())
        if len(mask_list) > 0:
            mask_list = np.concatenate(mask_list, axis=0)
        return mask_list, label_dict

    def get_nearest_object_by_type(self, object_type):
        metadata = self.last_event.metadata
        objs_of_type = [obj for obj in metadata['objects'] if obj['objectType'] == object_type]
        robot_position = metadata["agent"]["position"]
        robot_position_tuple = (robot_position["x"], robot_position["y"], robot_position["z"])
        distance_list = []
        object_id_list = []
        for objs_type_one in objs_of_type:
            object_dict = next(obj for obj in self.last_event.metadata["objects"]
                                  if obj["objectId"] == objs_type_one["objectId"])
            object_position = object_dict["position"]
            object_position_tuple = (object_position["x"], object_position["y"], object_position["z"])
            distance_one = np.sqrt((object_position_tuple[0] - robot_position_tuple[0]) ** 2 + (object_position_tuple[-1] - object_position_tuple[-1]) ** 2)
            if distance_one < 1.45:
                distance_list.append(distance_one)
                object_id_list.append(object_dict["objectId"])
        if len(distance_list) > 0:
            nearest_index = np.argmin(np.asarray(distance_list))
            return object_id_list[nearest_index]
        else:
            return ""

    def set_third_view(self):
        robot_position_dict = self.last_event.metadata["agent"]["position"]
        robot_rotation_dict = self.last_event.metadata["agent"]["rotation"]
        # 根据y值进行处理
        robot_camera_y = robot_rotation_dict["y"]
        if abs(robot_camera_y - 270) < 5:
            third_camera_position_dict = dict(x=robot_position_dict["x"] + 1, y=2.3, z=robot_position_dict["z"] - 0.5)
            third_camera_rotation_dict = dict(x=45, y=robot_rotation_dict["y"], z=0)
        elif abs(robot_camera_y - 180) < 5:
            third_camera_position_dict = dict(x=robot_position_dict["x"] + 0.5, y=2.3, z=robot_position_dict["z"] + 1)
            third_camera_rotation_dict = dict(x=45, y=robot_rotation_dict["y"], z=0)
        elif abs(robot_camera_y - 90) < 5:
            third_camera_position_dict = dict(x=robot_position_dict["x"] - 1, y=2.3, z=robot_position_dict["z"] + 0.5)
            third_camera_rotation_dict = dict(x=45, y=robot_rotation_dict["y"], z=0)
        else:
            third_camera_position_dict = dict(x=robot_position_dict["x"] - 0.5, y=2.3, z=robot_position_dict["z"] - 1)
            third_camera_rotation_dict = dict(x=45, y=robot_rotation_dict["y"], z=0)

        # print(third_camera_position_dict)
        # print(third_camera_rotation_dict)
        self.step(
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=0,
            position=third_camera_position_dict,
            rotation=third_camera_rotation_dict,
            fieldOfView=90
        )

    def add_third_camera(self):
        robot_position_dict = self.last_event.metadata["agent"]["position"]
        robot_rotation_dict = self.last_event.metadata["agent"]["rotation"]
        # 根据y值进行处理
        robot_camera_y = robot_rotation_dict["y"]
        if abs(robot_camera_y - 270) < 5:
            third_camera_position_dict = dict(x=robot_position_dict["x"] + 1, y=2.3, z=robot_position_dict["z"] - 0.5)
            third_camera_rotation_dict = dict(x=45, y=robot_rotation_dict["y"], z=0)
            is_visibility = self.check_third_view_visibility(third_camera_position_dict)
            if not is_visibility:
                third_camera_position_dict = dict(x=robot_position_dict["x"] + 1, y=2.3, z=robot_position_dict[
                                                                                               "z"] + 0.5)
        elif abs(robot_camera_y - 180) < 5:
            third_camera_position_dict = dict(x=robot_position_dict["x"] + 0.5, y=2.3, z=robot_position_dict["z"] + 1)
            third_camera_rotation_dict = dict(x=45, y=robot_rotation_dict["y"], z=0)
            is_visibility = self.check_third_view_visibility(third_camera_position_dict)
            if not is_visibility:
                third_camera_position_dict = dict(x=robot_position_dict["x"] - 0.5, y=2.3, z=robot_position_dict[
                                                                                               "z"] + 1)
        elif abs(robot_camera_y - 90) < 5:
            third_camera_position_dict = dict(x=robot_position_dict["x"] - 1, y=2.3, z=robot_position_dict["z"] + 0.5)
            third_camera_rotation_dict = dict(x=45, y=robot_rotation_dict["y"], z=0)
            is_visibility = self.check_third_view_visibility(third_camera_position_dict)
            if not is_visibility:
                third_camera_position_dict = dict(x=robot_position_dict["x"] - 1, y=2.3, z=robot_position_dict[
                                                                                               "z"] - 0.5)
        else:
            third_camera_position_dict = dict(x=robot_position_dict["x"] - 0.5, y=2.3, z=robot_position_dict["z"] - 1)
            third_camera_rotation_dict = dict(x=45, y=robot_rotation_dict["y"], z=0)
            is_visibility = self.check_third_view_visibility(third_camera_position_dict)
            if not is_visibility:
                third_camera_position_dict = dict(x=robot_position_dict["x"] + 0.5, y=2.3, z=robot_position_dict[
                                                                                               "z"] - 1)

        event = self.step(
            action="AddThirdPartyCamera",
            position=third_camera_position_dict,
            rotation=third_camera_rotation_dict,
            fieldOfView=90
        )

    def check_third_view_visibility(self, third_camera_position_dict):
        # 获取场景边界
        metadata = self.last_event.metadata
        sceneBounds = metadata["sceneBounds"]
        scene_size = sceneBounds["size"]
        x_min = 0
        z_min = 0
        x_max = scene_size["x"]
        z_max = scene_size["z"]
        camera_position_x = third_camera_position_dict["x"]
        camera_position_z = third_camera_position_dict["z"]
        if x_max > camera_position_x > x_min and z_max > camera_position_z > z_min:
            return True
        else:
            return False

    def check_complex_task(self, task_type=None, task_label_dict=None):
        success = False
        if task_type is None:
            return success
        if task_type == "get_a_drink_and_put_in_recep":
            ts = 3
            s = 0
            targets = self.get_targets(task_label_dict)

            # common beverages list
            common_beverages = ["winebottle", "water", "coffee", "cup", "mug", "bottle", "bowl"]
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')

            objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                             if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
            objs_drinkable = [p['objectId'] for p in pickupables for d in common_beverages
                              if d in p['objectId'].lower()]

            # check if object is in the receptacle
            if len(objs_in_place) > 0:
                s += 1

            # check if some object is drinkable
            if len(objs_drinkable) > 0:
                s += 1

            if np.any([obj_id in objs_drinkable for obj_id in objs_in_place]):
                s += 1

        elif task_type == "pick_and_place_hard":
            ts = 1
            s = 0
            targets = self.get_targets(task_label_dict)
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')

            if np.any([np.any([p['objectId'] in r['receptacleObjectIds']
                               for r in receptacles if r['receptacleObjectIds'] is not None])
                       for p in pickupables]):
                s += 1

        elif task_type == "pick_slice_heat_then_place_in_recep":
            ts = 4
            s = 0

            targets = self.get_targets(task_label_dict)
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')

            objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                             if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
            objs_heated = [p['objectId'] for p in pickupables if p['objectId'] in self.heated_objects]
            objs_sliced = []

            # slice egg的属性是isBroken
            for t in pickupables:
                if np.any(t['isSliced']) or np.any(t['isBroken']):
                    objs_sliced.append(t['objectId'])

            # check if object is in the receptacle
            if len(objs_in_place) > 0:
                s += 1
            # check if some object was sliced
            if len(objs_sliced) > 0:
                s += 1

            # check if some object was heated
            if len(objs_heated) > 0:
                s += 1
            objs_in_place = [obj_id.rsplit("|", maxsplit=1)[0] for obj_id in objs_in_place]

            objs_s_h_in_place = [p['objectId'] for p in pickupables if
                                 p['objectId'] in objs_heated and p["objectId"] in objs_in_place and p[
                                     'objectId'] in objs_sliced]

            if len(objs_s_h_in_place) > 0:
                s += 1

        elif task_type == "box_obj_in_recep":
            ts = 1
            s = 0
            targets = self.get_targets(task_label_dict)
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')

            if np.any([np.any([p['objectId'] in r['receptacleObjectIds']
                               for r in receptacles if r['receptacleObjectIds'] is not None])
                       for p in pickupables]):
                s += 1

        elif task_type == "pick_place_obj_in_recep_then_slice":
            ts = 3
            s = 0

            targets = self.get_targets(task_label_dict)
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')

            objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                             if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
            objs_sliced = []
            for t in pickupables:
                if np.any(t['isSliced']) or np.any(t['isBroken']):
                    objs_sliced.append(t)

            # check if object is in the receptacle
            if len(objs_in_place) > 0:
                s += 1
            # check if some object was sliced
            if len(objs_sliced) > 0:
                s += 1
            objs_sliced_id = [obj['objectId'] for obj in objs_sliced]

            if np.any([obj_id.rsplit('|', maxsplit=1)[0] in objs_sliced_id for obj_id in objs_in_place]):
                s += 1

        elif task_type == 'pick_place_then_toggle_obj':
            ts = 2
            s = 0
            targets = self.get_targets(task_label_dict)
            toggleables = self.get_objects_with_name_and_prop(targets['toggle'], 'toggleable')
            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')

            object_common_beverages = ["Apple", "Bread", "Laptop", "Book"]
            for object_common_one in object_common_beverages:
                pickupables_add = self.get_objects_with_name_and_prop(object_common_one, "pickupable")
                pickupables.extend(pickupables_add)

            if np.any([np.any([p['objectId'] in r['receptacleObjectIds']
                               for r in receptacles if r['receptacleObjectIds'] is not None])
                       for p in pickupables]):
                s += 1
            if np.any([t['isToggled'] for t in toggleables]):
                s += 1

        elif task_type == 'pick_clean_heat_then_place_in_recep':
            ts = 4
            s = 0

            targets = self.get_targets(task_label_dict)
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')

            object_common_beverages = ["Spoon", "Fork"]
            for object_common_one in object_common_beverages:
                pickupables_add = self.get_objects_with_name_and_prop(object_common_one, "pickupable")
                pickupables.extend(pickupables_add)

            objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                             if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
            objs_cleaned = [p['objectId'] for p in pickupables if p['objectId'] in self.cleaned_objects]
            objs_heated = [p['objectId'] for p in pickupables if p['objectId'] in self.heated_objects]

            # check if object is in the receptacle
            if len(objs_in_place) > 0:
                # print(f"there is sth in place")
                # print(objs_in_place)
                s += 1
            # check if some object was cleaned
            if len(objs_cleaned) > 0:
                s += 1
            # check if some object was heated
            if len(objs_heated) > 0:
                s += 1
            # check if the object is both in the receptacle and clean
            if np.any([(obj_id in objs_cleaned) and (obj_id in objs_heated) for obj_id in objs_in_place]):
                # print("sth cleaned and heated is in place")
                s += 1

        elif task_type == 'toggle_obj_then_pick_place_obj':
            ts = 2
            s = 0

            targets = self.get_targets(task_label_dict)
            toggleables = self.get_objects_with_name_and_prop(targets['toggle'], 'toggleable')
            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')

            object_common_beverages = ["Apple", "Bread", "Laptop", "Book"]
            for object_common_one in object_common_beverages:
                pickupables_add = self.get_objects_with_name_and_prop(object_common_one, "pickupable")
                pickupables.extend(pickupables_add)

            if np.any([np.any([p['objectId'] in r['receptacleObjectIds']
                               for r in receptacles if r['receptacleObjectIds'] is not None])
                       for p in pickupables]):
                s += 1
            if np.any([t['isToggled'] for t in toggleables]):
                s += 1

        elif task_type == 'open_recep_then_pick_obj':
            ts = 1
            s = 0

            targets = self.get_targets(task_label_dict)
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')

            object_common_beverages = ["Egg", "Tomato", "Potato", "Lettuce"]
            for object_common_one in object_common_beverages:
                pickupables_add = self.get_objects_with_name_and_prop(object_common_one, "pickupable")
                pickupables.extend(pickupables_add)

            if np.any([np.any([p['objectId'] in r['receptacleObjectIds']
                               for r in receptacles if r['receptacleObjectIds'] is not None])
                       for p in pickupables]):
                s += 1

        elif task_type == 'pick_clean_obj_in_light':
            ts = 3
            s = 0
            targets = self.get_targets(task_label_dict)
            toggleables = self.get_objects_with_name_and_prop(targets['toggle'], 'toggleable')
            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')

            object_common_beverages = ["Spoon", "Fork"]
            for object_common_one in object_common_beverages:
                pickupables_add = self.get_objects_with_name_and_prop(object_common_one, "pickupable")
                pickupables.extend(pickupables_add)

            objs_cleaned = [p['objectId'] for p in pickupables if p['objectId'] in self.cleaned_objects]
            objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                             if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]

            # check if the lamp is visible and turned on
            if np.any([t['isToggled'] for t in toggleables]):
                s += 1
            # check if sth is placed
            if len(objs_in_place) > 0:
                s += 1
            # check if some object was cleaned
            if len(objs_cleaned) > 0:
                s += 1

        elif task_type == "pick_wash_obj_then_place_in_recep_with_movable_recep":
            ts = 2
            s = 0
            targets = self.get_targets(task_label_dict)

            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
            movables = self.get_objects_with_name_and_prop(targets['mrecep'], 'pickupable')
            toggleables = self.get_objects_with_name_and_prop(targets['toggle'], 'toggleable')
            mrecep_common_beverages = ["Bowl", "Plate"]
            object_common_beverages = ["Spoon", "Fork"]
            # 增加新的候选
            for object_common_one in object_common_beverages:
                pickupables_add = self.get_objects_with_name_and_prop(object_common_one, "pickupable")
                pickupables.extend(pickupables_add)
            for mrecep_common_one in mrecep_common_beverages:
                movables_add = self.get_objects_with_name_and_prop(mrecep_common_one, "pickupable")
                movables.extend(movables_add)

            objs_cleaned = [p['objectId'] for p in pickupables if p['objectId'] in self.cleaned_objects]
            objs_cleaned.extend([m["objectId"] for m in movables if m["objectId"] in self.cleaned_objects])

            pickup_in_mrecep = [p['objectId'] for p in pickupables for m in movables
                                if 'receptacleObjectIds' in p and m['receptacleObjectIds'] is not None
                                and p['objectId'] in m['receptacleObjectIds']]

            # check if sth in mrecep
            if len(pickup_in_mrecep) > 0:
                # print("sth is in mrecep")
                # print(pickup_in_mrecep)
                s += 1

            # check if sth cleaned
            if len(objs_cleaned) > 0:
                # print("sth is cleaned")
                # print(objs_cleaned)
                s += 1

        elif task_type == "pick_clean_slice_obj_and_place":
            ts = 4
            s = 0
            targets = self.get_targets(task_label_dict)

            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
            toggleables = self.get_objects_with_name_and_prop(targets['toggle'], 'toggleable')
            objs_cleaned = [p['objectId'] for p in pickupables if p['objectId'] in self.cleaned_objects]

            objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                             if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
            objs_in_place = [obj_id.rsplit("|", maxsplit=1)[0] for obj_id in objs_in_place]

            objs_sliced = []
            for t in pickupables:
                if np.any(t['isSliced']) or np.any(t['isBroken']):
                    objs_sliced.append(t)
            objs_sliced_id = [obj['objectId'] for obj in objs_sliced]

            objs_clean_in_place = [p['objectId'] for p in pickupables if
                                   p['objectId'] in objs_cleaned and p["objectId"] in objs_in_place]

            if len(objs_cleaned) > 0:
                s += 1

            if len(objs_sliced) > 0:
                s += 1

            if len(objs_in_place) > 0:
                s += 1

            if len(objs_clean_in_place) > 0:
                s += 1

        elif task_type == "pick_obj_then_open_recep_and_place":
            ts = 1
            s = 0
            targets = self.get_targets(task_label_dict)

            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
            object_common_beverages = ["Spoon"]
            # 增加新的候选
            for object_common_one in object_common_beverages:
                pickupables_add = self.get_objects_with_name_and_prop(object_common_one, "pickupable")
                pickupables.extend(pickupables_add)

            if np.any([np.any([p['objectId'] in r['receptacleObjectIds']
                               for r in receptacles if r['receptacleObjectIds'] is not None])
                       for p in pickupables]):
                s += 1

        elif task_type == "pick_clean_and_cool_obj":
            ts = 3
            s = 0
            targets = self.get_targets(task_label_dict)

            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
            objs_cooled = [p['objectId'] for p in pickupables if p['objectId'] in self.cooled_objects]
            objs_cleaned = [p['objectId'] for p in pickupables if p['objectId'] in self.cleaned_objects]

            cool_and_clean_object = [p["objectId"] for p in pickupables if p["objectId"] in objs_cleaned and \
                                     p["objectId"] in objs_cooled]

            if len(objs_cleaned) > 0:
                s += 1
            if len(objs_cooled) > 0:
                s += 1
            if len(cool_and_clean_object) > 0:
                s += 1

        elif task_type == "pick_clean_heat_obj_with_movable_recep":
            ts = 3
            s = 0
            targets = self.get_targets(task_label_dict)
            pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
            receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
            movables = self.get_objects_with_name_and_prop(targets['mrecep'], 'receptacle')
            mrecep_common_beverages = ["Plate", "Bowl"]
            object_common_beverages = ["Egg", "Tomato", "Potato", "Lettuce"]
            # 增加新的候选
            for object_common_one in object_common_beverages:
                pickupables_add = self.get_objects_with_name_and_prop(object_common_one, "pickupable")
                pickupables.extend(pickupables_add)
            for mrecep_common_one in mrecep_common_beverages:
                movables_add = self.get_objects_with_name_and_prop(mrecep_common_one, "pickupable")
                movables.extend(movables_add)

            objs_heated = [p['objectId'] for p in pickupables if p['objectId'] in self.heated_objects]
            objs_cleaned = [p['objectId'] for p in pickupables if p['objectId'] in self.cleaned_objects]

            pickup_in_mrecep = [p['objectId'] for p in pickupables for m in movables
                                if 'receptacleObjectIds' in p and m['receptacleObjectIds'] is not None
                                and p['objectId'] in m['receptacleObjectIds']]

            clean_and_heated_object = [p['objectId'] for p in pickupables if
                                       p['objectId'] in objs_cleaned and p["objectId"] in objs_heated]

            # check if the object is in the final receptacle
            if len(pickup_in_mrecep) > 0:
                # print("pickup_in_mrecep")
                # print(pickup_in_mrecep)
                s += 1
            # check if the object is cleaned
            if len(objs_cleaned) > 0:
                # print("sth cleaned")
                # print(objs_cleaned)
                s += 1
            # check if the object is heated
            if len(objs_heated) > 0:
                # print("sth heated")
                # print(objs_heated)
                s += 1
            # check if the obj heated and cleaned is in mrecep

        else:
            ts = 0
            s = 0
            print('No matching task types exist!')
            print('No matching task types exist!')
            print('No matching task types exist!')
            response_dict = {"success": success,
                             "ts": 0,
                             "s": 0}

        if ts == s:
            success = True

        response_dict = {"success": success,
                         "ts": ts,
                         "s": s}

        return response_dict



































