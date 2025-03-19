import numpy as np
from models.Detic.segmentation_helper_procthor_detic import SemgnetationHelperProcThorDetic
import random
from typing import Tuple
from collections import deque
from typing import Optional, Sequence, cast

from tqdm import tqdm
import open3d as o3d

import torch
import json
import math
from PIL import Image
from utils.keypoint_utils import KeypointProposer

from kmeans_pytorch import kmeans
from sklearn.cluster import MeanShift

# 配置环境参数
AGENT_STEP_SIZE = 0.25
RECORD_SMOOTHING_FACTOR = 1
CAMERA_HEIGHT_OFFSET = 0.75
VISIBILITY_DISTANCE = 25


class KeypointHelper(object):
    def __init__(self, args):
        self.args = args
        # load Detic
        self.seg = SemgnetationHelperProcThorDetic(self)

        # object_dict
        self.total_cat2idx = json.load(open(args.total_cat2idx_procthor_path))
        self.total_idx2cat = {}
        for cat, index in self.total_cat2idx.items():
            self.total_idx2cat[index] = cat

        # load keypoint
        config = {
            'device': "cuda:0",
            'bounds_min': [0.0, 0.0, 0.0],
            'bounds_max': [30.0, 30.0, 30.0],
            'min_dist_bt_keypoints': 0.05,
            'max_mask_ratio': 0.5,
            'num_candidates_per_mask': 5,
            'seed': 42
        }

        self.config = config
        self.keypoint_proposer = KeypointProposer(config)


    def get_keypoint(self, obs_dict):
        rgb = obs_dict["rgb_image"]
        masks = obs_dict["masks"]
        depth_image = obs_dict["depth_image"]
        info = obs_dict["info"]

        points = self.get_point_cloud(depth_image, None, info, point_only=True)
        points = points.reshape([rgb.shape[0], rgb.shape[1], -1])
        candidate_keypoints, projected_image, candidate_pixels = self.keypoint_proposer.get_keypoints(
            rgb,
            points,
            masks
        )
        # 生成具体mask
        keypoint_mask = np.zeros_like(masks)
        for keypoint_count, pixel in enumerate(candidate_pixels):
            # draw a box
            box_width = 30
            box_height = 30
            bbox_min = [pixel[1] - box_width // 2, pixel[0] - box_height // 2]
            bbox_max = [pixel[1] + box_width // 2, pixel[0] + box_height // 2]
            keypoint_mask[bbox_min[1]:bbox_max[1], bbox_min[0]:bbox_max[0]] = 1

        # Lift to 3D
        affordance_score = keypoint_mask.reshape([-1, 1])
        points = points.reshape([-1, 3])
        # print(affordance_score.shape)
        # print(points.shape)
        world_space_point_cloud = np.concatenate((points, affordance_score), axis=1)

        mask = np.ones(depth_image.shape, dtype=bool)
        # rgb_color = rgb_image[:, :, ::-1]
        rgb_image = rgb[:, :, ::-1]
        r_color = rgb_image[:, :, 0][mask][None, :]
        g_color = rgb_image[:, :, 1][mask][None, :]
        b_color = rgb_image[:, :, 2][mask][None, :]
        rgb_color = np.concatenate((r_color, g_color, b_color), axis=0).T

        world_space_point_cloud = np.concatenate((world_space_point_cloud, rgb_color), axis=1)
        keypoint_output_dict = {"keypoint_map": world_space_point_cloud,
                                  "projected_image": projected_image}

        return keypoint_output_dict


    def cpu_only_depth_frame_to_camera_space_xyz(self,
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

        return camera_space_xyz * depth_frame[mask][None, :].astype(np.float16)

    def cpu_only_camera_space_xyz_to_world_xyz(self,
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
                [1, 0, 0],  # unchanged
                [0, cos_psi, sin_psi],
                [0, -sin_psi, cos_psi, ],
            ],
            np.float16,
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
                [0, 1, 0],  # unchanged
                [sin_phi, 0, cos_phi], ],
            np.float16,
        )
        # fmt: on

        # Apply the above transformations
        view_points = (rotation_transform @ horizon_transform) @ camera_space_xyzs

        # Translate the points w.r.t. the camera's position in world space.
        world_points = view_points + camera_world_xyz[:, None]
        return world_points

    def get_point_cloud(self, depth_one, affordance_score, info_dict_one, point_only=False):
        fov = info_dict_one["fov"]
        cameraHorizon = info_dict_one["cameraHorizon"]
        camera_world_xyz = info_dict_one["camera_world_xyz"]
        if not isinstance(camera_world_xyz, np.ndarray):
            camera_world_xyz = np.asarray(camera_world_xyz)
        rotation = info_dict_one["rotation"]

        # 映射点云
        camera_space_point_cloud = self.cpu_only_depth_frame_to_camera_space_xyz(depth_one, mask=None, fov=fov)
        partial_point_cloud = self.cpu_only_camera_space_xyz_to_world_xyz(camera_space_point_cloud,
            camera_world_xyz, rotation, cameraHorizon)

        if not point_only:
            select_mask = np.ones(depth_one.shape, dtype=bool)
            affordance_score = affordance_score[select_mask][None, :]
            world_space_point_cloud = np.concatenate((partial_point_cloud, affordance_score), axis=0)
            world_space_point_cloud = world_space_point_cloud.T
            world_space_point_cloud[:, [0, 1, 2, 3]] = world_space_point_cloud[:, [0, 2, 1, 3]]
            world_space_point_cloud = world_space_point_cloud.astype(np.float16)
        else:
            world_space_point_cloud = partial_point_cloud.T
            world_space_point_cloud[:, [0, 1, 2]] = world_space_point_cloud[:, [0, 2, 1]]
            world_space_point_cloud = world_space_point_cloud.astype(np.float16)

        return world_space_point_cloud
