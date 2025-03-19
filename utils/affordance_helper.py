import numpy as np
from models.Detic.segmentation_helper_procthor_detic import SemgnetationHelperProcThorDetic
import random
from typing import Tuple
from collections import deque
from typing import Optional, Sequence, cast

from tqdm import tqdm

from model import longclip
import open3d as o3d

from models.lisa.lisa_helper import LisaHelper
import torch
import json
import math
from PIL import Image

# 配置环境参数
AGENT_STEP_SIZE = 0.25
RECORD_SMOOTHING_FACTOR = 1
CAMERA_HEIGHT_OFFSET = 0.75
VISIBILITY_DISTANCE = 25


class AffordanceHelper(object):
    def __init__(self, args):
        self.args = args
        # load Detic
        self.seg = SemgnetationHelperProcThorDetic(self)
        # load LongCLIP
        self.long_clip_device = torch.device("cuda:" + str(args.long_clip_gpu) if args.cuda else "cpu")
        self.long_clip_model, self.long_clip_preprocess = longclip.load(self.args.long_clip_path, device=self.long_clip_device)
        self.long_clip_model.eval()
        # load Lisa
        self.lisa = LisaHelper(self.args)

        # object_dict
        self.total_cat2idx = json.load(open(args.total_cat2idx_procthor_path))
        self.total_idx2cat = {}
        for cat, index in self.total_cat2idx.items():
            self.total_idx2cat[index] = cat

    def get_affordance(self, obs_dict):
        rgb_image = obs_dict["rgb"]
        depth_image = obs_dict["depth"]
        info = obs_dict["info"]
        target = obs_dict["target"]
        action = obs_dict["action"]

        confidence = self.get_obj_conf(rgb_image, target)
        detic_mask_image = None
        if self.args.detic_mask:
            detic_mask_image = self.seg.visualize_sem_nav(rgb_image)

        # longclip 计算相关性
        input_dict = {"rgb_image": rgb_image,
                      "text_prompt": obs_dict["text_prompt"]}
        similarity_score = self.get_similarity_score(input_dict)

        # lisa mask
        # lisa_mask = np.zeros([10, 10])
        lisa_output_dict = self.lisa.get_lisa_mask(rgb_image, action, target)
        lisa_mask = lisa_output_dict["output_mask"]
        lisa_mask_image = lisa_output_dict["lisa_mask_image"]
        affordance_score = similarity_score * confidence
        # affordance_score = similarity_score
        affordance_mask = lisa_mask * affordance_score

        # Lift to 3D
        cost_point_cloud = self.get_point_cloud(depth_image, affordance_mask, info)
        mask = np.ones(depth_image.shape, dtype=bool)
        # rgb_color = rgb_image[:, :, ::-1]
        rgb_image = rgb_image[:, :, ::-1]
        r_color = rgb_image[:, :, 0][mask][None, :]
        g_color = rgb_image[:, :, 1][mask][None, :]
        b_color = rgb_image[:, :, 2][mask][None, :]
        rgb_color = np.concatenate((r_color, g_color, b_color), axis=0).T

        world_space_point_cloud = np.concatenate((cost_point_cloud, rgb_color), axis=1)
        affordance_output_dict = {"affordance_map": world_space_point_cloud,
                                  "detic_mask_image": detic_mask_image,
                                  "lisa_mask_image": lisa_mask_image}

        return affordance_output_dict

    def get_obj_conf(self, rgb, target):
        # Detic处理得到mask
        pred_dict = self.seg.get_pred_dict(rgb)
        pred_class = pred_dict["classes"]
        mask = pred_dict["masks"]
        scores = pred_dict["scores"]
        target_index = self.total_cat2idx[target] - 1
        target_score = scores[pred_class == target_index]
        if target_score.shape[0] <= 0:
            confidence = 0
        else:
            confidence = np.max(target_score)
        return confidence

    def get_similarity_score(self, input_dict):
        rgb_image = input_dict["rgb_image"]
        text_prompt = input_dict["text_prompt"]
        # 增加候选
        description_list = text_prompt.split(" captures")
        comparative_description = "The image does not capture" + description_list[-1]
        # comparative_description_2 = "There is an empty house."
        comparative_description_3 = "The image captures nothing."
        text = longclip.tokenize([text_prompt, comparative_description,
                                  comparative_description_3]).to(self.long_clip_device)
        image = self.long_clip_preprocess(Image.fromarray(rgb_image)).unsqueeze(0).to(self.long_clip_device)
        with torch.no_grad():
            # image_features = long_clip_model.encode_image(image)
            # text_features = long_clip_model.encode_text(text)
            logits_per_image, logits_per_text = self.long_clip_model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        score = probs[0][0]
        return score

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

    def get_point_cloud(self, depth_one, affordance_score, info_dict_one):
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

        select_mask = np.ones(depth_one.shape, dtype=bool)
        affordance_score = affordance_score[select_mask][None, :]
        world_space_point_cloud = np.concatenate((partial_point_cloud, affordance_score), axis=0)
        world_space_point_cloud = world_space_point_cloud.T
        # [x, y, z] --> [x, z, y]
        world_space_point_cloud[:, [0, 1, 2, 3]] = world_space_point_cloud[:, [0, 2, 1, 3]]
        world_space_point_cloud = world_space_point_cloud.astype(np.float16)

        return world_space_point_cloud


