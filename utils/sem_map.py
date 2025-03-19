import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Optional, Sequence, cast
import math
import cv2
from skimage import measure
from einops import asnumpy, repeat
import torch.nn.functional as F
import utils.depth_utils as du
import time
import MinkowskiEngine as ME
from types import SimpleNamespace
from models.depth.alfred_perception_models import AlfredSegmentationAndDepthModel
from collections import Counter, OrderedDict
from model import longclip
import open3d as o3d
from models.point_clip.point_clip_helper import PointClipHelper


class Semantic_Mapping(nn.Module):
    def __init__(self, args, clip_model, clip_preprocess, total_cat2idx, llm_model=None, tokenizer=None, only_sem=False):
        super(Semantic_Mapping, self).__init__()
        self.args = args
        if args.alfred_scene:
            self.num_sem_categories = args.num_sem_categories_alfred
            self.map_size = args.map_size_m_alfred
            self.env_frame_width = args.env_frame_width_alfred
            self.env_frame_height = args.env_frame_height_alfred
            self.min_x = args.min_x_alfred
            self.min_z = args.min_z_alfred
            self.max_x = args.max_x_alfred
            self.max_z = args.max_z_alfred
        else:
            self.num_sem_categories = args.num_sem_categories
            self.map_size = args.map_size_m
            self.env_frame_width = args.env_frame_width
            self.env_frame_height = args.env_frame_height
            self.min_x = args.min_x
            self.min_z = args.min_z
            self.max_x = args.max_x
            self.max_z = args.max_z
        self.map_resolution = args.map_resolution
        self.clip_model = clip_model
        self.preprocess = clip_preprocess
        # self.device = self.clip_model.device
        if clip_model is not None:
            self.open_clip_device = next(clip_model.parameters()).device
        self.heightmap_size = np.ceil(((self.map_size - 0) / self.map_resolution,
                                  (self.map_size - 0) / self.map_resolution)).astype(int)
        self.global_sem_map = np.zeros([self.num_sem_categories, self.heightmap_size[0], self.heightmap_size[1]])
        self.feature_map_device = torch.device("cuda:" + str(args.sem_feature_map_gpu) if args.cuda else "cpu")
        if self.args.sem_feature_map_cuda:
            self.global_sem_feature_map = torch.zeros([self.heightmap_size[0], self.heightmap_size[1], 1024], device=self.feature_map_device, dtype=torch.float16)
        else:
            self.global_sem_feature_map = np.zeros([self.heightmap_size[0], self.heightmap_size[1], 1024], dtype=np.float16)
        self.vis_frontiers_mask = np.zeros([1, self.heightmap_size[0], self.heightmap_size[1]])
        self.total_cat2idx = total_cat2idx
        self.total_idx2cat = {}
        for k, v in self.total_cat2idx.items():
            self.total_idx2cat[v] = k
        self.max_y = 2.3
        self.min_y = 0
        self.floor_id = 0
        self.object_boundary = 1
        self.ME_device = torch.device('cuda')
        if args.alfred_scene:
            self.frontiers_thresholds = args.frontiers_thresholds_alfred
        else:
            self.frontiers_thresholds = args.frontiers_thresholds
        self.robot_heightmap_point = None
        self.height_map = None

        # new add
        self.llm_model = llm_model
        if self.llm_model is not None:
            self.llm_device = next(self.llm_model.parameters()).device
        self.tokenizer = tokenizer
        self.frontiers_label_dict_list = None
        self.only_sem = only_sem
        if self.args.vis_similarity:
            self.similarity_map = np.zeros([self.heightmap_size[0], self.heightmap_size[1]])
        else:
            self.similarity_map = None
        self.similarity_weights_list = []
        self.similarity_score_list = []

        # long clip
        if self.args.fusion_weights:
            self.long_clip_device = torch.device("cuda:" + str(args.long_clip_gpu) if args.cuda else "cpu")
            self.long_clip_model, self.long_clip_preprocess = longclip.load(self.args.long_clip_path, device=self.long_clip_device)
            self.long_clip_model.eval()

        # FILM depth model
        # self.init_depth_model(args)

        # 增加点云映射参数
        self.vision_range = args.vision_range
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.view_angles = [0.0]
        self.agent_height = args.camera_height
        self.shift_loc = [self.vision_range * self.map_resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(self.env_frame_width, self.env_frame_height, self.fov)

        # add object instance point
        self.object_instance_dict = {}
        self.point_clip = PointClipHelper(procthor_config=self.args)

    def reset(self):
        self.global_sem_map = np.zeros([self.num_sem_categories, self.heightmap_size[0], self.heightmap_size[1]])
        if self.args.sem_feature_map_cuda:
            self.global_sem_feature_map = torch.zeros([self.heightmap_size[0], self.heightmap_size[1],
                                                       1024], device=self.feature_map_device, dtype=torch.float16)
        else:
            self.global_sem_feature_map = np.zeros([self.heightmap_size[0], self.heightmap_size[1],
                                                    1024], dtype=np.float16)
        self.similarity_weights_list = []
        self.similarity_score_list = []

        self.object_instance_dict = {}

    def forward(self, rgb_image, depth_image, masks, info_dict_one, task="update", only_seg=False):
        if "update" in task:
            global_sem_map, global_sem_feature_map = self.update_sem_map(rgb_image, depth_image, masks, info_dict_one, only_seg=only_seg)
            return global_sem_map, global_sem_feature_map
        else:
            select_feature_list, frontiers_label_dict_list = self.nav_frontiers(info_dict_one)
            return select_feature_list, frontiers_label_dict_list

    def update_sem_map(self, rgb_image, depth_image, masks, info_dict_one, only_seg=False):
        # if not self.only_sem or len(masks) > 0:
        if not only_seg or len(masks) > 0:
            rgb_image_feature = self.extract_feature(rgb_image, masks)
        else:
            # rgb_image_feature = torch.zeros([depth_image.shape[0], depth_image.shape[1], 1024], dtype=torch.half)
            rgb_image_feature = torch.zeros([depth_image.shape[0], depth_image.shape[1], 1024], dtype=torch.half, device=self.open_clip_device)
        # 计算融合权重----------------------
        if not only_seg and self.args.fusion_weights and "text_prompt" in info_dict_one:
            # 计算相似度
            input_dict = {"rgb_image": rgb_image,
                          "text_prompt": info_dict_one["text_prompt"]}
            similarity_score = self.get_similarity_score(input_dict)
            # 更新权重
            self.similarity_score_list.append(similarity_score)
            # 计算相加权重
            norm_score_list = np.asarray(self.similarity_score_list) / np.sum(self.similarity_score_list)
            if len(self.similarity_weights_list) > 0:
                # weight_one = norm_score_list[0] / self.similarity_score_list[0]
                weight_one = 1 - norm_score_list[-1]
            else:
                weight_one = 1
            self.similarity_weights_list.append(weight_one)
            weight_two = norm_score_list[-1]
        # ----------------------------
        world_space_point_cloud = self.get_point_cloud(depth_image, masks, info_dict_one)
        rgb_image_feature = rgb_image_feature.reshape([-1, 1024])
        # if self.args.alfred_scene and self.args.use_sem_seg and not self.args.procthor_scene:
        if self.args.use_sem_seg:
            robot_position = info_dict_one["robot_position"]
            robot_position_mask = self.get_nav_position_mask([robot_position])
            robot_region_mask = self.get_robot_region(robot_position_mask, region_size=self.args.region_size_alfred)
            args_dict = {"only_seg": only_seg}
            semantic_map, semantic_feature_map, ob_mask, semantic_height_map = self.get_sem_map_alfred(world_space_point_cloud, rgb_image_feature, robot_region_mask, args_dict)
            # for debug
            # self.height_map = semantic_height_map
        else:
            semantic_map, semantic_feature_map, ob_mask = self.get_sem_map(world_space_point_cloud, rgb_image_feature)

        # 增加robot_position_mask
        robot_position = info_dict_one["robot_position"]
        robot_position_mask = self.get_nav_position_mask([robot_position])
        # 增加region限制
        if self.args.limit_vis_region:
            robot_region_mask = self.get_robot_region(robot_position_mask, region_size=self.args.limit_vis_region_size)
            semantic_map = semantic_map * robot_region_mask
        self.global_sem_map = np.maximum(self.global_sem_map, semantic_map)
        # if info_dict_one["interactive_object"] is not None:
        #     self.fine_grained_sem_map_update(semantic_map, info_dict_one)
        if info_dict_one["add_robot_mask"]:
            robot_region_mask = self.get_robot_region(robot_position_mask, region_size=info_dict_one["region_size"])
            self.global_sem_map[0, :, :] = np.maximum(self.global_sem_map[0, :, :], robot_region_mask)

        if self.args.vis_similarity and "text_prompt" in info_dict_one and not only_seg:
            # similarity_score = info_dict_one["similarity_score"]
            # self.similarity_score_list.append(similarity_score)
            similarity_map_one = self.get_vis_similarity_map(world_space_point_cloud, similarity_score)
            self.similarity_map = self.similarity_map * weight_one + similarity_map_one * weight_two

        # self.global_sem_map[:, process_ob_map] = semantic_map[:, process_ob_map]
        if not only_seg and not self.args.fusion_weights:
            global_sem_ob_map = self.get_ob_map(self.global_sem_map)
            local_ob_mask = self.get_ob_map(semantic_map)
            if np.sum(global_sem_ob_map) > 0:
                process_mask = self.mask_true_false(global_sem_ob_map, local_ob_mask)
                process_ob_map = self.set_false_in_C(process_mask, local_ob_mask)
            else:
                process_ob_map = local_ob_mask
            # if np.sum(semantic_feature_map) > 0:
            if torch.count_nonzero(semantic_feature_map).item() > 0:
                semantic_feature_map = semantic_feature_map.to(self.feature_map_device)
                self.global_sem_feature_map[process_ob_map] = semantic_feature_map[process_ob_map]

        # 进行权重融合
        if not only_seg and self.args.fusion_weights and "text_prompt" in info_dict_one:
            semantic_feature_map = semantic_feature_map.to(self.feature_map_device)
            self.global_sem_feature_map = self.global_sem_feature_map * weight_one + semantic_feature_map * weight_two

        return self.global_sem_map, self.global_sem_feature_map

    def nav_frontiers(self, info_dict_one):
        # process_frontiers_mask, vis_frontiers_mask = self.get_frontiers(self.global_sem_map, info_dict_one)
        frontiers_response_dict = self.get_frontiers(self.global_sem_map, info_dict_one)
        robot_position = info_dict_one["robot_position"]
        # self.vis_frontiers_mask = vis_frontiers_mask
        self.vis_frontiers_mask = frontiers_response_dict["vis_frontiers_mask"]
        robot_position_mask = self.get_nav_position_mask([robot_position])
        robot_position_mask = self.dilate_frontiers_mask(robot_position_mask, kernel_size=5, padding=2)
        # select_feature_list, frontiers_label_dict_list = self.get_nav_label(robot_position_mask, self.global_sem_map, vis_frontiers_mask, process_frontiers_mask, self.global_sem_feature_map)
        select_feature_list, frontiers_label_dict_list = self.get_nav_label_v2(robot_position_mask, frontiers_response_dict["frontiers_dict_list"], self.global_sem_feature_map)
        self.frontiers_label_dict_list = frontiers_label_dict_list
        return select_feature_list, frontiers_label_dict_list

    def get_nav_label_v2(self, robot_position_mask, frontiers_dict_list, sem_feature):
        total_feature_num = 256
        sem_feature_map = sem_feature.reshape([-1, 1024])
        # frontiers_dict
        # frontiers_dict_one = {"vis_frontiers": local_vis_frontiers_mask,
        #                       "process_frontiers": local_process_frontiers_mask,
        #                       "props_one": props_one,
        #                       "local_frontiers_mask": local_frontiers_mask}
        # 直接读取dict获取边界mask
        frontiers_label_dict_list = []
        start_ascii = 65
        if len(frontiers_dict_list) > 0:
            process_frontiers_pixel = []
            process_mask_list = []
            for frontiers_dict_index, frontiers_dict_one in enumerate(frontiers_dict_list):
                nav_label = False
                choose_str = chr(start_ascii + frontiers_dict_index)
                centroid = frontiers_dict_one["props_one"].centroid
                mask_one = frontiers_dict_one["local_frontiers_mask"]
                # coords = frontiers_dict_one["props_one"].coords
                bbox = frontiers_dict_one["props_one"].bbox
                frontiers_label_dict_one = dict(choose=choose_str, label=nav_label, centroid=np.asarray(centroid), mask=mask_one, bbox=np.asarray(bbox))
                frontiers_label_dict_list.append(frontiers_label_dict_one)

                process_frontiers = frontiers_dict_one["process_frontiers"]
                process_frontiers_pixel.append(np.sum(process_frontiers))
                process_mask_list.append(process_frontiers)

            # 处理feature
            if self.args.pad_frontiers_token:
                if len(process_frontiers_pixel) > 0:
                    select_feature_list = []
                    for index, frontiers_token_num_one in enumerate(process_frontiers_pixel):
                        token_number = self.args.frontiers_token_number
                        process_mask_one = process_mask_list[index]
                        process_mask_one = process_mask_one.reshape([-1, 1])
                        sem_feature_map_one = sem_feature_map[process_mask_one[:, 0] == 1, :]
                        selected_indices = torch.randperm(sem_feature_map_one.shape[0])[:token_number]
                        select_feature_token = sem_feature_map_one[selected_indices]
                        select_feature_list.append(select_feature_token)
                    select_feature_list = torch.cat(select_feature_list, dim=0)
                    # 使用pad进行补齐
                    pad_size = 256 - select_feature_list.shape[0]
                    # 执行填充操作
                    select_feature_list = torch.nn.functional.pad(select_feature_list, [0, 0, 0, pad_size], value=0)
                else:
                    selected_indices = torch.randperm(sem_feature_map.shape[0])[:total_feature_num]
                    select_feature_list = sem_feature_map[selected_indices]
            else:
                process_frontiers_pixel = np.asarray(process_frontiers_pixel)
                process_frontiers_pixel_percentage = process_frontiers_pixel / np.sum(process_frontiers_pixel)
                used_token_number = 0
                if len(process_frontiers_pixel_percentage) > 0:
                    select_feature_list = []
                    for index, frontiers_token_num_one in enumerate(process_frontiers_pixel_percentage):
                        token_number = int(total_feature_num * frontiers_token_num_one)
                        if index == len(process_frontiers_pixel_percentage) - 1:
                            token_number = total_feature_num - used_token_number
                        process_mask_one = process_mask_list[index]
                        process_mask_one = process_mask_one.reshape([-1, 1])
                        sem_feature_map_one = sem_feature_map[process_mask_one[:, 0] == 1, :]
                        # selected_indices = np.random.choice(sem_feature_map_one.shape[0], token_number, replace=True)
                        # select_feature_token = sem_feature_map_one[selected_indices]
                        selected_indices = torch.randperm(sem_feature_map_one.shape[0])[:token_number]
                        select_feature_token = sem_feature_map_one[selected_indices]
                        select_feature_list.append(select_feature_token)
                        used_token_number = used_token_number + token_number

                    # select_feature_list = np.concatenate(select_feature_list, axis=0)  # [256, 1024]
                    select_feature_list = torch.cat(select_feature_list, dim=0)
                    if select_feature_list.shape[0] < 256:
                        # 使用pad进行补齐
                        pad_size = 256 - select_feature_list.shape[0]
                        # 执行填充操作
                        select_feature_list = torch.nn.functional.pad(select_feature_list, [0, 0, 0, pad_size], value=0)
                    else:
                        select_feature_list = select_feature_list[:256]
                else:
                    # selected_indices = np.random.choice(sem_feature_map.shape[0], 256, replace=True)
                    selected_indices = torch.randperm(sem_feature_map.shape[0])[:256]
                    select_feature_list = sem_feature_map[selected_indices]
        else:
            selected_indices = torch.randperm(sem_feature_map.shape[0])[:256]
            select_feature_list = sem_feature_map[selected_indices]

        add_centroid = self.compute_centroid(robot_position_mask)
        add_bbox = self.compute_bbox(robot_position_mask)
        self.robot_heightmap_point = add_centroid

        assert select_feature_list.shape[0] == 256
        assert select_feature_list.shape[1] == 1024
        return select_feature_list, frontiers_label_dict_list

    def get_nav_label(self, robot_position_mask, sem_map, vis_frontiers_mask, process_frontier_mask, sem_feature):
        total_feature_num = 256
        # local_ob_mask = self.get_ob_map(sem_map)
        # 重新计算连通域
        img_label, num = measure.label(vis_frontiers_mask, connectivity=2, return_num=True)
        props = measure.regionprops(img_label)
        local_frontiers_mask_dict_list = []
        for props_id in range(len(props)):
            local_frontiers_mask = np.zeros_like(vis_frontiers_mask)
            props_one = props[props_id]
            centroid = props_one.centroid  # 质心坐标 [0, y, x]
            coords = props_one.coords  # mask内像素索引 [0, y_index, x_index]
            local_frontiers_mask[img_label == props_id + 1] = 1
            bbox = props_one.bbox  # [0, 0, min_row, min_col, max_row, max_col]
            local_frontiers_mask_dict = dict(centroid=centroid, coords=coords, mask=local_frontiers_mask, bbox=np.asarray(bbox))
            local_frontiers_mask_dict_list.append(local_frontiers_mask_dict)

        frontiers_label_dict_list = []
        start_ascii = 65
        if len(local_frontiers_mask_dict_list) > 0:
            for frontiers_dict_index, frontiers_dict_one in enumerate(local_frontiers_mask_dict_list):
                nav_label = False
                choose_str = chr(start_ascii + frontiers_dict_index)
                centroid = frontiers_dict_one["centroid"]
                mask_one = frontiers_dict_one["mask"]
                coords = frontiers_dict_one["coords"]
                bbox = frontiers_dict_one["bbox"]
                frontiers_label_dict_one = dict(choose=choose_str, label=nav_label, centroid=np.asarray(centroid), mask=mask_one, bbox=bbox)
                frontiers_label_dict_list.append(frontiers_label_dict_one)

        add_centroid = self.compute_centroid(robot_position_mask)
        add_bbox = self.compute_bbox(robot_position_mask)
        self.robot_heightmap_point = add_centroid
        # add_nav_label_dict = dict(choose=chr(start_ascii + len(frontiers_label_dict_list)),
        #     label=True, centroid=add_centroid, mask=robot_position_mask, bbox=add_bbox)
        # frontiers_label_dict_list.append(add_nav_label_dict)

        # 生成frontiers_feature [256, 1024]
        # 重新计算process_frontiers连通域
        img_label, num = measure.label(process_frontier_mask, connectivity=2, return_num=True)
        process_frontiers_props = measure.regionprops(img_label)
        process_mask_list = []
        process_frontiers_pixel = []
        for props_id in range(len(process_frontiers_props)):
            local_process_frontiers_mask = np.zeros_like(process_frontier_mask)
            props_one = process_frontiers_props[props_id]
            centroid = props_one.centroid  # 质心坐标 [0, y, x]
            coords = props_one.coords  # mask内像素索引 [0, y_index, x_index]
            local_process_frontiers_mask[img_label == props_id + 1] = 1
            area = props_one.area
            process_frontiers_pixel.append(area)
            process_mask_list.append(local_process_frontiers_mask)
        sem_feature_map = sem_feature.reshape([-1, 1024])
        if self.args.pad_frontiers_token:
            if len(process_frontiers_pixel) > 0:
                select_feature_list = []
                for index, frontiers_token_num_one in enumerate(process_frontiers_pixel):
                    token_number = self.args.frontiers_token_number
                    process_mask_one = process_mask_list[index]
                    process_mask_one = process_mask_one.reshape([-1, 1])
                    sem_feature_map_one = sem_feature_map[process_mask_one[:, 0] == 1, :]
                    selected_indices = torch.randperm(sem_feature_map_one.shape[0])[:token_number]
                    select_feature_token = sem_feature_map_one[selected_indices]
                    select_feature_list.append(select_feature_token)
                select_feature_list = torch.cat(select_feature_list, dim=0)
                # 使用pad进行补齐
                pad_size = 256 - select_feature_list.shape[0]
                # 执行填充操作
                select_feature_list = torch.nn.functional.pad(select_feature_list, [0, 0, 0, pad_size], value=0)
            else:
                selected_indices = torch.randperm(sem_feature_map.shape[0])[:256]
                select_feature_list = sem_feature_map[selected_indices]
        else:
            process_frontiers_pixel = np.asarray(process_frontiers_pixel)
            process_frontiers_pixel_percentage = process_frontiers_pixel / np.sum(process_frontiers_pixel)
            used_token_number = 0
            if len(process_frontiers_pixel_percentage) > 0:
                select_feature_list = []
                for index, frontiers_token_num_one in enumerate(process_frontiers_pixel_percentage):
                    token_number = int(total_feature_num * frontiers_token_num_one)
                    if index == len(process_frontiers_pixel_percentage) - 1:
                        token_number = total_feature_num - used_token_number
                    process_mask_one = process_mask_list[index]
                    process_mask_one = process_mask_one.reshape([-1, 1])
                    sem_feature_map_one = sem_feature_map[process_mask_one[:, 0] == 1, :]
                    # selected_indices = np.random.choice(sem_feature_map_one.shape[0], token_number, replace=True)
                    # select_feature_token = sem_feature_map_one[selected_indices]
                    selected_indices = torch.randperm(sem_feature_map_one.shape[0])[:token_number]
                    select_feature_token = sem_feature_map_one[selected_indices]
                    select_feature_list.append(select_feature_token)
                    used_token_number = used_token_number + token_number

                # select_feature_list = np.concatenate(select_feature_list, axis=0)  # [256, 1024]
                select_feature_list = torch.cat(select_feature_list, dim=0)
            else:
                # selected_indices = np.random.choice(sem_feature_map.shape[0], 256, replace=True)
                selected_indices = torch.randperm(sem_feature_map.shape[0])[:256]
                select_feature_list = sem_feature_map[selected_indices]
        print(select_feature_list.shape)
        assert select_feature_list.shape[0] == 256
        assert select_feature_list.shape[1] == 1024
        return select_feature_list, frontiers_label_dict_list

    def get_sem_map_alfred(self, point_cloud, feature_mask, robot_region_mask, args_dict):
        # 读取参数
        only_seg = args_dict["only_seg"]
        # prediction信息处理
        semantic_map_list = []
        for i in range(0, self.num_sem_categories - 1):
            object_class_point_cloud = point_cloud[point_cloud[:, 3] == i]
            semantic_map_one = np.zeros(self.heightmap_size)
            if len(object_class_point_cloud) <= 0:
                semantic_map_list.append(semantic_map_one[None, :, :])
                continue

            object_class_point_cloud = object_class_point_cloud[object_class_point_cloud[:, 2] < self.max_y]

            # new add: 维护object instance dict
            if i not in [0, 94]:
                if object_class_point_cloud.shape[0] > 0:
                    if self.args.use_3D_feature:
                        self.update_instance_point_cloud(object_class_point_cloud)

            # 先转换为高度图
            sort_z_ind = np.argsort(object_class_point_cloud[:, 2])
            surface_pts = object_class_point_cloud[sort_z_ind]
            heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(surface_pts[:, 0] > self.min_x, surface_pts[:,
                                                                                                      0] < self.max_x),
                surface_pts[:, 1] > self.min_z), surface_pts[:, 1] < self.max_z)

            surface_pts = surface_pts[heightmap_valid_ind]
            heightmap_pix_x = np.floor((surface_pts[:, 0] - self.min_x) / self.map_resolution).astype(int)
            heightmap_pix_y = np.floor((surface_pts[:, 1] - self.min_z) / self.map_resolution).astype(int)
            # 过滤
            keep_idx = (heightmap_pix_x >= 0) * (heightmap_pix_x < semantic_map_one.shape[-1]) * \
                       (heightmap_pix_y >= 0) * (heightmap_pix_y < semantic_map_one.shape[-2])

            heightmap_pix_x = heightmap_pix_x[keep_idx]
            heightmap_pix_y = heightmap_pix_y[keep_idx]
            surface_pts = surface_pts[keep_idx, :]
            semantic_map_one[heightmap_pix_y, heightmap_pix_x] = surface_pts[:, 2]
            semantic_map_one[semantic_map_one < 0.0] = 0.00
            # 变成mask
            if i > 0:
                if self.args.alfred_scene:
                    semantic_map_one[semantic_map_one != 0] = 1
                else:
                    semantic_map_one[semantic_map_one < 0.01] = 0
                    semantic_map_one[semantic_map_one > 0.01] = 1
                    # semantic_map_one[semantic_map_one != 0] = 1
                # semantic_map_one[semantic_map_one != 0] = 1
            else:
                semantic_map_one[semantic_map_one < 0.0] = 0
                semantic_map_one[semantic_map_one > 0.0] = 1

            if self.args.use_learned_depth:
                semantic_map_list.append(semantic_map_one[None, :, :] * robot_region_mask)
            else:
                semantic_map_list.append(semantic_map_one[None, :, :])
        # 点云预处理 --------------------
        contmap = np.zeros(self.heightmap_size, dtype=int)
        heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(np.logical_and(point_cloud[:,
                                                                                          0] > self.min_x, point_cloud[
                                                                                                           :,
                                                                                                           0] < self.max_x),
            point_cloud[:, 1] > self.min_z), point_cloud[:, 1] < self.max_z), point_cloud[:, 2] < self.max_y)

        # add batch and channel
        semantic_feature_map_shape = torch.zeros([1, 1024, self.heightmap_size[0], self.heightmap_size[1]]).shape
        point_cloud = point_cloud[heightmap_valid_ind]
        # point_cloud_copy = point_cloud.copy()
        feature_mask = feature_mask[heightmap_valid_ind]
        # 计算映射坐标
        heightmap_pix_x = np.floor((point_cloud[:, 0] - self.min_x) / self.map_resolution).astype(int)
        heightmap_pix_y = np.floor((point_cloud[:, 1] - self.min_z) / self.map_resolution).astype(int)
        # 过滤
        keep_idx = (heightmap_pix_x >= 0) * (heightmap_pix_x < self.heightmap_size[1]) * \
                   (heightmap_pix_y >= 0) * (heightmap_pix_y < self.heightmap_size[0])

        heightmap_pix_x = heightmap_pix_x[keep_idx]
        heightmap_pix_y = heightmap_pix_y[keep_idx]
        point_cloud = point_cloud[keep_idx, :]
        # point_cloud_copy = point_cloud_copy[keep_idx, :]

        semantic_height_map = np.zeros(self.heightmap_size)
        semantic_height_map[heightmap_pix_y, heightmap_pix_x] = point_cloud[:, 2]
        semantic_map_wall = np.zeros(self.heightmap_size)
        semantic_map_wall[semantic_height_map > self.args.alfred_height_threshold] = 1
        if self.args.use_learned_depth:
            semantic_map_list.insert(1, semantic_map_wall[None, :, :] * robot_region_mask)
        else:
            semantic_map_list.insert(1, semantic_map_wall[None, :, :])
        assert len(semantic_map_list) == self.num_sem_categories
        semantic_map = np.concatenate(semantic_map_list, axis=0)

        # ------------------------
        if not only_seg:
            # semantic_feature_map = np.zeros([self.heightmap_size[0], self.heightmap_size[1], 1024], dtype=np.float16)
            # 送入GPU
            point_cloud = torch.from_numpy(point_cloud).to(self.ME_device)
            # feature_mask = torch.tensor(feature_mask, dtype=torch.float32).to(self.ME_device)
            feature_mask = feature_mask.to(self.ME_device).float()
            feature_mask = feature_mask[keep_idx, :]
            # feature_mask = feature_mask.astype(np.float32)
            coordinates, features = ME.utils.batch_sparse_collate([
                (point_cloud[:, [1, 0]] / self.map_resolution, feature_mask)], device=self.ME_device)
            keep_idx = (coordinates[:, 0] >= 0) * (coordinates[:, 0] < self.heightmap_size[0]) * \
                       (coordinates[:, 1] >= 0) * (coordinates[:, 1] < self.heightmap_size[1])
            coordinates, features = coordinates[keep_idx], features[keep_idx]
            if len(coordinates) > 0:
                # semantic_feature_map_sparse = ME.TensorField(coordinates=coordinates, features=features,
                #     quantization_mode=ME.SparseTensorQuantizationMode.MAX_POOL).sparse()
                semantic_feature_map_sparse = ME.TensorField(coordinates=coordinates, features=features, quantization_mode=ME.SparseTensorQuantizationMode.MAX_POOL)
                semantic_feature_map_sparse = semantic_feature_map_sparse.sparse()
                semantic_feature_map = \
                semantic_feature_map_sparse.dense(shape=semantic_feature_map_shape, min_coordinate=torch.IntTensor([0,
                                                                                                                    0]))[
                    0]
                semantic_feature_map = semantic_feature_map.permute(0, 2, 3, 1)
                # semantic_feature_map = semantic_feature_map.cpu()
                # torch.cuda.empty_cache()
                # semantic_feature_map = semantic_feature_map.half().numpy()[0]
                semantic_feature_map = semantic_feature_map.half()[0]
            else:
                # torch.cuda.empty_cache()
                # semantic_feature_map = np.zeros([self.heightmap_size[0], self.heightmap_size[1],
                #                                  1024], dtype=np.float16)
                semantic_feature_map = torch.zeros([self.heightmap_size[0], self.heightmap_size[1],
                                                    1024], dtype=torch.float16, device=self.ME_device)
        else:
            # semantic_feature_map = np.zeros([self.heightmap_size[0], self.heightmap_size[1],
            #                                  1024], dtype=np.float16)
            semantic_feature_map = torch.zeros([self.heightmap_size[0], self.heightmap_size[1],
                                                1024], dtype=torch.float16, device=self.ME_device)
        # Update contmap
        np.add.at(contmap, (heightmap_pix_y, heightmap_pix_x), 1)
        ob_mask = np.zeros_like(contmap)
        ob_mask[contmap >= 1] = 1
        ob_mask = ob_mask.astype(np.bool_)

        return semantic_map, semantic_feature_map, ob_mask, semantic_height_map

    def get_sem_map(self, point_cloud, feature_mask):
        # [N, 4] --> [x, z, y, class_id]
        # heightmap_size = np.ceil(((self.map_size - 0) / MAP_RESOLUTION,
        #                           (self - 0) / MAP_RESOLUTION)).astype(int)
        # 首先根据类别进行分层
        # feature_mask = feature_mask.cpu().numpy()
        semantic_map_list = []
        for i in range(0, self.num_sem_categories):
            # i = 0 --> 通路 id = 95 没有用目前
            # id = 1 wall
            object_class_point_cloud = point_cloud[point_cloud[:, 3] == i]
            semantic_map_one = np.zeros(self.heightmap_size)
            if len(object_class_point_cloud) <= 5:
                if i == 94:
                    semantic_map_list.insert(1, semantic_map_one[None, :, :])
                else:
                    semantic_map_list.append(semantic_map_one[None, :, :])
                continue
            object_class_point_cloud = object_class_point_cloud[object_class_point_cloud[:, 2] < self.max_y]

            # new add: 维护object instance dict
            if i not in [0, 94]:
                if object_class_point_cloud.shape[0] > 0:
                    if self.args.use_3D_feature:
                        self.update_instance_point_cloud(object_class_point_cloud)

            # 先转换为高度图
            sort_z_ind = np.argsort(object_class_point_cloud[:, 2])
            surface_pts = object_class_point_cloud[sort_z_ind]
            heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(surface_pts[:, 0] > self.min_x, surface_pts[:,
                                                                                                      0] < self.max_x),
                surface_pts[:, 1] > self.min_z), surface_pts[:, 1] < self.max_z)

            surface_pts = surface_pts[heightmap_valid_ind]
            heightmap_pix_x = np.floor((surface_pts[:, 0] - self.min_x) / self.map_resolution).astype(int)
            heightmap_pix_y = np.floor((surface_pts[:, 1] - self.min_z) / self.map_resolution).astype(int)
            semantic_map_one[heightmap_pix_y, heightmap_pix_x] = surface_pts[:, 2]
            # 变成mask
            semantic_map_one[semantic_map_one < 0] = 0
            semantic_map_one[semantic_map_one > 0] = 1

            if i == 95:
                semantic_map_list.insert(1, semantic_map_one[None, :, :])
            else:
                semantic_map_list.append(semantic_map_one[None, :, :])

        # assert len(semantic_map_list) == 96
        assert len(semantic_map_list) == self.num_sem_categories
        semantic_map = np.concatenate(semantic_map_list, axis=0)
        # 生成sem_feature_map
        contmap = np.zeros(self.heightmap_size, dtype=int)
        # semantic_feature_map = np.zeros([self.heightmap_size[0], self.heightmap_size[1], 1024], dtype=np.float16)
        heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(point_cloud[:, 0] > self.min_x, point_cloud[:,
                                                                                                  0] < self.max_x),
            point_cloud[:, 1] > self.min_z), point_cloud[:, 1] < self.max_z)

        # add batch and channel
        semantic_feature_map_shape = torch.zeros([1, 1024, self.heightmap_size[0], self.heightmap_size[1]]).shape
        point_cloud = point_cloud[heightmap_valid_ind]
        feature_mask = feature_mask[heightmap_valid_ind]
        # 计算映射坐标
        heightmap_pix_x = np.floor((point_cloud[:, 0] - self.min_x) / self.map_resolution).astype(int)
        heightmap_pix_y = np.floor((point_cloud[:, 1] - self.min_z) / self.map_resolution).astype(int)
        # 送入GPU
        point_cloud = torch.from_numpy(point_cloud).to(self.ME_device)
        # feature_mask = torch.tensor(feature_mask, dtype=torch.float32).to(self.ME_device)
        feature_mask = feature_mask.to(self.ME_device).float()
        # feature_mask = feature_mask.astype(np.float32)
        coordinates, features = ME.utils.batch_sparse_collate([(point_cloud[:, [1, 0]] / self.map_resolution, feature_mask)], device=self.ME_device)
        keep_idx = (coordinates[:, 0] >= 0) * (coordinates[:, 0] < self.heightmap_size[0]) * \
                   (coordinates[:, 1] >= 0) * (coordinates[:, 1] < self.heightmap_size[1])
        coordinates, features = coordinates[keep_idx], features[keep_idx]
        if len(coordinates) > 0:
            # semantic_feature_map_sparse = ME.TensorField(coordinates=coordinates, features=features,
            #     quantization_mode=ME.SparseTensorQuantizationMode.MAX_POOL).sparse()
            semantic_feature_map_sparse = ME.TensorField(coordinates=coordinates, features=features, quantization_mode=ME.SparseTensorQuantizationMode.MAX_POOL)
            semantic_feature_map_sparse = semantic_feature_map_sparse.sparse()
            semantic_feature_map = semantic_feature_map_sparse.dense(shape=semantic_feature_map_shape, min_coordinate=torch.IntTensor([0, 0]))[0]
            semantic_feature_map = semantic_feature_map.permute(0, 2, 3, 1)
            # semantic_feature_map = semantic_feature_map.cpu()
            # torch.cuda.empty_cache()
            # semantic_feature_map = semantic_feature_map.half().numpy()[0]
            semantic_feature_map = semantic_feature_map.half()[0]
        else:
            # torch.cuda.empty_cache()
            # semantic_feature_map = np.zeros([self.heightmap_size[0], self.heightmap_size[1], 1024], dtype=np.float16)
            semantic_feature_map = torch.zeros([self.heightmap_size[0], self.heightmap_size[1], 1024], dtype=torch.float16, device=self.ME_device)
        # Update contmap
        np.add.at(contmap, (heightmap_pix_y, heightmap_pix_x), 1)
        ob_mask = np.zeros_like(contmap)
        ob_mask[contmap >= 1] = 1
        ob_mask = ob_mask.astype(np.bool_)
        # start_time = time.time()
        # Update semantic_feature_map
        """
        mask_indices = contmap[heightmap_pix_y, heightmap_pix_x] > 1
        semantic_feature_map[heightmap_pix_y[mask_indices], heightmap_pix_x[mask_indices]] = np.maximum(
            semantic_feature_map[heightmap_pix_y[mask_indices], heightmap_pix_x[mask_indices]],
            feature_mask[mask_indices]
        )
        semantic_feature_map[heightmap_pix_y[~mask_indices], heightmap_pix_x[~mask_indices]] = feature_mask[
            ~mask_indices]
        """
        # mask_indices = contmap[heightmap_pix_y, heightmap_pix_x] > 1
        #
        # # Vectorized operations
        # masked_indices = (heightmap_pix_y[mask_indices], heightmap_pix_x[mask_indices])
        # unmasked_indices = (heightmap_pix_y[~mask_indices], heightmap_pix_x[~mask_indices])
        #
        # # Apply updates in a single step
        # semantic_feature_map[masked_indices] = np.maximum(
        #     semantic_feature_map[masked_indices],
        #     feature_mask[mask_indices]
        # )
        # semantic_feature_map[unmasked_indices] = feature_mask[~mask_indices]

        # print("feature更新耗时: {:.2f}秒".format(time.time() - start_time))
        # [1, 1024, 1200, 1200]
        # semantic_feature_map = torch.tensor(semantic_feature_map_t[0].permute(2, 1, 0), dtype=torch.float16)
        # semantic_feature_map = semantic_feature_map.permute(0, 2, 3, 1).half()
        # semantic_feature_map = semantic_feature_map[0].numpy().astype(np.float16)
        # semantic_feature_map = np.transpose(semantic_feature_map, (2, 1, 0))

        return semantic_map, semantic_feature_map, ob_mask

    def get_point_cloud(self, depth_one, masks, info_dict_one):
        fov = info_dict_one["fov"]
        cameraHorizon = info_dict_one["cameraHorizon"]
        camera_world_xyz = info_dict_one["camera_world_xyz"]
        if not isinstance(camera_world_xyz, np.ndarray):
            camera_world_xyz = np.asarray(camera_world_xyz)
        rotation = info_dict_one["rotation"]
        label_dict_list = info_dict_one["label_info"]

        # 映射点云
        camera_space_point_cloud = self.cpu_only_depth_frame_to_camera_space_xyz(depth_one, mask=None, fov=fov)
        partial_point_cloud = self.cpu_only_camera_space_xyz_to_world_xyz(camera_space_point_cloud,
            camera_world_xyz, rotation, cameraHorizon)

        semantic_seg = {}
        for label_dict_index, label_dict_one in label_dict_list.items():
            try:
                cat = self.total_cat2idx[label_dict_one["class_name"]]
            except:
                continue
            mask_one = masks[label_dict_index]
            mask_one = mask_one.astype(np.bool_)
            if cat not in semantic_seg.keys():
                semantic_seg[cat] = []
                semantic_seg[cat].append(mask_one)
            else:
                semantic_seg[cat].append(mask_one)

        class_mask_list = []
        occ_mask_list = []
        for semantic_seg_one_id in range(self.num_sem_categories):
            if semantic_seg_one_id not in semantic_seg.keys():
                continue
            mask_list = semantic_seg[semantic_seg_one_id]

            for mask_one in mask_list:
                class_mask_one = np.zeros(depth_one.shape)  # [mask_one][None, :] * semantic_seg_one_id
                select_mask = np.ones(depth_one.shape, dtype=bool)
                class_mask_one[mask_one] = class_mask_one[mask_one] + semantic_seg_one_id
                class_mask_one = class_mask_one[select_mask][None, :]
                occ_mask_one = np.zeros(class_mask_one.shape)
                occ_mask_one[class_mask_one > 0] = 1
                class_mask_list.append(class_mask_one)
                occ_mask_list.append(occ_mask_one)

        # 整体拼接
        if len(class_mask_list) > 0:
            # 预处理
            occ_mask_list = np.concatenate(occ_mask_list, axis=0)
            occ_mask_list = np.sum(occ_mask_list, axis=0)
            occ_mask = np.ones(occ_mask_list.shape)
            occ_mask[occ_mask_list > 1] = 0
            class_mask_list = np.concatenate(class_mask_list, axis=0)
            # class_mask = np.sum(class_mask_list, axis=0)[None, :]
            class_mask = np.sum(class_mask_list, axis=0) * occ_mask
            class_mask = class_mask[None, :]
        else:
            class_mask = np.zeros([1, partial_point_cloud.shape[1]])

        world_space_point_cloud = np.concatenate((partial_point_cloud, class_mask), axis=0)
        world_space_point_cloud = world_space_point_cloud.T
        # [x, y, z] --> [x, z, y]
        world_space_point_cloud[:, [0, 1, 2, 3]] = world_space_point_cloud[:, [0, 2, 1, 3]]
        world_space_point_cloud = world_space_point_cloud.astype(np.float16)
        return world_space_point_cloud

    def extract_feature(self, rgb, masks):
        LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH = rgb.shape[0], rgb.shape[1]
        global_feat = None
        # 转换RGB格式
        global_rgb = Image.fromarray(rgb)
        with torch.cuda.amp.autocast():
            # print("Extracting global CLIP features...")
            _img = self.preprocess(global_rgb).unsqueeze(0)
            global_feat = self.clip_model.encode_image(_img.to(self.open_clip_device))
            global_feat /= global_feat.norm(dim=-1, keepdim=True)

        global_feat = global_feat.half().to(self.open_clip_device)
        global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # --> (1, 1024)
        feat_dim = global_feat.shape[-1]
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        feat_per_roi = []
        roi_nonzero_inds = []
        similarity_scores = []

        for maskidx in range(len(masks)):
            cur_mask = masks[maskidx]
            bbox = self.get_bbox_around_mask_cpu(cur_mask)
            x0, y0, x1, y1 = bbox
            nonzero_inds = torch.argwhere(torch.from_numpy(masks[maskidx]))
            bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
            img_area = LOAD_IMG_WIDTH * LOAD_IMG_HEIGHT
            iou = bbox_area / img_area

            if iou < 0.005:  # 差不多为1/10
                continue

            img_roi = rgb[x0:x1, y0:y1]
            img_roi = Image.fromarray(img_roi)
            img_roi = self.preprocess(img_roi).unsqueeze(0).to(self.open_clip_device)
            roifeat = self.clip_model.encode_image(img_roi)
            roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
            feat_per_roi.append(roifeat)
            roi_nonzero_inds.append(nonzero_inds)
            _sim = cosine_similarity(global_feat, roifeat)
            similarity_scores.append(_sim)
        if len(similarity_scores) < 1:
            # return torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, feat_dim, dtype=torch.half)
            return torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, feat_dim, dtype=torch.half, device=self.open_clip_device)
        similarity_scores = torch.cat(similarity_scores)
        softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0)
        # outfeat = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, feat_dim, dtype=torch.half)
        outfeat = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, feat_dim, dtype=torch.half, device=self.open_clip_device)
        for maskidx in range(softmax_scores.shape[0]):
            _weighted_feat = softmax_scores[maskidx] * global_feat + (1 - softmax_scores[maskidx]) * feat_per_roi[
                maskidx]
            _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
            # outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] += _weighted_feat[
            #     0].detach().cpu().half()
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] += _weighted_feat[0].half()
            outfeat[
                roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] = torch.nn.functional.normalize(
                outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]].float(), dim=-1
            ).half()

        outfeat = outfeat.unsqueeze(0).float()  # interpolate is not implemented for float yet in pytorch
        outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
        # outfeat = torch.nn.functional.interpolate(outfeat, [args.desired_height,
        #                                                     args.desired_width], mode="nearest")
        outfeat = torch.nn.functional.interpolate(outfeat, [LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH], mode="nearest")
        outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
        outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
        outfeat = outfeat[0].half()  # --> H, W, feat_dim

        return outfeat

    def get_bbox_around_mask_cpu(self, mask):
        bbox = None
        nonzero_inds = np.nonzero(mask)
        if nonzero_inds[0].shape[0] == 0:
            topleft = [0, 0]
            botright = [mask.shape[0], mask.shape[1]]
            bbox = [topleft[0], topleft[1], botright[0], botright[1]]  # (x0, y0, x1, y1)
            topleft_x = topleft[0]
            topleft_y = topleft[1]
            botright_x = botright[0]
            botright_y = botright[1]
        else:
            topleft_x = nonzero_inds[0].min()  # H
            topleft_y = nonzero_inds[1].min()  # H
            botright_x = nonzero_inds[0].max()  # W
            botright_y = nonzero_inds[0].max()  # W
            # bbox = [topleft_x, topleft_y, botright_x, botright_y]

        # 增加过滤
        topleft_x = np.maximum(topleft_x, 0)
        topleft_y = np.maximum(topleft_y, 0)
        botright_x = np.minimum(botright_x, mask.shape[1])
        botright_y = np.minimum(botright_y, mask.shape[0])
        bbox = (topleft_x, topleft_y, botright_x, botright_y)
        return bbox

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

    def mask_true_false(self, A, B):
        return np.logical_and(A, np.logical_not(B))

    def set_false_in_C(self, mask, C):
        return np.where(mask, False, C)

    def get_ob_map(self, sem_map):
        ob_map = np.zeros([sem_map.shape[1], sem_map.shape[2]])
        _ob_map = np.sum(sem_map, axis=0)
        ob_map[_ob_map > 0] = 1
        return ob_map.astype(np.bool_)

    def get_frontiers(self, sem_map, info_dict_one):
        original_sem_map = sem_map.copy()
        sem_map, ob_map, _ob_map = self.process_map(sem_map)
        in_semmap = sem_map.copy()
        in_semmap = torch.from_numpy(in_semmap).bool()
        in_semmap = in_semmap.unsqueeze(0)
        N = in_semmap.shape[1]
        free_map = in_semmap[0, self.floor_id]
        free_map = free_map.float().unsqueeze(0).unsqueeze(1)
        free_map = torch.nn.functional.max_pool2d(
            free_map, 7, stride=1, padding=3
        )
        free_map = free_map.bool().squeeze(1).squeeze(0)

        exp_map = torch.any(in_semmap, dim=1)[0]  # (H, W)
        exp_map = exp_map | free_map
        unk_map = ~exp_map
        unk_map = unk_map.numpy()
        free_map = free_map.numpy()
        frontiers = self.get_frontiers_np(unk_map, free_map)  # (H, W)

        # 针对frontiers_mask做一些后处理
        frontiers = frontiers - ob_map
        frontiers[frontiers > 1] = 1
        frontiers[frontiers < 0] = 0
        frontiers = torch.from_numpy(frontiers).unsqueeze(0).unsqueeze(1)
        # Dilate the frontiers mask
        frontiers_mask = torch.nn.functional.max_pool2d(
            frontiers.float(), 3, stride=1, padding=1
        ).bool()  # (1, N or 1, H, W)

        # print(frontiers_mask.shape)
        frontiers_mask = frontiers_mask.cpu().numpy().astype(np.int32).reshape([1, frontiers_mask.shape[-2],
                                                                                frontiers_mask.shape[-1]])

        # add 去除周围边界
        # robot_position = info_dict_one["robot_position"]
        # robot_position_mask = self.get_nav_position_mask([robot_position])
        # robot_region_mask = self.get_robot_region(robot_position_mask)
        # robot_region_mask = np.expand_dims(robot_region_mask, axis=0)
        #
        # frontiers_mask = np.multiply(frontiers_mask, robot_region_mask)
        # add 去除ob_map离散边界
        frontiers_mask = self.remove_out_range_frontiers(frontiers_mask, sem_map)

        # 计算扩张后的frontiers_mask
        process_frontiers_mask = np.zeros_like(frontiers_mask)
        vis_frontiers_mask = np.zeros_like(frontiers_mask)
        process_frontiers_id_list = []
        img_label, num = measure.label(frontiers_mask, connectivity=2, return_num=True)
        props = measure.regionprops(img_label)

        # add frontiers_dict_list
        frontiers_dict_list = []

        # 下面根据props进行筛除
        for props_id in range(len(props)):
            local_frontiers_mask = np.zeros_like(frontiers_mask)
            props_one = props[props_id]
            # frontiers属性
            pixels_num = props_one.area
            if pixels_num > self.frontiers_thresholds:
                local_frontiers_mask[img_label == props_id + 1] = 1
                # local_frontiers_mask = cv2.dilate(local_frontiers_mask.astype(np.uint8), kernel)
                local_vis_frontiers_mask = self.dilate_frontiers_mask(local_frontiers_mask, kernel_size=3, padding=1)
                local_process_frontiers_mask = self.dilate_frontiers_mask(local_frontiers_mask, kernel_size=7, padding=3)
                local_process_frontiers_mask = np.multiply(local_process_frontiers_mask, np.expand_dims(_ob_map, axis=0))

                process_frontiers_mask[local_process_frontiers_mask == 1] = 1  # for token
                vis_frontiers_mask[local_vis_frontiers_mask == 1] = 1  # for vis
                process_frontiers_id_list.append(props_id + 1)

                # frontiers_dict
                frontiers_dict_one = {"vis_frontiers": local_vis_frontiers_mask,
                                      "process_frontiers": local_process_frontiers_mask,
                                      "props_one": props_one,
                                      "local_frontiers_mask": local_frontiers_mask}

                frontiers_dict_list.append(frontiers_dict_one)

        # 使用dict进行回复
        response_dict = {"process_frontiers_mask": process_frontiers_mask,
                         "vis_frontiers_mask": vis_frontiers_mask,
                         "frontiers_dict_list": frontiers_dict_list}

        # return process_frontiers_mask, vis_frontiers_mask
        return response_dict

    def process_map(self, sem_map):
        sem_map_list = []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # ex_map = sem_map[1:, :, :].copy()
        ex_map = sem_map.copy()
        # 逐步扩张
        for ex_map_one in ex_map:
            ex_map_one = cv2.dilate(ex_map_one, kernel)
            sem_map_list.append(ex_map_one)
        ex_map = np.stack(sem_map_list, axis=0)
        # 障碍物图
        ob_map_list = np.stack(sem_map_list[1:], axis=0)
        ob_map = np.sum(ob_map_list, axis=0)
        # ob_map[ob_map > 1] = 1
        ob_map[ob_map > 0] = 1

        _ob_map = np.sum(sem_map, axis=0)
        # _ob_map[_ob_map > 1] = 1
        _ob_map[_ob_map > 0] = 1

        has_exp_map = np.zeros([ex_map.shape[1], ex_map.shape[2]])
        sem_map_sum = np.sum(ex_map, axis=0)
        has_exp_map[sem_map_sum >= 1] = 1

        # 所有ex_map sum一下
        ex_map_sum = np.sum(ex_map[1:, :, :], axis=0)
        nav_map = np.zeros_like(ex_map_sum)
        nav_map[ex_map_sum == 0] = 1

        nav_map = np.multiply(nav_map, has_exp_map)
        nav_map = np.expand_dims(nav_map, axis=0)
        process_sem_map = np.concatenate((nav_map, ex_map[1:, :, :]), axis=0)
        return process_sem_map, ob_map, _ob_map

    def get_frontiers_np(self, unexp_map: np.array, free_map: np.array):
        r"""
        Computes the map frontiers given unexplored and free spaces on the map.
        Works for numpy arrays. Reference:
        https://github.com/facebookresearch/exploring_exploration/blob/09d3f9b8703162fcc0974989e60f8cd5b47d4d39/exploring_exploration/models/frontier_agent.py#L132

        Args:
            unexp_map - (H, W) int numpy array with 1 for unexplored cells, 0 o/w.
            free_map - (H, W) int numpy array with 1 for explored free cells, 0 o/w.

        Outputs:
            frontiers - (H, W) boolean numpy array
        """
        unexp_map_shiftup = np.pad(
            unexp_map, ((0, 1), (0, 0)), mode="constant", constant_values=0
        )[1:, :]
        unexp_map_shiftdown = np.pad(
            unexp_map, ((1, 0), (0, 0)), mode="constant", constant_values=0
        )[:-1, :]
        unexp_map_shiftleft = np.pad(
            unexp_map, ((0, 0), (0, 1)), mode="constant", constant_values=0
        )[:, 1:]
        unexp_map_shiftright = np.pad(
            unexp_map, ((0, 0), (1, 0)), mode="constant", constant_values=0
        )[:, :-1]
        frontiers = (
                            (free_map == unexp_map_shiftup)
                            | (free_map == unexp_map_shiftdown)
                            | (free_map == unexp_map_shiftleft)
                            | (free_map == unexp_map_shiftright)
                    ) & (
                            free_map == 1
                    )  # (H, W)

        return frontiers

    def dilate_frontiers_mask(self, frontiers, kernel_size, padding):
        frontiers = torch.from_numpy(frontiers).unsqueeze(0)
        # Dilate the frontiers mask
        frontiers_mask = torch.nn.functional.max_pool2d(
            frontiers.float(), kernel_size, stride=1, padding=padding
        ).bool()  # (1, N or 1, H, W)

        frontiers_mask = frontiers_mask.cpu().numpy().astype(np.int32).reshape([1, frontiers_mask.shape[-2],
                                                                                frontiers_mask.shape[-1]])
        return frontiers_mask

    def compute_centroid(self, label):
        # 计算连通域内所有像素的坐标的平均值
        # rows, cols = np.nonzero(label)
        if np.sum(label) <=0 :
            return None
        index_list = np.nonzero(label)
        # centroid = np.mean(rows), np.mean(cols)
        centroid = [np.mean(index_one) for index_one in index_list]
        return np.asarray(centroid)


    def compute_bbox(self, label):
        if np.sum(label) <= 0:
            return np.asarray([0, 0])
        index_list = np.nonzero(label)
        bbox_min = [np.min(index_one) for index_one in index_list]
        bbox_max = [np.max(index_one) for index_one in index_list]
        bbox = []
        for i in range(len(bbox_min)):
            bbox.extend([bbox_min[i], bbox_max[i]])
        return np.asarray(bbox)

    def get_nav_position_mask(self, nav_path):
        # heightmap_size = np.ceil(((self.map_size - 0) / MAP_RESOLUTION,
        #                           (MAP_SIZE - 0) / MAP_RESOLUTION)).astype(int)
        heightmap_mask = np.zeros(self.heightmap_size)
        nav_path_x = []
        nav_path_z = []
        for nav_path_one in nav_path:
            nav_path_x.append(nav_path_one[0])
            nav_path_z.append(nav_path_one[-1])
        nav_path_x = np.asarray(nav_path_x)
        nav_path_z = np.asarray(nav_path_z)

        heightmap_pix_x = np.floor((nav_path_x - self.min_x) / self.map_resolution).astype(int)
        heightmap_pix_y = np.floor((nav_path_z - self.min_z) / self.map_resolution).astype(int)

        # print(heightmap_pix_x)
        # print(heightmap_pix_y)

        # 过滤
        keep_idx = (heightmap_pix_x >= 0) * (heightmap_pix_x < heightmap_mask.shape[-1]) * \
                   (heightmap_pix_y >= 0) * (heightmap_pix_y < heightmap_mask.shape[-2])

        heightmap_pix_x = heightmap_pix_x[keep_idx]
        heightmap_pix_y = heightmap_pix_y[keep_idx]

        heightmap_mask[heightmap_pix_y, heightmap_pix_x] = 1

        return heightmap_mask

    def pixel2world_point(self, pixel_point):
        pix_x = pixel_point[-1]
        pix_y = pixel_point[1]
        world_point_x = pix_x * self.map_resolution + self.min_x
        world_point_y = pix_y * self.map_resolution + self.min_z
        return np.asarray([world_point_x, world_point_y])

    def fine_grained_sem_map_update(self, semantic_map, info_dict_one):
        target_name = info_dict_one["interactive_object"].split("|")[0]
        cat = self.total_cat2idx[target_name] + 1
        global_cat_sem_map = self.global_sem_map[cat]
        local_cat_sem_map = semantic_map[cat]
        global_cat_sem_ob_map = self.get_ob_map(np.expand_dims(global_cat_sem_map, axis=0))
        local_cat_ob_mask = self.get_ob_map(np.expand_dims(local_cat_sem_map, axis=0))
        # update 地图
        if np.sum(global_cat_sem_ob_map) > 0:
            process_mask = self.mask_true_false(global_cat_sem_ob_map, local_cat_ob_mask)
            process_ob_map = self.set_false_in_C(process_mask, local_cat_ob_mask)
        else:
            process_ob_map = local_cat_ob_mask
        global_cat_sem_map[process_ob_map] = local_cat_sem_map[process_ob_map]
        self.global_sem_map[cat] = global_cat_sem_map

    def get_sem_map_object_position(self, target_object_id):
        if "|" not in target_object_id:
            target_name = target_object_id
        else:
            target_name = target_object_id.split("|")[0]
        # cat = self.total_cat2idx[target_name] + 1
        # print(cat)

        cat = -1
        if "knife" in target_name.lower():
            for object_name, cat_id in self.total_cat2idx.items():
                if target_name.lower() in object_name.lower():
                    cat = cat_id + 1
                    break
        else:
            for object_name, cat_id in self.total_cat2idx.items():
                if target_name.lower() == object_name.lower():
                    cat = cat_id + 1
                    break

        sem_map, ob_map, _ob_map = self.process_map(self.global_sem_map)
        cat_map = sem_map[cat]
        # 计算连通域, 得到中心点
        img_label, num = measure.label(cat_map[None, :, :], connectivity=2, return_num=True)
        props = measure.regionprops(img_label)
        cat_map_centroid_list = []
        area_list = []
        for props_id in range(len(props)):
            props_one = props[props_id]
            area = props_one.area
            centroid = props_one.centroid  # 质心坐标 [0, y, x]
            # coords = props_one.coords  # mask内像素索引 [0, y_index, x_index]
            # bbox = props_one.bbox  # [0, 0, min_row, min_col, max_row, max_col]
            cat_map_centroid_list.append(centroid)
            area_list.append(area)
        sort_area_list = np.argsort(np.asarray(area_list))[::-1]
        if len(cat_map_centroid_list) > 0:
            nav_point = self.pixel2world_point(cat_map_centroid_list[sort_area_list[0]])
            return nav_point
        else:
            return None

    def get_sem_map_object_position_list(self, target_object_id):
        if "|" not in target_object_id:
            target_name = target_object_id
        else:
            target_name = target_object_id.split("|")[0]
        if target_name not in self.total_cat2idx.keys():
            print("not ours target")
            return []
        cat = self.total_cat2idx[target_name] + 1
        # print(cat)
        sem_map, ob_map, _ob_map = self.process_map(self.global_sem_map)
        cat_map = sem_map[cat]
        # 计算连通域, 得到中心点
        img_label, num = measure.label(cat_map[None, :, :], connectivity=2, return_num=True)
        props = measure.regionprops(img_label)
        cat_map_centroid_list = []
        nav_point_list = []
        area_list = []
        for props_id in range(len(props)):
            props_one = props[props_id]
            area = props_one.area
            centroid = props_one.centroid  # 质心坐标 [0, y, x]
            # coords = props_one.coords  # mask内像素索引 [0, y_index, x_index]
            # bbox = props_one.bbox  # [0, 0, min_row, min_col, max_row, max_col]
            cat_map_centroid_list.append(centroid)
            area_list.append(area)
        sort_area_list = np.argsort(np.asarray(area_list))[::-1]
        nav_point_list = []
        for center_point_index in sort_area_list:
            center_point_one = cat_map_centroid_list[center_point_index]
            nav_point_one = self.pixel2world_point(center_point_one)
            nav_point_list.append(nav_point_one)
        return nav_point_list

    def get_sem_map_object_position_by_name(self, object_name):
        cat = self.total_cat2idx[object_name] + 1
        sem_map, ob_map, _ob_map = self.process_map(self.global_sem_map)
        cat_map = sem_map[cat]
        # 计算连通域, 得到中心点
        img_label, num = measure.label(cat_map[None, :, :], connectivity=2, return_num=True)
        props = measure.regionprops(img_label)
        cat_map_centroid_list = []
        area_list = []
        for props_id in range(len(props)):
            props_one = props[props_id]
            centroid = props_one.centroid  # 质心坐标 [0, y, x]
            area = props_one.area
            # coords = props_one.coords  # mask内像素索引 [0, y_index, x_index]
            # bbox = props_one.bbox  # [0, 0, min_row, min_col, max_row, max_col]
            cat_map_centroid_list.append(centroid)
            area_list.append(area)
        sort_area_list = np.argsort(np.asarray(area_list))[::-1]
        nav_point_list = []
        for center_point_index in sort_area_list:
            center_point_one = cat_map_centroid_list[center_point_index]
            nav_point_one = self.pixel2world_point(center_point_one)
            nav_point_list.append(nav_point_one)
        return nav_point_list

    def l3mvn_planner(self, target_name):
        region_size = 60
        frontiers_object_list = {}
        for frontiers_id, frontiers_one in enumerate(self.frontiers_label_dict_list):
            object_name_list_one = []
            centroid = frontiers_one["centroid"]
            center_y = centroid[1]
            center_x = centroid[-1]
            min_x = center_x - region_size / 2
            max_x = center_x + region_size / 2
            min_y = center_y - region_size / 2
            max_y = center_y + region_size / 2
            min_x = np.maximum(min_x, 0)
            min_y = np.maximum(min_y, 0)
            max_x = np.minimum(max_x, self.heightmap_size[1])
            max_y = np.minimum(max_y, self.heightmap_size[0])
            frontiers_region = self.global_sem_map[2:, int(min_y):int(max_y), int(min_x):int(max_x)]
            nonzero_indices = np.nonzero(frontiers_region)[0]
            object_class_index = np.unique(nonzero_indices) + 3
            for object_class_one in object_class_index:
                if object_class_one in self.total_idx2cat:
                    object_name_list_one.append(self.total_idx2cat[object_class_one])
            frontiers_object_list[frontiers_id] = object_name_list_one
        # 进行打分
        frontier_score_list = []
        for frontiers_id, object_name_list in frontiers_object_list.items():
            if len(object_name_list) > 0:
                # ref_dist = F.softmax(self.construct_dist(object_name_list),
                #     dim=0).to(self.device)
                ref_dist = F.softmax(self.construct_dist(object_name_list),
                    dim=0)
                new_dist = ref_dist
                if target_name in list(self.total_cat2idx.keys()):
                    frontier_score_list.append(new_dist[list(self.total_cat2idx).index(target_name)].cpu())
                else:
                    frontier_score_list.append(0.1)
            else:
                frontier_score_list.append(0.1)
        if len(frontier_score_list) > 0:
            frontiers_select_id = np.argmax(np.asarray(frontier_score_list))
            select_frontier = self.frontiers_label_dict_list[frontiers_select_id]
            nav_point = select_frontier["centroid"]
        else:
            nav_point = np.asarray([0, 0, 0])
        return nav_point

    def construct_dist(self, objs):
        query_str = "A room containing "
        for ob in objs:
            query_str += ob + ", "
        query_str += "and"

        TEMP = []
        for label in list(self.total_cat2idx.keys()):
            TEMP_STR = query_str + " "
            TEMP_STR += label + "."

            score = self.scoring_fxn(TEMP_STR)
            TEMP.append(score)
        dist = torch.tensor(TEMP)

        return dist

    def scoring_fxn(self, text):
        tokens_tensor = self.tokenizer.encode(text,
            add_special_tokens=False,
            return_tensors="pt").to(self.llm_device)
        with torch.no_grad():
            output = self.llm_model(tokens_tensor, labels=tokens_tensor)
            loss = output[0]

            return -loss

    def get_robot_region(self, robot_position_mask, region_size=0.25):
        robot_region_mask = np.zeros(robot_position_mask.shape, dtype=np.int32)
        # 计算中心点
        robot_position_mask_center = self.compute_centroid(robot_position_mask)  # [y, x]
        if robot_position_mask_center is None:
            return robot_region_mask
        region_size = int(region_size / self.map_resolution) * 2
        center_y = robot_position_mask_center[0]
        center_x = robot_position_mask_center[-1]
        min_x = center_x - region_size / 2
        max_x = center_x + region_size / 2
        min_y = center_y - region_size / 2
        max_y = center_y + region_size / 2
        min_x = int(np.maximum(min_x, 0))
        min_y = int(np.maximum(min_y, 0))
        max_x = int(np.minimum(max_x, self.heightmap_size[1]))
        max_y = int(np.minimum(max_y, self.heightmap_size[0]))
        robot_region_mask[min_y:max_y, min_x:max_x] = 1
        return robot_region_mask

    def remove_out_range_frontiers(self, frontiers_mask, sem_map):
        sem_ob_map = self.get_ob_map(sem_map).astype(np.int32)
        sem_map_mask = np.zeros(sem_ob_map.shape, dtype=np.int32)
        img_label, num = measure.label(sem_ob_map, connectivity=2, return_num=True)
        props = measure.regionprops(img_label)
        number_list = []
        bbox_list = []
        # 下面根据props进行筛除
        for props_id in range(len(props)):
            props_one = props[props_id]
            pixels_num = props_one.area
            bbox = props_one.bbox
            number_list.append(pixels_num)
            bbox_list.append(bbox)
        if len(number_list) > 0:
            max_index = np.argmax(np.asarray(number_list))
            max_bbox = bbox_list[max_index]
            sem_map_mask[max_bbox[0]:max_bbox[2], max_bbox[1]:max_bbox[3]] = 1
            frontiers_mask = np.multiply(frontiers_mask, np.expand_dims(sem_map_mask, axis=0))
            return frontiers_mask
        else:
            return frontiers_mask

    def get_point_cloud_du(self, depth):
        # point_cloud_t = du.get_point_cloud_from_z_t(depth, self.camera_matrix, self.device, scale=self.du_scale)
        point_cloud_t = du.get_point_cloud_from_z(depth, self.camera_matrix, scale=self.du_scale)
        agent_view_t = du.transform_camera_view(point_cloud_t, self.agent_height, self.view_angles)
        agent_view_centered_t = du.transform_pose(agent_view_t, self.shift_loc)

    def set_view_angles(self, view_angle):
        self.view_angles = [-view_angle]

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

    def get_object_name_list_from_map(self):
        # self.total_idx2cat
        global_sem_map = self.global_sem_map
        object_name_list = []
        for index, global_sem_map_one in enumerate(global_sem_map):
            object_index = index - 1
            if np.sum(global_sem_map_one) > 0:
                if object_index in self.total_idx2cat.keys():
                    object_name_list.append(self.total_idx2cat[object_index])
        object_name_list = list(set(object_name_list))
        return object_name_list

    def check_is_found_object_in_map(self, target):
        object_name_list = self.get_object_name_list_from_map()
        is_find = False
        if "knife" in target.lower():
            for object_one in object_name_list:
                if "knife" in object_one.lower():
                    is_find = True
                    break
        else:
            for object_one in object_name_list:
                if target.lower() in object_one.lower():
                    is_find = True
                    break
        return is_find

    def remove_unseen_target_map(self, object_name, robot_position, remove_type="unseen"):
        # agent靠近物体, 但没有找到目标
        # robot_position = np.asarray(list(self.last_event.metadata["agent"]["position"].values()))
        robot_position_mask = self.get_nav_position_mask([robot_position])
        if remove_type == "unseen":
            robot_region_mask = self.get_robot_region(robot_position_mask, region_size=self.args.remove_object_region_size)
        else:
            robot_region_mask = self.get_robot_region(robot_position_mask, region_size=1.5)
        robot_region_mask = (~robot_region_mask.astype(np.bool_)).astype(np.int32)
        cat = self.total_cat2idx[object_name] + 1
        self.global_sem_map[cat] = self.global_sem_map[cat] * robot_region_mask

    def get_object_position_dict(self):
        object_position_dict = {}
        object_name_list = self.get_object_name_list_from_map()
        for object_name_one in object_name_list:
            if object_name_one not in self.total_cat2idx.keys():
                continue
            # 已经是世界坐标
            object_position_list = self.get_sem_map_object_position_by_name(object_name_one)
            if object_name_one not in object_position_dict.keys():
                object_position_dict[object_name_one] = []
                object_position_dict[object_name_one].extend(object_position_list)
            else:
                object_position_dict[object_name_one].extend(object_position_list)
        return object_position_dict

    def get_object_id_from_sem_map(self, input_dict):
        metadata = input_dict["metadata"]
        target_name = input_dict["target_name"]
        if "Sink" in target_name:
            target_name = "SinkBasin"
        robot_position_dict = metadata["agent"]["position"]
        robot_position_list = (robot_position_dict["x"], robot_position_dict["y"], robot_position_dict["z"])
        object_metadata = metadata["objects"]
        object_id_list = []
        distance_list = []
        for object_dict_one in object_metadata:
            object_position_dict_one = object_dict_one["position"]
            object_id_one = object_dict_one["objectId"]
            if "Sink" in target_name:
                object_name_one = object_dict_one["objectType"]
            else:
                object_name_one = object_id_one.split("|")[0]
            object_position_tuple_one = (object_position_dict_one["x"], object_position_dict_one["y"], object_position_dict_one["z"])
            distance = np.sqrt((object_position_tuple_one[0] - robot_position_list[0]) ** 2 +\
                               (object_position_tuple_one[-1] - robot_position_list[-1]) ** 2)

            if "knife" in target_name.lower():
                if target_name.lower() in object_name_one.lower():
                    distance_list.append(distance)
                    object_id_list.append(object_id_one)
            else:
                if object_name_one.lower() == target_name.lower():
                    distance_list.append(distance)
                    object_id_list.append(object_id_one)

        response_dict = {}
        if len(distance_list) > 0:
            nearest_index = np.argmin(np.asarray(distance_list))
            min_distance = distance_list[nearest_index]
            nearest_object_id = object_id_list[nearest_index]

            # 0.75
            if min_distance < 0.75:
                response_dict["success"] = True
            else:
                response_dict["success"] = False
            response_dict["target_id"] = nearest_object_id
        else:
            response_dict["success"] = False
            response_dict["target_id"] = None
        return response_dict

    def get_vis_similarity_map(self, point_cloud, similarity_score):
        heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(np.logical_and(point_cloud[:,
                                                                                          0] > self.min_x, point_cloud[
                                                                                                           :,
                                                                                                           0] < self.max_x),
            point_cloud[:, 1] > self.min_z), point_cloud[:, 1] < self.max_z), point_cloud[:, 2] < self.max_y)

        # add batch and channel
        filter_point_cloud = point_cloud[heightmap_valid_ind]
        # 计算映射坐标
        heightmap_pix_x = np.floor((filter_point_cloud[:, 0] - self.min_x) / self.map_resolution).astype(int)
        heightmap_pix_y = np.floor((filter_point_cloud[:, 1] - self.min_z) / self.map_resolution).astype(int)

        height_map = np.zeros(self.heightmap_size)
        height_map[heightmap_pix_y, heightmap_pix_x] = filter_point_cloud[:, 2]
        similarity_map = np.zeros(self.heightmap_size)
        similarity_map[height_map > 0] = 1
        similarity_map = similarity_map * similarity_score

        return similarity_map

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

    def update_instance_point_cloud(self, point_cloud):
        # point_cloud: [n, 4] --> [x, y, z, c]
        # index start from 1 expect 94
        class_index = np.unique(point_cloud[:, 3])[0]
        # 遍历所有instance
        has_merge = False
        for instance_index, object_dict in self.object_instance_dict.items():
            object_class_one = object_dict["class_index"]
            object_instance_point_cloud = object_dict["pts"]
            source_point_cloud = point_cloud[:, :3]
            if class_index == object_class_one:
                is_same = self.check_is_same_object(source_point_cloud, object_instance_point_cloud)
                # 如果相同, 则进行合并
                if is_same:
                    has_merge = True
                    new_instance = self.merge_instance_object_point(source_point_cloud, object_instance_point_cloud)
                    object_dict["pts"] = new_instance
                    # 更新中心点
                    centroid = np.mean(new_instance, axis=0)
                    centroid_pix_x = np.floor((centroid[0] - self.min_x) / self.map_resolution).astype(int)
                    centroid_pix_y = np.floor((centroid[1] - self.min_z) / self.map_resolution).astype(int)
                    object_dict["center_point"] = np.asarray([centroid_pix_x, centroid_pix_y])
                    break
        # 如果没有匹配上, 增加instance
        if not has_merge:
            add_index = len(list(self.object_instance_dict.keys()))
            sample_point_cloud = self.sample_point_cloud(point_cloud[:, :3])
            centroid = np.mean(sample_point_cloud, axis=0)
            centroid_pix_x = np.floor((centroid[0] - self.min_x) / self.map_resolution).astype(int)
            centroid_pix_y = np.floor((centroid[1] - self.min_z) / self.map_resolution).astype(int)
            add_dict_one = {"class_index": class_index,
                            "pts": sample_point_cloud,
                            "center_point": np.asarray([centroid_pix_x, centroid_pix_y])}
            self.object_instance_dict[add_index] = add_dict_one

    def check_is_same_object(self, source, target):
        is_same = False
        # 遍历所有instance
        source_point_cloud = o3d.geometry.PointCloud()
        source_point_cloud.points = o3d.utility.Vector3dVector(source)
        target_point_cloud = o3d.geometry.PointCloud()
        target_point_cloud.points = o3d.utility.Vector3dVector(target)

        # 计算bbox
        source_bbox = source_point_cloud.get_axis_aligned_bounding_box()
        target_bbox = target_point_cloud.get_axis_aligned_bounding_box()

        min_corner = np.maximum(source_bbox.min_bound, target_bbox.min_bound)
        max_corner = np.minimum(source_bbox.max_bound, target_bbox.max_bound)
        intersection = np.maximum(0.0, max_corner - min_corner)
        intersection_volume = np.prod(intersection)

        # 计算两个边界框的并集体积
        volume1 = np.prod(source_bbox.get_extent())
        volume2 = np.prod(target_bbox.get_extent())
        union_volume = volume1 + volume2 - intersection_volume

        # 避免分母为零的情况
        if union_volume == 0:
            return is_same

        iou = intersection_volume / union_volume

        if iou > self.args.object_iou_threshold:
            is_same = True

        return is_same

    def merge_instance_object_point(self, source, target, num_samples=2048):
        new_instance = np.concatenate((source, target), axis=0)

        # 进行降采样
        # 获取点云的大小
        N, _ = new_instance.shape
        if N >= num_samples:
            # 当点云大小大于或等于 2048 时，随机选择 2048 个点
            indices = torch.randperm(N)[:num_samples]
        else:
            # 当点云大小小于 2048 时，带替换的随机采样
            indices = torch.randint(0, N, (num_samples,), dtype=torch.long)
        # 根据索引采样点云
        sampled_point_cloud = new_instance[indices, :]

        return sampled_point_cloud

    def sample_point_cloud(self, point_cloud, num_samples=2048):
        N, _ = point_cloud.shape
        if N >= num_samples:
            # 当点云大小大于或等于 2048 时，随机选择 2048 个点
            indices = torch.randperm(N)[:num_samples]
        else:
            # 当点云大小小于 2048 时，带替换的随机采样
            indices = torch.randint(0, N, (num_samples,), dtype=torch.long)
        # 根据索引采样点云
        sampled_point_cloud = point_cloud[indices, :]
        return sampled_point_cloud

    def get_nav_inter_feature(self, input_dict):
        object_position = input_dict["object_position"]
        object_name = input_dict["object_name"]
        # world point --> pixel point
        object_position_pix_x = np.floor((object_position[0] - self.min_x) / self.map_resolution).astype(int)
        object_position_pix_y = np.floor((object_position[1] - self.min_z) / self.map_resolution).astype(int)
        # 根据region size生成候选区域
        inter_region_size = self.args.inter_region_size
        x1 = object_position_pix_x - inter_region_size / 2
        y1 = object_position_pix_y - inter_region_size / 2
        x2 = object_position_pix_x + inter_region_size / 2
        y2 = object_position_pix_y + inter_region_size / 2
        region_x1x2 = np.asarray([x1, x2]).astype(int)
        region_y1y2 = np.asarray([y1, y2]).astype(int)
        region_x1x2 = np.clip(region_x1x2, 0, self.heightmap_size[1])
        region_y1y2 = np.clip(region_y1y2, 0, self.heightmap_size[0])

        # crop region and select 256 tokens
        object_neighbor_feature = self.global_sem_feature_map[int(region_y1y2[0]):int(region_y1y2[1]), int(region_x1x2[0]):int(region_x1x2[1])].reshape([-1, 1024])
        if self.args.use_3D_feature:
            selected_indices = torch.randperm(object_neighbor_feature.shape[0])[:246]
            select_feature_token = object_neighbor_feature[selected_indices, :]

            # 根据物体位置确定对应点云
            target_position = np.asarray([object_position_pix_x, object_position_pix_y])
            target_index = self.total_cat2idx[object_name]
            distance_list = []
            for instance_index, object_dict in self.object_instance_dict.items():
                object_position_one = object_dict["center_point"]
                class_index_one = object_dict["class_index"]
                if class_index_one == target_index:
                    distance = np.sqrt((target_position[0] - object_position_one[0]) ** 2 + \
                                       (target_position[-1] - object_position_one[-1]) ** 2)
                    distance_list.append(distance)
                else:
                    distance_list.append(10000)
            nearest_index = np.argmin(np.asarray(distance_list))
            object_pts = self.object_instance_dict[nearest_index]["pts"]

            # 平移归一化处理, 提取点云特征
            object_pts = self.shift_norm_point(object_pts)
            object_pts = torch.from_numpy(object_pts).to(self.point_clip.device).float()
            object_pts = object_pts.view([1, -1, 3])
            point_clip_feature = self.point_clip.extract_point_feature(object_pts)
            fusion_feature = torch.cat((select_feature_token, point_clip_feature), dim=0)
        else:
            selected_indices = torch.randperm(object_neighbor_feature.shape[0])[:256]
            fusion_feature = object_neighbor_feature[selected_indices, :]
        # print(fusion_feature.shape)
        # 进一步生成候选point
        surround_point_list = []
        bias_mat = np.asarray([[0.5, 0],
                               [0, 0.5],
                               [-0.5, 0],
                               [0, -0.5]])
        object_position = np.asarray(object_position)
        for i in range(4):
            surround_point_list.append(object_position + bias_mat[i])

        response_dict = {"fusion_feature": fusion_feature,
                         "surround_point_list": surround_point_list}
        return response_dict

    def shift_norm_point(self, point_cloud):
        # 计算点云的中心
        center = np.mean(point_cloud, axis=0)

        # 将点云平移到以原点为中心
        centered_point_cloud = point_cloud - center

        # 计算每个点的欧几里得范数
        norms = np.linalg.norm(centered_point_cloud, axis=1)

        # 避免除以零的情况
        norms[norms == 0] = 1

        # 将每个点除以其范数进行归一化
        normalized_point_cloud = centered_point_cloud / norms[:, np.newaxis]
        return normalized_point_cloud





