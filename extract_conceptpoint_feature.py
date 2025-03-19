import os
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d
import open_clip
import torch
from PIL import Image
from tqdm import tqdm, trange
from typing_extensions import Literal
import argparse
import pickle
import blosc
import json
from utils.procthor_config import Config as proc_Config
from utils.sem_map import Semantic_Mapping
import torch.nn.functional as F
import random

# add point lip
from models.point_clip.point_clip_helper import PointClipHelper

def get_bbox_around_mask_cpu(mask):
    bbox = None
    nonzero_inds = np.nonzero(mask)
    if nonzero_inds[0].shape[0] == 0:
        topleft = [0, 0]
        botright = [mask.shape[0], mask.shape[1]]
        bbox = (topleft[0], topleft[1], botright[0], botright[1])  # (x0, y0, x1, y1)
    else:
        topleft_x = nonzero_inds[0].min()  # H
        topleft_y = nonzero_inds[1].min()  # H
        botright_x = nonzero_inds[0].max()  # W
        botright_y = nonzero_inds[0].max()  # W
        bbox = (topleft_x, topleft_y, botright_x, botright_y)

    return bbox


def get_house_name(rgb_name_list):
    house_id_list = [x.split("_")[1] for x in rgb_name_list]
    house_id_list = sorted(list(set(house_id_list)))
    house_name_list = ["house_"+str(x) for x in house_id_list]
    return house_name_list


def get_house_object_list(rgb_name_list, house_name_list):
    house_id2object_dict = {}
    for house_name_one in house_name_list:
        house_id2object_dict[house_name_one] = []
        for rgb_name_one in rgb_name_list:
            rgb_house_name = "house_" + rgb_name_one.split("_")[1]
            if rgb_house_name == house_name_one:
                house_id2object_dict[house_name_one].append(rgb_name_one)
    return house_id2object_dict


def get_object_group(house_one_object_list):
    house_one_object_dict = {}
    for house_object_one in house_one_object_list:
        object_index = int(house_object_one.split("_")[2])
        if object_index not in house_one_object_dict:
            house_one_object_dict[object_index] = []
            house_one_object_dict[object_index].append(house_object_one)
        else:
            house_one_object_dict[object_index].append(house_object_one)
    return house_one_object_dict


def save_blosc_file(path, sem_feature):
    sem_feature = sem_feature.astype(np.float16)
    compressed_array = blosc.compress(sem_feature.tobytes())
    # Save the compressed data to a file
    with open(path, 'wb') as f:
        f.write(compressed_array)


def shift_norm_point(point_cloud):
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


def sample_point_cloud(point_cloud, num_samples=2048):
    # 获取点云的大小
    N, _ = point_cloud.shape

    if N >= num_samples:
        # 当点云大小大于或等于 2048 时，随机选择 2048 个点
        indices = torch.randperm(N)[:num_samples]
    else:
        # 当点云大小小于 2048 时，带替换的随机采样
        indices = torch.randint(0, N, (num_samples,), dtype=torch.long)

    # 根据索引采样点云
    sampled_point_cloud = point_cloud[indices, :]

    return sampled_point_cloud, indices


def main(args):
    # CLIP model config
    open_clip_model = "ViT-H-14"
    open_clip_pretrained_dataset = "laion2b_s32b_b79k"

    # point clip
    point_clip = PointClipHelper()
    # 测试trainner
    # 随机生成2000x3的点云
    # point_cloud = torch.randn(1, 20000, 3)
    # # 将点云数据移动到GPU
    # point_cloud = point_cloud.to(point_clip.device)
    #
    # output = point_clip.extract_point_feature(point_cloud)
    # print(output.shape)
    # exit()

    torch.autograd.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Initializing model...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        open_clip_model, pretrained="./checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
    )
    model.cuda()
    model.eval()

    total_cat2idx_path = proc_Config.total_cat2idx_procthor_path
    total_cat2idx = json.load(open(total_cat2idx_path))

    sem_map = Semantic_Mapping(
        args=proc_Config,
        clip_model=model,
        clip_preprocess=preprocess,
        total_cat2idx=total_cat2idx
    )
    sem_map.reset()

    # 配置基本路径
    mask_base_path = args.mask_dir_path
    rgb_base_path = args.rgb_dir_path
    depth_base_path = args.depth_dir_path
    info_dict_base_path = args.info_dict_dir_path
    feature_base_save_path = args.feature_save_path
    point_clip_feature_base_save_path = "./vision_dataset/nav_inter_dataset/point_clip_feature/"
    label_base_save_path = "./vision_dataset/nav_inter_dataset/label/"

    if not os.path.exists(feature_base_save_path):
        os.makedirs(feature_base_save_path)
    if not os.path.exists(label_base_save_path):
        os.makedirs(label_base_save_path)
    if not os.path.exists(point_clip_feature_base_save_path):
        os.makedirs(point_clip_feature_base_save_path)

    print("Computing pixel-aligned features...")

    # 遍历文件夹, 获取整体mask
    rgb_name_list = sorted(os.listdir(rgb_base_path))
    # 获取每个房间的物体列表
    house_name_list = get_house_name(rgb_name_list)
    house_id2object_dict = get_house_object_list(rgb_name_list, house_name_list)

    # main loop ---------------------
    for house_name, house_one_object_list in tqdm(house_id2object_dict.items()):
        object_group = get_object_group(house_one_object_list)
        for object_index, object_name_list in object_group.items():
            # 开始进行读取image
            object_feature_save_name = house_name + "_" + str(object_index)
            object_pts_list = []
            object_pts_feature_list = []
            choice_list = []
            label_list = []
            for rgb_name_one in object_name_list:
                rgb_path_one = os.path.join(rgb_base_path, rgb_name_one)
                mask_path_one = os.path.join(mask_base_path, rgb_name_one.replace(".png", ".npz"))
                info_dict_path_one = os.path.join(info_dict_base_path, rgb_name_one.replace(".png", ".pkl"))
                depth_path_one = os.path.join(depth_base_path, rgb_name_one.replace(".png", ".npz"))
                with open(info_dict_path_one, "rb") as f:
                    info_dict_one = pickle.load(f)
                is_find = info_dict_one["is_find"]
                agent_position = info_dict_one["agent_position"]
                if is_find:
                    label_list.append(1)
                else:
                    label_list.append(0)
                choice_list.append(agent_position)

                img = cv2.imread(rgb_path_one)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                masks = np.load(mask_path_one)["mask"]
                try:
                    # self.open_clip_device
                    mask_feature = sem_map.extract_feature(img, masks)
                    mask_feature = mask_feature.view([-1, 1024])
                except:
                    break
                # 增加点云特征映射
                depth_image = np.load(depth_path_one)["depth_image"]
                world_space_point_cloud = sem_map.get_point_cloud(depth_image, masks, info_dict_one)
                target_name = info_dict_one["target_name"]
                cat_index = total_cat2idx[target_name]
                target_point = world_space_point_cloud[world_space_point_cloud[:, 3] == cat_index]
                target_point_feature = mask_feature[world_space_point_cloud[:, 3] == cat_index]
                if target_point.shape[0] > 0:
                    object_pts_list.append(target_point)
                    object_pts_feature_list.append(target_point_feature)
            # feature提取完成后, 整体送入point_clip
            if len(object_pts_list) <= 0:
                continue
            object_pts = np.concatenate(object_pts_list, axis=0)
            if object_pts.shape[0] == 0:
                continue
            object_pts = object_pts[:, :3]
            object_pts_feature = torch.cat(object_pts_feature_list, dim=0)
            # object_pts 特征提取
            object_pts = shift_norm_point(object_pts)
            object_pts = torch.from_numpy(object_pts).to(point_clip.device).float()
            object_pts, indices = sample_point_cloud(object_pts)
            object_pts_feature = object_pts_feature[indices, :]
            object_pts = object_pts.view([1, -1, 3])
            point_clip_feature = point_clip.extract_point_feature(object_pts)
            # object_pts_feature = object_pts_feature + point_clip_feature

            feature_save_path_one = os.path.join(feature_base_save_path, object_feature_save_name + ".blosc")
            save_blosc_file(feature_save_path_one, object_pts_feature.cpu().numpy())

            point_clip_feature_save_path_one = os.path.join(point_clip_feature_base_save_path, object_feature_save_name + ".blosc")
            save_blosc_file(point_clip_feature_save_path_one, point_clip_feature.cpu().numpy())
            # exit()
            # 采样到256
            # 计算填充的大小
            # N, M = object_pts_feature.shape
            # if N < 256:
            #     pad_m = 1024 - M if M < 1024 else 0
            #     pad_n = 256 - N if N < 256 else 0
            #     # 使用 F.pad 进行填充，注意填充顺序为 (dim-1, dim-1, dim-0, dim-0)
            #     object_pts_feature = F.pad(object_pts_feature, (0, pad_m, 0, pad_n), "constant", 0)
            # else:
            #     object_pts_feature = object_pts_feature[:256, :]

            # 制作label
            # start_ascii = 65
            # assert len(choice_list) == len(label_list)
            # prompt = "I need move to " + target_name + " , which point I should go?"
            # question = "\n"
            # question_list = []
            # for choice_index, choice_one in enumerate(choice_list):
            #     # choose_str = chr(start_ascii + choice_index)
            #     choice_one = np.around(choice_one, 2)
            #     # choose_one_str = choose_str + ". " + "[" + str(choice_one[0]) + ", " + str(choice_one[1]) + "]"
            #     choose_one_str = "[" + str(choice_one[0]) + ", " + str(choice_one[1]) + "]"
            #     # question = question + choose_one_str + "\n"
            #     question_list.append(choose_one_str)
            # # question = question.rstrip("\n")
            # # human_input = prompt + question
            # if np.sum(np.asarray(label_list)) <= 0:
            #     continue
            # choice_index = label_list.index(1)
            # label = question_list[choice_index]
            # # 进行洗牌
            # random.shuffle(question_list)
            # answer_str = chr(start_ascii + 0)
            # for choice_index, question_one in enumerate(question_list):
            #     choose_str = chr(start_ascii + choice_index)
            #     if question_one == label:
            #         answer_str = choose_str
            #     question_one = choose_str + ". " + question_one
            #     question = question + question_one + "\n"
            # question = question.rstrip("\n")
            # human_input = prompt + question
            # response = "The answer is " + answer_str + ", " + "so I can see " + target_name
            # save_dict = dict(human_input=human_input, response=response)
            # save_path_one = os.path.join(label_base_save_path, object_feature_save_name + ".json")
            # with open(save_path_one, 'w') as f:
            #     json_str = json.dumps(save_dict, indent=2)
            #     f.write(json_str)
            #     f.write('\n')

            # 保存特征
            # feature_save_path_one = os.path.join(feature_base_save_path, object_feature_save_name + ".blosc")
            # save_blosc_file(feature_save_path_one, object_pts_feature.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify dirs")
    parser.add_argument("--rgb-dir-path", default="./vision_dataset/nav_inter_dataset/rgb/", type=str)
    parser.add_argument("--mask-dir-path", default="./vision_dataset/nav_inter_dataset/mask/", type=str)
    parser.add_argument("--info-dict-dir-path", default="./vision_dataset/nav_inter_dataset/info/", type=str)
    parser.add_argument("--depth-dir-path", default="./vision_dataset/nav_inter_dataset/depth/", type=str)
    # parser.add_argument("--save-dir-path", default="./nps_sam_clip/", type=str)
    parser.add_argument("--feature-save-path", default="./vision_dataset/nav_inter_dataset/vision_feature_blosc_v2/", type=str)
    args = parser.parse_args()
    main(args)