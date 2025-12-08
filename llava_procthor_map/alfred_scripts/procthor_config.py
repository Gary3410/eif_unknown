import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Type, Union

import torch
from typing_extensions import Self

@dataclass
class Config:
    agent: str = "sem_exp"
    alfred: int = 1
    frame_height: int = 150
    frame_width: int = 150
    env_frame_width: int = 600
    env_frame_height: int = 600
    env_frame_width_alfred: int = 300
    env_frame_height_alfred: int = 300
    num_sem_categories: int = 97
    num_sem_categories_alfred: int = 108
    map_size_m: int = 12  # 16
    map_size_m_alfred: int = 10
    min_x: int = 0
    min_z: int = 0
    min_x_alfred: int = -5
    min_z_alfred: int = -5
    max_x: int = 12  # 16
    max_z: int = 12  # 16
    max_x_alfred: int = 5
    max_z_alfred: int = 5
    map_resolution: float = 0.02
    du_scale: int = 1
    vision_range: int = 100
    camera_height: float = 1.576
    hfov: float = 90.0
    frontiers_thresholds: int = 150
    frontiers_thresholds_alfred: int = 50
    frontiers_distance_threshold: float = 0.75
    use_sem_seg: bool = True
    use_learned_depth: bool = False
    depth_gpu: int = 3
    sem_seg_gpu: int = 3
    map_pred_threshold: int = 65
    no_pickup_update: bool = True
    cat_pred_threshold: int = 10
    valts_depth: bool = True
    valts_trustworthy: bool = True
    valts_trustworthy_prop: float = 0.9
    valts_trustworthy_obj_prop0: float = 1.0
    valts_trustworthy_obj_prop: float = 1.0
    learned_visibility: bool = True
    learned_visibility_no_mask: bool = True
    separate_depth_for_straight: bool = True
    with_mask_above_05: bool = True
    sem_seg_threshold_small: float = 0.8
    sem_seg_threshold_large: float = 0.8
    sem_seg_threshold_procthor: float = 0.8
    alfworld_mrcnn: bool = True
    alfworld_both: bool = True
    explore_prob: float = 0.0
    total_cat2idx_procthor_path: str = "./utils/total_cat2idx.json"
    total_cat2idx_alfred_path: str = "./utils/total_cat2idx_alfred.json"
    seg_model_path: str = "models/segmentation/maskrcnn_alfworld/mrcnn_alfred_objects_008_v3.pth"
    alfred_scene: bool = True
    procthor_scene: bool = False
    cuda: bool = True
    visualize: bool = False
    save_pictures: bool = False
    min_depth: float = 0.5
    max_depth: float = 5.0
    region_size_alfred: float = 2.0
    alfred_height_threshold: float = 0.12
    remove_object_region_size: float = 1.45
    vis_demo: bool = False
    limit_vis_region: bool = False
    limit_vis_region_size: float = 5
    vis_similarity: bool = False
    fusion_weights: bool = False
    long_clip_path: str = "./checkpoints/longclip-B.pt"
    long_clip_gpu: int = 2
    sem_feature_map_cuda: bool = True
    sem_feature_map_gpu: int = 2
    pad_frontiers_token: bool = False
    frontiers_token_number: int = 32
    open_clip_gpu: int = 2


