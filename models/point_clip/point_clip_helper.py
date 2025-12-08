import os
import torch
import argparse
from dassl.engine import build_trainer
from dassl.config import get_cfg_default
from dassl.utils import setup_logger, set_random_seed, collect_env_info

import models.point_clip.datasets.scanobjnn
import models.point_clip.datasets.modelnet40
from models.point_clip.trainers import best_param

from models.point_clip.trainers import zeroshot
from models.point_clip.trainers.post_search import search_weights_zs, search_prompt_zs
from dataclasses import dataclass


@dataclass
class PontClipConfig:
    output_dir: str = "/home/wzy/workplace/llava_procthor/models/point_clip/outputs"
    seed: int = 2
    transforms: str = ""
    config_file: str = "/home/wzy/workplace/llava_procthor/models/point_clip/configs/trainers/PointCLIPV2_ZS/rn50.yaml"
    dataset_config_file: str = "/home/wzy/workplace/llava_procthor/models/point_clip/configs/datasets/modelnet40.yaml"
    trainer: str = "PointCLIPV2_ZS"
    backbone: str = ""
    head: str = ""
    zero_shot: bool = True
    post_search: bool = True
    model_dir: str = ""
    load_epoch: int = 175
    no_train: bool = True


def reset_cfg(cfg, args):
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.seed:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN
    cfg.TRAINER.EXTRA = CN()


def setup_cfg(args, procthor_config=None):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    if procthor_config is not None:
        cfg.GPU = "cuda:" + str(procthor_config.sem_feature_map_gpu)

    # 4. From optional input arguments
    # cfg.merge_from_list(args.opts)
    cfg.merge_from_list([])
    cfg.freeze()
    return cfg


class PointClipHelper:
    def __init__(self, procthor_config=None):
        args = PontClipConfig
        cfg = setup_cfg(args, procthor_config)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            torch.backends.cudnn.benchmark = True

        self.device = cfg.GPU
        # print_args(args, cfg)
        print('Collecting env info ...')
        print('** System info **\n{}\n'.format(collect_env_info()))

        trainer = build_trainer(cfg)
        self.model = trainer

    def model_inference(self, pc):
        output = self.model.model_inference(pc, label=None)
        return output

    def extract_point_feature(self, pc):
        output = self.model.extract_point_feature(pc)
        # [10, 1024] --> [1, 1024]
        # return output[0:1, :]
        return output
