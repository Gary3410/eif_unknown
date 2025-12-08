import argparse
import collections
import logging
import math
import os
import random
import time

import numpy as np

import torch
import torch.distributed as distrib
import torch.nn.functional as F
import tqdm
from .default import get_cfg
from .poni_model_utils import get_semantic_encoder_decoder

from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

class SemanticMapperModule_poni(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Define models
        ndirs = None
        self.dirs_map = None
        self.inv_dists_map = None

        object_loss_type = self.cfg.MODEL.object_loss_type

        enable_directions = self.cfg.DATASET.enable_directions
        ndirs = len(self.cfg.DATASET.prediction_directions)
        enable_locations = self.cfg.DATASET.enable_locations
        enable_actions = self.cfg.DATASET.enable_actions
        assert not (enable_locations and enable_directions)
        assert not (enable_actions and enable_directions)
        assert not (enable_actions and enable_locations)
        enable_area_head = self.cfg.DATASET.enable_unexp_area

        if enable_directions:
            assert object_loss_type == "xent"
            assert self.cfg.MODEL.object_activation == "none"
            self.cfg.defrost()
            self.cfg.MODEL.output_type = "dirs"
            self.cfg.MODEL.ndirs = ndirs
            self.cfg.freeze()
        if enable_locations:
            assert object_loss_type in ["l1", "l2"]
            assert self.cfg.MODEL.object_activation == "sigmoid"
            self.cfg.defrost()
            self.cfg.MODEL.output_type = "locs"
            self.cfg.freeze()
        if enable_actions:
            assert object_loss_type == "xent"
            assert self.cfg.MODEL.object_activation == "none"
            self.cfg.defrost()
            self.cfg.MODEL.output_type = "acts"
            self.cfg.freeze()
        if enable_area_head:
            self.cfg.defrost()
            self.cfg.MODEL.enable_area_head = enable_area_head
            self.cfg.freeze()

        (
            self.encoder,
            self.object_decoder,
            self.area_decoder,
        ) = get_semantic_encoder_decoder(self.cfg)

        print("初始化encoder 完成")

    def forward(self, x, encoder_only=True):
        if encoder_only:
            embedding = self.encoder(x)
            return embedding
        else:
            embedding = self.encoder(x)
            object_preds = self.object_activation(self.object_decoder(embedding))
            area_preds = None
            if self.area_decoder is not None:
                area_preds = self.area_activation(self.area_decoder(embedding))
            return object_preds, area_preds

    def infer(self, x, encoder_only=True):
        if encoder_only:
            embedding = self(x)
            return embedding
        else:
            object_preds, area_preds = self(x)
            return object_preds, area_preds

    def convert_to_distributed_data_parallel(self, rank):
        if self.cfg.LOGGING.verbose:
            print(f"=======> (0.3) breakpoint reached in local proc: {rank}")
        encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        object_decoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.object_decoder)
        if self.area_decoder is not None:
            area_decoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.area_decoder)
        if self.cfg.LOGGING.verbose:
            print(f"=======> (0.4) breakpoint reached in local proc: {rank}")
        self.encoder = DDP(encoder, device_ids=[rank])
        self.object_decoder = DDP(object_decoder, device_ids=[rank])
        if self.area_decoder is not None:
            self.area_decoder = DDP(area_decoder, device_ids=[rank])