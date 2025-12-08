import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from .poni.default import get_cfg
from .poni.poni_model import SemanticMapperModule_poni as PFModel
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import yaml
from deepspeed import zero

class Potential_Function_Semantic_Policy(nn.Module):
    def __init__(self,
                 pf_model_path,
                 vision_tower,
                 args,
                 delay_load=False
                 ):
        super().__init__()
        # loaded_state = torch.load(pf_model_ckpt, map_location="cpu")
        self.pf_model_path = pf_model_path

        yaml_file = pf_model_path.replace("pt", "yaml")
        pf_model_cfg = get_cfg(yaml_file)
        # pf_model_cfg.merge_from_other_cfg(loaded_state["cfg"])
        # pf_model_cfg.merge_from_file(yaml_file)
        # 读取yaml文件
        # with open(yaml_file, 'r') as file:
        #     pf_model_cfg = yaml.safe_load(file)
        #
        # print(pf_model_cfg)
        self.pf_model = PFModel(pf_model_cfg)
        # Remove dataparallel modules
        # state_dict = {
        #     k.replace(".module", ""): v for k, v in loaded_state["state_dict"].items()
        # }
        # self.eval()

        # 增加具体属性
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)


    def forward(self, inputs, encoder_only=True):
        # inputs - (bs, N, H, W)
        # x_pf - (bs, N, H, W), x_a - (bs, 1, H, W)
        if encoder_only:
            embedding = self.pf_model.infer(inputs, encoder_only=encoder_only)
            return embedding
        else:
            x_pf, x_a = self.pf_model.infer(inputs, encoder_only=encoder_only)
            return x_pf, x_a

    def add_agent_dists_to_object_dists(self, pfs, agent_dists):
        # pfs - (B, N, H, W)
        # agent_dists - (B, H, W)
        object_dists = self.convert_object_pf_to_distance(pfs)
        agent2obj_dists = agent_dists.unsqueeze(1) + object_dists
        # Convert back to pf
        return self.pf_model.convert_distance_to_pf(agent2obj_dists)

    def convert_object_pf_to_distance(self, pfs):
        return self.pf_model.convert_object_pf_to_distance(pfs)

    def convert_distance_to_pf(self, dists):
        return self.pf_model.convert_distance_to_pf(dists)

    @property
    def cfg(self):
        return self.pf_model.cfg

    def load_model(self):
        self.is_loaded = True
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        model_wight = torch.load(self.pf_model_path, map_location="cpu")
        print("poni load weight------------")
        state_dict = {
            k.replace(".module", ""): v for k, v in model_wight.items()
        }
        with zero.GatheredParameters(self.pf_model.parameters()):
            self.pf_model.load_state_dict(state_dict)
            self.pf_model.requires_grad_(False)

    def feature_select(self, image_forward_outs):
        pass

    @property
    def device(self):
        return self.global_map_vit.device

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
