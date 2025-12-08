import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch.twins_svt import TwinsSVT
import numpy as np
from torchvision import transforms

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class SemanticMapTokenizer(nn.Module):
    '''
    Input: Semantic map, [500,500] tensor
    Output: Tokens, [1,token_length] tensor
    '''
    def __init__(self,
                 vision_tower,
                 args,
                 delay_load=False,
                 token_length=768,
                 map_size=[500, 500]):
        super().__init__()
        self.global_map_vit = TwinsSVT(
            num_classes = token_length,      # number of output classes
            s1_emb_dim = 64,          # stage 1 - patch embedding projected dimension
            s1_patch_size = 4,        # stage 1 - patch size for patch embedding
            s1_local_patch_size = 7,  # stage 1 - patch size for local attention
            s1_global_k = 7,          # stage 1 - global attention key / value      reduction factor, defaults to 7 as specified in paper
            s1_depth = 1,             # stage 1 - number of transformer blocks (local       attn -> ff -> global attn -> ff)
            s2_emb_dim = 128,         # stage 2 (same as above)
            s2_patch_size = 2,
            s2_local_patch_size = 7,
            s2_global_k = 7,
            s2_depth = 1,
            s3_emb_dim = 256,         # stage 3 (same as above)
            s3_patch_size = 2,
            s3_local_patch_size = 7,
            s3_global_k = 7,
            s3_depth = 5,
            s4_emb_dim = 512,         # stage 4 (same as above)
            s4_patch_size = 2,
            s4_local_patch_size = 7,
            s4_global_k = 7,
            s4_depth = 4,
            peg_kernel_size = 3,      # positional encoding generator kernel size
            dropout = 0.              # dropout
            )
        # self.global_map_vit = global_map_vit
        # 增加具体的属性
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)


        self.data_composer = transforms.Compose(
    [
        transforms.Resize(448),
        transforms.ToTensor(),
    ]
)
        self.map_size = map_size

    def forward(self, sem_map):
        sem_map = sem_map / 100.0
        sem_map_composed = transforms.Resize((448, 448))(sem_map.unsqueeze(0))
        zooms = [1.0]
        preds = []
        for zoom in zooms:
            resized_sem_map = F.interpolate(sem_map_composed.unsqueeze(0), scale_factor=(1, zoom, zoom), mode='nearest').squeeze(0)
            # pred = self.global_map_vit(resized_sem_map.unsqueeze(1))
            # print('resized_sem_map.dtype: ', resized_sem_map.dtype)
            resized_sem_map = resized_sem_map.to(torch.bfloat16)
            pred = self.global_map_vit(resized_sem_map)
            preds.append(pred)
        x = torch.cat(preds, dim=1)

        return x

    def load_model(self):
        self.is_loaded = True
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

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
