import os
from .poni_encoder import Potential_Function_Semantic_Policy

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    # pf_model_path = "/mnt/data-2/data/yh/LLaVA_3D/checkpoints/poni/poni_seed_123_Gibson.pt"
    # pf_model_path = "/mnt/data-2/data/yh/LLaVA_3D/checkpoints/poni/poni_seed_123_MP3D.pt"
    pf_model_path = vision_tower_cfg.poni_pretrain_model
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return Potential_Function_Semantic_Policy(pf_model_path, vision_tower, args=vision_tower_cfg, **kwargs)