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
    num_sem_categories: int = 93
    map_size_cm: int = 10
    map_resolution: float = 0.02
