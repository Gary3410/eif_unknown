"""
使用规划GT、巡航GT
仅用来过滤生成数据集
"""
import numpy as np
from prior import LazyJsonDataset
import gzip
import prior
import json
from tqdm import tqdm
from env import ThorEnvCode
from procthor_config import Config as proc_Config


# 加载具体场景
def load_dataset() -> prior.DatasetDict:
    """Load the houses dataset."""
    print(
        "[AI2-THOR WARNING] There has been an update to ProcTHOR-10K that must be used with AI2-THOR version 5.0+. To use the new version of ProcTHOR-10K, please update AI2-THOR to version 5.0+ by running:\n"
        "    pip install --upgrade ai2thor\n"
        "Alternatively, to downgrade to the old version of ProcTHOR-10K, run:\n"
        '   prior.load_dataset("procthor-10k", revision="ab3cacd0fc17754d4c080a3fd50b18395fae8647")'
    )
    data = {}
    for split, size in [("train", 10_000), ("val", 1_000), ("test", 1_000)]:
        with gzip.open(f"{split}.jsonl.gz", "r") as f:
            houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
        data[split] = LazyJsonDataset(
            data=houses, dataset="procthor-dataset", split=split
        )
    return prior.DatasetDict(**data)


def perspective_camera_view(controller):
    # 默认视角为0.0
    normal_view_angle = 0.0
    # 先获取当前视角的角度
    cameraHorizon = controller.last_event.metadata["agent"]["cameraHorizon"]
    if int(cameraHorizon) - normal_view_angle < 0:
        # 需要进行look_down
        event = controller.step(action="LookDown", degrees=normal_view_angle - int(cameraHorizon))
    elif int(cameraHorizon) - normal_view_angle > 0:
        # 需要进行look_up
        event = controller.step(action="LookUp", degrees=int(cameraHorizon) - normal_view_angle)
    else:
        normal_view_angle = 30


def main():
    val_file_path = "./data/spaced_parse_instruction_easy_v8_train.json"
    instruction_dict_list = json.load(open(val_file_path))
    # 加载环境
    args = proc_Config
    controller_env = ThorEnvCode(args, use_CloudRendering=True)
    # dataset = prior.load_dataset("procthor-10k")
    dataset = load_dataset()
    dataset_train = dataset["train"]
    rgb_frame, depth_frame = controller_env.load_scene(house_name=dataset_train[0])

    # 执行动作xxx
    for _ in range(10):
        rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
        controller_env.to_thor_api_exec("RotateLeft", smooth_nav=True)


if __name__ == "__main__":
    main()





