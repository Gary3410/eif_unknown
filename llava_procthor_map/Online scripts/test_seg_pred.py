import os
import numpy as np
import json
from tqdm import tqdm
from utils.sem_map import Semantic_Mapping
from utils.thor_env_code_v2 import ThorEnvCode
from utils.procthor_config import Config as proc_Config
import torch
import open_clip
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from einops import asnumpy, repeat
from PIL import Image, ImageFont, ImageDraw
import random
from tqdm import tqdm
import prior
from prior import LazyJsonDataset
import gzip


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
        with gzip.open(f"./procthor_house/{split}.jsonl.gz", "r") as f:
            houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
        data[split] = LazyJsonDataset(
            data=houses, dataset="procthor-dataset", split=split
        )
    return prior.DatasetDict(**data)


def perspective_camera_view(controller):
    # 默认视角为30
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


def visualize_map(semmap, bg=1.0, dataset='gibson'):
    n_cat = semmap.shape[0] - 2  # Exclude floor and wall

    # 加载调色盘
    # color_map = np.load("D:\\python_code\\prompter_procthor\\utils\\color_map.npy")
    color_map = np.load("./utils/color_map.npy")

    def compress_semmap(semmap):
        c_map = np.zeros((semmap.shape[1], semmap.shape[2]))
        for i in range(semmap.shape[0]):
            c_map[semmap[i] > 0.] = i + 1
        return c_map

    palette = [
        int(bg * 255), int(bg * 255), int(bg * 255),  # Out of bounds
        230, 230, 230,  # Free space
        77, 77, 77,  # Obstacles
    ]

    palette += [c for color in color_map[:n_cat]
                for c in color.tolist()]

    semmap = asnumpy(semmap)
    c_map = compress_semmap(semmap)
    semantic_img = Image.new("P", (c_map.shape[1], c_map.shape[0]))
    semantic_img.putpalette(palette)
    semantic_img.putdata((c_map.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = np.array(semantic_img)

    return semantic_img


def visualize_frontiers_mask(vis_semmap, frontiers_mask, dset='gibson', color_index=255, channel_index=0):
    frontiers_mask = asnumpy(frontiers_mask)[0]  # (H, W)
    frontiers_mask = frontiers_mask[..., np.newaxis]  # (H, W)

    # print(frontiers_mask.shape)

    red_image = np.zeros_like(vis_semmap)
    # red_image[..., 0] = color_index
    red_image[..., channel_index] = color_index
    smpf = red_image * frontiers_mask + vis_semmap * (1 - frontiers_mask)
    smpf = smpf.astype(np.uint8)

    return smpf


def save_step_image(rgb_base_path, frontiers_base_path, mask_base_path, rgb_image, semantic_map, frontiers_dict=None, info_dict=None, id=0):
    rgb_save_path = os.path.join(rgb_base_path, str(id) + ".png")
    mask_save_path = os.path.join(mask_base_path, str(id) + ".png")
    height_map_save_path = os.path.join(height_map_base_save_path, str(id) + ".npy")
    frontiers_save_path = os.path.join(frontiers_base_path, str(id) + ".png")
    save_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    mask_map_save_path = os.path.join(mask_map_base_save_path, str(id) + ".npy")
    # np.save(mask_map_save_path, semantic_map)

    vis_mask = visualize_map(semantic_map)
    if frontiers_dict is not None:
        object_mask = frontiers_dict["object_mask"]
        robot_mask = frontiers_dict["robot_mask"]
        vis_frontiers_mask = frontiers_dict["frontiers_mask"]
        nav_path_mask = frontiers_dict["nav_path_mask"]
        vis_mask = visualize_frontiers_mask(vis_mask, vis_frontiers_mask)
        vis_mask = visualize_frontiers_mask(vis_mask, object_mask, color_index=255, channel_index=2)
        vis_mask = visualize_frontiers_mask(vis_mask, nav_path_mask, color_index=180)
        vis_mask = visualize_frontiers_mask(vis_mask, robot_mask, color_index=255, channel_index=1)
    plt.figure(figsize=(40, 20))
    plt.imshow(vis_mask)
    # plt.savefig(frontiers_save_path, dpi=800)
    plt.savefig(frontiers_save_path)
    plt.close()
    cv2.imwrite(rgb_save_path, save_rgb)
    if "vis_mask" in info_dict.keys():
        sem_vis = info_dict["vis_mask"]
        sem_vis = cv2.resize(sem_vis, (500, 500))
        cv2.imwrite(mask_save_path, sem_vis)
    height_map_save = info_dict["height_map"]
    np.save(height_map_save_path, height_map_save)


def nav_location(controller_env, target_nav_position, sem_map, save_id):
    # controller.perspective_camera_view(reset_view_angle=45)
    nav_action, path = controller_env.parse_nav_action(target_nav_position)
    path_distance = (len(path) - 1) * 0.25
    for nav_action_id, action_one in enumerate(tqdm(nav_action)):
        rgb, depth_frame, mask_list, info_dict = controller_env.to_thor_api_exec([action_one])
        if nav_action_id % 1 == 0:
            info_dict["add_robot_mask"] = False
            semantic_map, _ = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
        info_dict["height_map"] = sem_map.height_map
        robot_position = info_dict["camera_world_xyz"]
        robot_position_mask = sem_map.get_nav_position_mask([robot_position])
        robot_position_mask = sem_map.dilate_frontiers_mask(robot_position_mask, kernel_size=5, padding=2)
        nav_path_list.append(robot_position)
        nav_path_mask = sem_map.get_nav_position_mask(nav_path_list)
        nav_path_mask = sem_map.dilate_frontiers_mask(nav_path_mask, kernel_size=5, padding=2)

        vis_dict = dict(
            object_mask=object_mask,
            robot_mask=robot_position_mask,
            frontiers_mask=sem_map.vis_frontiers_mask,
            nav_path_mask=nav_path_mask)

        save_step_image(rgb_base_save_path, frontiers_base_save_path, vis_mask_base_save_path, rgb, semantic_map, vis_dict, info_dict, id=save_id)
        save_id = save_id + 1
    return path_distance, save_id

# 配置仿真场景
kitchens_val = [f"FloorPlan{i}" for i in range(21, 26)]
living_rooms_val = [f"FloorPlan{200 + i}" for i in range(21, 26)]
bedrooms_val = [f"FloorPlan{300 + i}" for i in range(21, 26)]
bathrooms_val = [f"FloorPlan{400 + i}" for i in range(21, 26)]

open_clip_model = "ViT-H-14"
torch.autograd.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Initializing model...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    open_clip_model, pretrained="./checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
)
clip_model.cuda("cuda:0")
clip_model.eval()

# room_list = kitchens_val[:2] + living_rooms_val[:2] + bathrooms_val[:2] + bedrooms_val[:2]
# room_list = [living_rooms_val[0]]

# dataset = load_dataset()
dataset = prior.load_dataset("procthor-10k")
dataset_train = dataset["train"]
room_list = [dataset_train[2271], dataset_train[2253], dataset_train[2260]]
# room_position_list = []
args = proc_Config
# min_position_list = []
# max_position_list = []
controller = ThorEnvCode(args, use_CloudRendering=True)
# gb_frame, depth_frame = controller.load_scene(house_name=living_rooms_val[0])
# for room_one in tqdm(room_list):
#     rgb_frame, depth_frame = controller.load_scene(house_name=room_one)
#     position_tuple = controller.position_tuple
#     position_tuple = np.asarray(position_tuple)
#     print("=====================================")
#     min_position = np.min(position_tuple, axis=0)
#     max_position = np.max(position_tuple, axis=0)
#     room_h = max_position[0] - min_position[0]
#     room_w = max_position[-1] - min_position[-1]
#     min_position_list.append(min_position)
#     max_position_list.append(max_position)
#     print("min: ", np.min(position_tuple, axis=0))
#     print("max:, ", np.max(position_tuple, axis=0))
#     print("room size: ", [room_h, room_w])
#     print("=====================================")
# min_position_list = np.asarray(min_position_list)
# max_position_list = np.asarray(max_position_list)
# print("min: ", np.min(min_position_list, axis=0))
# print("max: ", np.max(max_position_list, axis=0))
# exit()

if proc_Config.alfred_scene:
    total_cat2idx = json.load(open(proc_Config.total_cat2idx_alfred_path))
else:
    if proc_Config.use_sem_seg:
        total_cat2idx = json.load(open(proc_Config.total_cat2idx_procthor_path))
    else:
        total_cat2idx = json.load(open(proc_Config.total_cat2idx_procthor_path))
        # 增加wall
        total_cat2idx["wall"] = 94
        total_cat2idx["floor"] = 95

sem_map = Semantic_Mapping(
    args=proc_Config,
    clip_model=clip_model,
    clip_preprocess=clip_preprocess,
    total_cat2idx=total_cat2idx
)
base_save_path = "./visualization/visualization_results_test_prediction_v2"
# 开始环视


for room_index, room_one in enumerate(tqdm(room_list)):
    room_name = "house_" + str(room_index)
    rgb_base_save_path = os.path.join(base_save_path, room_name + "_" + "rgb")
    frontiers_base_save_path = os.path.join(base_save_path, room_name + "_" + "frontiers")
    vis_mask_base_save_path = os.path.join(base_save_path, room_name + "_" + "masks")
    height_map_base_save_path = os.path.join(base_save_path, room_name + "_" + "height_map")
    mask_map_base_save_path = os.path.join(base_save_path, room_name + "_" + "mask_map")

    if not os.path.exists(rgb_base_save_path):
        os.makedirs(rgb_base_save_path)
    if not os.path.exists(frontiers_base_save_path):
        os.makedirs(frontiers_base_save_path)
    if not os.path.exists(vis_mask_base_save_path):
        os.makedirs(vis_mask_base_save_path)
    if not os.path.exists(height_map_base_save_path):
        os.makedirs(height_map_base_save_path)
    if not os.path.exists(mask_map_base_save_path):
        os.makedirs(mask_map_base_save_path)

    save_id = 0
    gb_frame, depth_frame = controller.load_scene(house_name=room_one)
    sem_map.reset()

    nav_path_list = []
    position_tuple = controller.position_tuple
    # print(position_tuple)
    # exit()
    for _ in range(5):
        # 进行环视
        if proc_Config.alfred_scene:
            controller.perspective_camera_view(reset_view_angle=45)
        else:
            controller.perspective_camera_view(reset_view_angle=30)
        for _ in range(4):
            # 增加转圈
            rgb, depth_frame, mask_list, info_dict = controller.get_obs()
            info_dict["add_robot_mask"] = True
            info_dict["region_size"] = 0.20
            global_sem_map, global_sem_feature_map = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
            info_dict["height_map"] = sem_map.height_map
            save_step_image(rgb_base_save_path, frontiers_base_save_path, vis_mask_base_save_path, rgb, global_sem_map, info_dict=info_dict, id=save_id)
            save_id = save_id + 1
            event = controller.step(action="RotateRight", forceAction=True)
        print("end surrounding ----------------------")
        rgb, depth_frame, mask_list, info_dict = controller.get_obs()
        select_feature_list, frontiers_label_dict_list = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="frontiers")
        target_nav_position_tuple = random.choice(position_tuple)
        target_nav_position = [target_nav_position_tuple[0], target_nav_position_tuple[-1]]

        object_mask = sem_map.get_nav_position_mask([target_nav_position])
        object_mask = sem_map.dilate_frontiers_mask(object_mask, kernel_size=5, padding=2)
        if not proc_Config.alfred_scene:
            controller.perspective_camera_view()
        # 机械巡航路径
        path_distance, save_id = nav_location(controller, target_nav_position, sem_map, save_id)

        # 输出观察物体列表
        object_name_list = controller.seen_object_name_alfred()
        print("seen object name: ", object_name_list)
        print("--------------------------------------")
        map_object_name_list = sem_map.get_object_name_list_from_map()
        print("map object name: ", map_object_name_list)

        unseen_object_name_list = []
        for object_name_one in object_name_list:
            if object_name_one not in map_object_name_list:
                unseen_object_name_list.append(object_name_one)

        print("未记录物体", unseen_object_name_list)









