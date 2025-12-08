import pickle
import os
import numpy as np
import os
from tqdm import tqdm
import open_clip
from utils.sem_map import Semantic_Mapping
from utils.thor_env_code_v2 import ThorEnvCode
import prior
from utils.procthor_config import Config as proc_Config
import torch
import json
from einops import asnumpy, repeat
from PIL import Image, ImageFont, ImageDraw
import cv2
import matplotlib.pyplot as plt


def perspective_camera_view(controller):
    # 默认视角为30
    normal_view_angle = 30
    # 先获取当前视角的角度
    cameraHorizon = controller.last_event.metadata["agent"]["cameraHorizon"]
    if cameraHorizon - normal_view_angle < 0:
        # 需要进行look_down
        event = controller.step(action="LookDown", degrees=30)
    elif cameraHorizon - normal_view_angle > 0:
        # 需要进行look_up
        event = controller.step(action="LookUp", degrees=30)
    else:
        normal_view_angle = 30


def visualize_map(semmap, bg=1.0, dataset='gibson'):
    n_cat = semmap.shape[0] - 2  # Exclude floor and wall

    # 加载调色盘
    # color_map = np.load("D:\\python_code\\prompter_procthor\\utils\\color_map.npy")
    color_map = np.load("/home/wzy/workplace/procthor/utils/color_map.npy")

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


def visualize_frontiers_mask(vis_semmap, frontiers_mask, dset='gibson', color_index=255):
    frontiers_mask = asnumpy(frontiers_mask)[0]  # (H, W)
    frontiers_mask = frontiers_mask[..., np.newaxis]  # (H, W)

    # print(frontiers_mask.shape)

    red_image = np.zeros_like(vis_semmap)
    red_image[..., 0] = color_index
    smpf = red_image * frontiers_mask + vis_semmap * (1 - frontiers_mask)
    smpf = smpf.astype(np.uint8)

    return smpf


def save_step_image(rgb_base_path, frontiers_base_path, rgb_image, semantic_map, frontiers_dict=None, id=0):
    rgb_save_path = os.path.join(rgb_base_path, str(id) + ".png")
    frontiers_save_path = os.path.join(frontiers_base_path, str(id) + ".png")
    save_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    vis_mask = visualize_map(semantic_map)
    if frontiers_dict is not None:
        object_mask = frontiers_dict["object_mask"]
        robot_mask = frontiers_dict["robot_mask"]
        vis_frontiers_mask = frontiers_dict["frontiers_mask"]
        nav_path_mask = frontiers_dict["nav_path_mask"]
        vis_mask = visualize_frontiers_mask(vis_mask, vis_frontiers_mask)
        vis_mask = visualize_frontiers_mask(vis_mask, object_mask, color_index=200)
        vis_mask = visualize_frontiers_mask(vis_mask, robot_mask, color_index=100)
        vis_mask = visualize_frontiers_mask(vis_mask, nav_path_mask, color_index=180)
    plt.figure(figsize=(40, 20))
    plt.imshow(vis_mask)
    # plt.savefig(vis_frontiers_mask_save_path_one, dpi=800)
    plt.savefig(frontiers_save_path)
    plt.close()

    cv2.imwrite(rgb_save_path, save_rgb)


def nav_location(controller_env, target_nav_position, sem_map, save_id):
    nav_action, path = controller_env.parse_nav_action(target_nav_position)
    path_distance = (len(path) - 1) * 0.25
    for nav_action_id, action_one in enumerate(tqdm(nav_action)):
        rgb, depth_frame, mask_list, info_dict = controller_env.to_thor_api_exec([action_one])
        # if nav_action_id % 5 == 0:
        semantic_map, _ = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
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

        save_step_image(rgb_base_save_path, frontiers_base_save_path, rgb, semantic_map, vis_dict, id=save_id)

    return path_distance


# load clip
open_clip_model = "ViT-H-14"
torch.autograd.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Initializing model...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    open_clip_model, pretrained="./checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
)
clip_model.cuda()
clip_model.eval()

# 开始执行具体的环境配置
args = proc_Config
dataset = prior.load_dataset("procthor-10k")
dataset_train = dataset["train"]
controller_env = ThorEnvCode(args, use_CloudRendering=True)

# 加载文件
total_cat2idx_path = "./utils/total_cat2idx.json"
total_cat2idx = json.load(open(total_cat2idx_path))
# 增加wall
total_cat2idx["wall"] = 94
total_cat2idx["floor"] = 95

sem_map = Semantic_Mapping(
    args=proc_Config,
    clip_model=clip_model,
    clip_preprocess=clip_preprocess,
    total_cat2idx=total_cat2idx
)
sem_map.reset()

# load_pkl文件
pkl_file = "./log_file/llama_vln_parsed_response_v8_val_result_all.pkl"
with open(pkl_file, "rb") as f:
    info_dict = pickle.load(f)

base_save_path = "./visualization_results"

for info_dict_one in tqdm(info_dict[10:50]):
    house_id = info_dict_one["house_id"]
    step_action_list = info_dict_one["step_action_list"]
    success = info_dict_one["success"]
    if not success:
        continue
    # env load scene
    rgb_frame, depth_frame = controller_env.load_scene(house_name=dataset_train[house_id])
    # 开始环视
    rgb_base_save_path = os.path.join(base_save_path, "house-" + str(house_id) + "_" + "rgb")
    frontiers_base_save_path = os.path.join(base_save_path, "house-" + str(house_id) + "_" + "frontiers")

    if not os.path.exists(rgb_base_save_path):
        os.makedirs(rgb_base_save_path)
    if not os.path.exists(frontiers_base_save_path):
        os.makedirs(frontiers_base_save_path)

    sem_map.reset()
    save_id = 0
    for _ in range(4):
        # 增加转圈
        rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
        global_sem_map, global_sem_feature_map = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
        save_step_image(rgb_base_save_path, frontiers_base_save_path, rgb, global_sem_map, id=save_id)
        save_id = save_id + 1
        event = controller_env.step(action="RotateRight", forceAction=True)
    print("end surrounding ----------------------")
    nav_path_list = []
    for step_one in tqdm(step_action_list):
        rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
        select_feature_list, frontiers_label_dict_list = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="frontiers")
        action = step_one["action"]
        target = step_one["target"]
        target_nav_position = step_one["nav_position"]
        # 根据target生成候选
        candidate_position_list, candidate_target_obj_id_list = controller_env.get_candidate_by_name(target)
        candidate_index_list = controller_env.get_candidate_id_by_object_id(candidate_target_obj_id_list)
        if len(candidate_index_list) > 0:
            # 视野中有物体, 则直接选取s1规划
            print("find object, start s1")
            select_candidate_index = candidate_index_list[0]
            # target_nav_position = candidate_position_list[select_candidate_index]
            target_object_id = candidate_target_obj_id_list[select_candidate_index]
        else:
            # target_nav_position = sem_map.pixel2world_point(nav_point)
            target_object_id = None

        object_mask = sem_map.get_nav_position_mask(candidate_position_list)
        object_mask = sem_map.dilate_frontiers_mask(object_mask, kernel_size=5, padding=2)
        # 执行具体巡航动作
        if "goto" in action.lower() or target_object_id is None:
            perspective_camera_view(controller_env)
            print("target_nav_position: ", target_nav_position)
            path_distance = nav_location(controller_env, target_nav_position, sem_map, save_id)
            save_id = save_id + 1
        else:
            path_distance = 0
            # 环视寻找物体
            is_find = controller_env.check_target_frame(target_object_id)
            rgb, depth_frame, mask_list, info_dict = controller_env.execAction(action=[
                action], target_arg=target_object_id.split("|")[0],
                target_position=target_nav_position, target_object_id=target_object_id)
            semantic_map, _ = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
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

            save_step_image(rgb_base_save_path, frontiers_base_save_path, rgb, semantic_map, vis_dict, id=save_id)
            save_id = save_id + 1
