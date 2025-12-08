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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gzip
from prior import LazyJsonDataset


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


def save_step_image(rgb_base_path, frontiers_base_path, rgb_image, semantic_map, frontiers_dict=None, info_dict=None, id=0):
    rgb_save_path = os.path.join(rgb_base_path, str(id) + ".png")
    frontiers_save_path = os.path.join(frontiers_base_path, str(id) + ".png")
    save_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    mask_save_path = os.path.join(vis_mask_base_save_path, str(id) + ".png")
    third_rgb_save_path = os.path.join(third_view_base_save_path, str(id) + ".png")

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
    if "vis_mask" in info_dict.keys():
        sem_vis = info_dict["vis_mask"]
        cv2.imwrite(mask_save_path, sem_vis)
    cv2.imwrite(rgb_save_path, save_rgb)

    # 增加第三视角
    third_view_rgb = controller_env.last_event.third_party_camera_frames[0]
    third_view_rgb = cv2.cvtColor(third_view_rgb, cv2.COLOR_BGR2RGB)
    third_view_rgb = np.array(third_view_rgb)
    cv2.putText(third_view_rgb, img_caption, (250, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.imwrite(third_rgb_save_path, third_view_rgb)


def nav_location(controller_env, target_nav_position, sem_map, save_id):
    nav_action, path = controller_env.parse_nav_action(target_nav_position)
    path_distance = (len(path) - 1) * 0.25
    for nav_action_id, action_one in enumerate(tqdm(nav_action)):
        if len(nav_action) > 3:
            if "move" in action_one.lower():
                controller_env.set_third_view()
        rgb, depth_frame, mask_list, info_dict = controller_env.to_thor_api_exec([action_one])
        if nav_action_id % 5 == 0:
            info_dict["add_robot_mask"] = False
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

        save_step_image(rgb_base_save_path, frontiers_base_save_path, rgb, semantic_map, vis_dict, info_dict=info_dict, id=save_id)
        save_id = save_id + 1
    return path_distance, save_id


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
# dataset = prior.load_dataset("procthor-10k")
dataset = load_dataset()
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
# pkl_file = "./log_file/llama_vln_parsed_response_v8_val_result_2258.pkl"
# with open(pkl_file, "rb") as f:
#     info_dict = pickle.load(f)
# pkl_file = "./log_file/llava_vln_parsed_response_v8_val_result_v4_planning_gt.pkl"
# pkl_file = "./log_file/llava_s1_s2_vln_parsed_response_v8_val_result_frontiers_150.pkl"
# pkl_file = "./log_file/llava_vln_parsed_response_v8_val_result_v4_planning_gt_200.pkl"
# with open(pkl_file, "rb") as f:
#     info_dict = pickle.load(f)
pkl_file = "./log_file/llava_s1_s2_vln_parsed_response_procthor_nav_seg_gt/29_house_2271.pkl"
with open(pkl_file, "rb") as f:
    info_dict = pickle.load(f)
# 65
# house_index_list = [49, 52, 65, 71, 94, 121, 187, 192]
# house_index_list = [66, 127]
# house_index_list = [121]
house_index_list = [190, 152, 150, 148, 128, 121, 71]
# house_index_list = [152, 190]
use_sourrding = True

base_save_path = "./visualization/visualization_results_test_prediction_v2"
# base_save_path = "./visualization_results_random"

for val_index, info_dict_one in enumerate(tqdm([info_dict])):
    # if val_index not in house_index_list:
    #     continue
    house_id = info_dict_one["house_id"]
    print(house_id)
    step_action_list = info_dict_one["step_action_list"]
    success = info_dict_one["success"]
    print("是否成功：", success)
    # if not success:
    #     continue
    # env load scene
    rgb_frame, depth_frame = controller_env.load_scene(house_name=dataset_train[house_id])
    # 增加第三视角
    controller_env.add_third_camera()
    # 开始环视
    rgb_base_save_path = os.path.join(base_save_path, "house-" + str(house_id) + "_" + "rgb")
    frontiers_base_save_path = os.path.join(base_save_path, "house-" + str(house_id) + "_" + "frontiers")
    vis_mask_base_save_path = os.path.join(base_save_path, "house-" + str(house_id) + "_" + "masks")
    third_view_base_save_path = os.path.join(base_save_path, "house-" + str(house_id) + "_" + "third_view")

    if not os.path.exists(rgb_base_save_path):
        os.makedirs(rgb_base_save_path)
    if not os.path.exists(frontiers_base_save_path):
        os.makedirs(frontiers_base_save_path)
    if not os.path.exists(vis_mask_base_save_path):
        os.makedirs(vis_mask_base_save_path)
    if not os.path.exists(third_view_base_save_path):
        os.makedirs(third_view_base_save_path)

    sem_map.reset()
    save_id = 0
    for _ in range(4):
        # 增加转圈
        rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
        info_dict["add_robot_mask"] = True
        info_dict["region_size"] = 0.55
        img_caption = "Look Around"
        global_sem_map, global_sem_feature_map = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
        save_step_image(rgb_base_save_path, frontiers_base_save_path, rgb, global_sem_map, info_dict=info_dict, id=save_id)
        save_id = save_id + 1
        event = controller_env.step(action="RotateRight", forceAction=True)
    print("end surrounding ----------------------")
    nav_path_list = []
    for step_index, step_one in enumerate(tqdm(step_action_list)):
        if use_sourrding:
            if step_index >= 1:
                for _ in range(4):
                    # 增加转圈
                    rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
                    # 如果seen_object_list有更新就启动
                    # if info_dict["new_add_object"]:
                    info_dict["add_robot_mask"] = True
                    info_dict["region_size"] = 0.30
                    global_sem_map, global_sem_feature_map = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
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
                    save_step_image(rgb_base_save_path, frontiers_base_save_path, rgb, global_sem_map, vis_dict, info_dict=info_dict, id=save_id)
                    save_id = save_id + 1
                    event = controller_env.step(action="RotateRight", forceAction=True)
                print("end surrounding ----------------------")

                # event = controller_env.step(action="LookDown", degrees=30)
                #
                # for _ in range(4):
                #     # 增加转圈
                #     rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
                #     # 如果seen_object_list有更新就启动
                #     # if info_dict["new_add_object"]:
                #     info_dict["add_robot_mask"] = True
                #     info_dict["region_size"] = 0.20
                #     global_sem_map, global_sem_feature_map = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
                #     robot_position = info_dict["camera_world_xyz"]
                #     robot_position_mask = sem_map.get_nav_position_mask([robot_position])
                #     robot_position_mask = sem_map.dilate_frontiers_mask(robot_position_mask, kernel_size=5, padding=2)
                #     nav_path_list.append(robot_position)
                #     nav_path_mask = sem_map.get_nav_position_mask(nav_path_list)
                #     nav_path_mask = sem_map.dilate_frontiers_mask(nav_path_mask, kernel_size=5, padding=2)
                #
                #     vis_dict = dict(
                #         object_mask=object_mask,
                #         robot_mask=robot_position_mask,
                #         frontiers_mask=sem_map.vis_frontiers_mask,
                #         nav_path_mask=nav_path_mask)
                #     save_step_image(rgb_base_save_path, frontiers_base_save_path, rgb, global_sem_map, vis_dict, id=save_id)
                #     save_id = save_id + 1
                #     event = controller_env.step(action="RotateRight", forceAction=True)
                # print("end surrounding ----------------------")
                # perspective_camera_view(controller_env)

        # nav_path_list = []
        rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
        select_feature_list, frontiers_label_dict_list = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="frontiers")
        action = step_one["action"]
        target = step_one["target"]
        target_nav_position = step_one["nav_position"]
        target_object_id = step_one["target_id"]
        print(action)
        # 根据target生成候选
        candidate_position_list, candidate_target_obj_id_list = controller_env.get_candidate_by_name(target)
        candidate_index_list = controller_env.get_candidate_id_by_object_id(candidate_target_obj_id_list)
        # if len(candidate_index_list) > 0:
        #     # 视野中有物体, 则直接选取s1规划
        #     print("find object, start s1")
        #     select_candidate_index = candidate_index_list[0]
        #     # target_nav_position = candidate_position_list[select_candidate_index]
        #     # target_object_id = candidate_target_obj_id_list[select_candidate_index]
        # else:
        #     # target_nav_position = sem_map.pixel2world_point(nav_point)
        #     # target_object_id = None

        object_mask = sem_map.get_nav_position_mask(candidate_position_list)
        object_mask = sem_map.dilate_frontiers_mask(object_mask, kernel_size=5, padding=2)
        # 执行具体巡航动作
        if "goto" in action.lower() or target_object_id is None:
            input_dict = {"metadata": controller_env.last_event.metadata,
                          "target_name": target}
            # response_dict = sem_map.get_object_id_from_sem_map(input_dict)
            # if response_dict["success"]:
            #     continue
            perspective_camera_view(controller_env)
            img_caption = action + ": " + target
            print("target_nav_position: ", target_nav_position)
            path_distance, save_id = nav_location(controller_env, target_nav_position, sem_map, save_id)
            save_id = save_id + 1
        else:
            path_distance = 0
            # add 距离限制
            # is_near = controller_env.check_is_near_target(target_object_id)
            # if not is_near:
            #     path_distance, save_id = nav_location(controller_env, target_nav_position, sem_map, save_id)
            #     save_id = save_id + 1
            # 环视寻找物体
            # is_find = controller_env.check_target_frame(target_object_id)
            # 增加实现细节------------------------------
            is_find = False
            if controller_env.args.alfred_scene:
                look_action_list = []
            else:
                look_action_list = [{"action": "LookUp", "degrees": 30},
                                    {"action": "LookDown", "degrees": 60},
                                    {"action": "LookUp", "degrees": 30}]
            for i in range(4):
                rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
                # print(controller_env.last_event.instance_detections2D.keys())
                if proc_Config.use_sem_seg:
                    is_find_in_frame = controller_env.check_isin_frame_alfred(target_object_id)
                else:
                    is_find_in_frame = controller_env.check_isin_frame(target_object_id)
                if is_find_in_frame:
                    is_find = True
                    break
                for action_id, action_dict_one in enumerate(look_action_list):
                    action_name = action_dict_one["action"]
                    degrees = action_dict_one["degrees"]
                    event = controller_env.step(action=action_name, degrees=degrees)
                    rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
                    # print(controller_env.last_event.instance_detections2D.keys())
                    # metadata = controller.last_event.metadata
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
                    img_caption = "Look for Target"
                    save_step_image(rgb_base_save_path, frontiers_base_save_path, rgb, global_sem_map, vis_dict, info_dict=info_dict, id=save_id)
                    save_id = save_id + 1
                    if proc_Config.use_sem_seg:
                        is_find_in_frame = controller_env.check_isin_frame_alfred(target_object_id)
                    else:
                        is_find_in_frame = controller_env.check_isin_frame(target_object_id)

                    if is_find_in_frame:
                        is_find = True
                        break
                if is_find:
                    break
                # 增加转圈
                event = controller_env.step(action="RotateRight", forceAction=True)
            print("是否找到物体:", is_find)
            # if not is_find:
            #     print("not find:", target_object_id)
            #     continue
                # print("not find:", target_object_id)
                # new_nav_point = sem_map.get_sem_map_object_position(target_object_id)
                # if new_nav_point is not None:
                #     add_path_distance, save_id = nav_location(controller_env, new_nav_point, sem_map, save_id)
                #     save_id = save_id + 1
                #     path_distance = path_distance + add_path_distance
            img_caption = action + ": " + target
            if "PutObject" in [action]:
                hand_object_list = controller_env.last_event.metadata["inventoryObjects"]
                if len(hand_object_list) > 0:
                    hand_object = controller_env.last_event.metadata['inventoryObjects'][0]["objectId"]
                    isInteractable = next(obj["isInteractable"] for obj in controller_env.last_event.metadata["objects"]
                                          if obj["objectId"] == hand_object)
                    if not isInteractable:
                        rgb, depth_frame, mask_list, info_dict = controller_env.to_thor_api_exec([
                            "EnableObject"], object_id=hand_object)
                    info_dict["interactive_object"] = hand_object
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

                    save_step_image(rgb_base_save_path, frontiers_base_save_path, rgb, semantic_map, vis_dict, info_dict=info_dict, id=save_id)
                    save_id = save_id + 1

            rgb, depth_frame, mask_list, info_dict = controller_env.execAction(action=[
                action], target_arg=target, target_position=target_nav_position, target_object_id=target_object_id)
            print(controller_env.last_event)
            semantic_map, _ = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
            robot_position = info_dict["camera_world_xyz"]
            robot_position_mask = sem_map.get_nav_position_mask([robot_position])
            robot_position_mask = sem_map.dilate_frontiers_mask(robot_position_mask, kernel_size=5, padding=2)
            nav_path_list.append(robot_position)
            nav_path_mask = sem_map.get_nav_position_mask(nav_path_list)
            nav_path_mask = sem_map.dilate_frontiers_mask(nav_path_mask, kernel_size=5, padding=2)
            img_caption = action + ": " + target
            vis_dict = dict(
                object_mask=object_mask,
                robot_mask=robot_position_mask,
                frontiers_mask=sem_map.vis_frontiers_mask,
                nav_path_mask=nav_path_mask)
            rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
            save_step_image(rgb_base_save_path, frontiers_base_save_path, rgb, semantic_map, vis_dict, info_dict=info_dict, id=save_id)
            save_id = save_id + 1
            controller_env.reset_arrival_position()
            # if action == "OpenObject" and target == "Fridge":
            #     # agent后退两步, 从而规避碰撞
            #     for _ in range(2):
            #         move_back_action = dict(action="MoveBack", forceAction=True)
            #         controller_env.step(move_back_action)
            #         robot_position = info_dict["camera_world_xyz"]
            #         robot_position_mask = sem_map.get_nav_position_mask([robot_position])
            #         robot_position_mask = sem_map.dilate_frontiers_mask(robot_position_mask, kernel_size=5, padding=2)
            #         nav_path_list.append(robot_position)
            #         nav_path_mask = sem_map.get_nav_position_mask(nav_path_list)
            #         nav_path_mask = sem_map.dilate_frontiers_mask(nav_path_mask, kernel_size=5, padding=2)
            #         img_caption = "Avoid Collision"
            #         vis_dict = dict(
            #             object_mask=object_mask,
            #             robot_mask=robot_position_mask,
            #             frontiers_mask=sem_map.vis_frontiers_mask,
            #             nav_path_mask=nav_path_mask)
            #         rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
            #         save_step_image(rgb_base_save_path, frontiers_base_save_path, rgb, semantic_map, vis_dict, info_dict=info_dict, id=save_id)
            #         save_id = save_id + 1

            if "PickupObject" in [action]:
                hand_object_list = controller_env.last_event.metadata["inventoryObjects"]
                if len(hand_object_list) > 0:
                    hand_object = controller_env.last_event.metadata['inventoryObjects'][0]["objectId"]
                    isInteractable = next(obj["isInteractable"] for obj in controller_env.last_event.metadata["objects"]
                                          if obj["objectId"] == hand_object)
                    if isInteractable:
                        rgb, depth_frame, mask_list, info_dict = controller_env.to_thor_api_exec([
                            "DisableObject"], object_id=hand_object)
                        info_dict["interactive_object"] = hand_object
                        _, _ = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
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

                        save_step_image(rgb_base_save_path, frontiers_base_save_path, rgb, semantic_map, vis_dict, info_dict=info_dict, id=save_id)
                        save_id = save_id + 1