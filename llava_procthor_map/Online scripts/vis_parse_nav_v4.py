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
from utils.done_action import DoneAction
from model import longclip

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


def save_step_image(input_dict):
    sem_map = input_dict["sem_map"]
    save_id = input_dict["save_id"]
    rgb_base_path = input_dict["rgb_base_save_path"]
    frontiers_base_path = input_dict["frontiers_base_save_path"]
    vis_mask_base_save_path = input_dict["vis_mask_base_save_path"]
    rgb_image = input_dict["rgb_image"]
    frontiers_dict = input_dict["vis_dict"]
    info_dict = input_dict["info_dict"]
    # add ----------------
    long_clip_model = input_dict["long_clip"]
    long_clip_preprocess = input_dict["long_clip_preprocess"]
    image_description_text = input_dict["image_description_text"]
    # ---------------------
    controller_env = input_dict["controller_env"]
    third_rgb_base_path = input_dict["third_rgb_base_path"]

    rgb_save_path = os.path.join(rgb_base_path, str(save_id) + ".png")
    rgb_save_ori_path = os.path.join(rgb_base_path, "ori_" + str(save_id) + ".png")
    frontiers_save_path = os.path.join(frontiers_base_path, str(save_id) + ".png")
    mask_save_path = os.path.join(vis_mask_base_save_path, str(save_id) + ".png")
    third_rgb_save_path = os.path.join(third_rgb_base_path, str(save_id) + ".png")

    vis_mask = visualize_map(sem_map)
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
    plt.savefig(frontiers_save_path, dpi=300)
    plt.close()
    if "vis_mask" in info_dict.keys():
        sem_vis = info_dict["vis_mask"]
        cv2.imwrite(mask_save_path, sem_vis)
    rgb_image = np.array(rgb_image)
    device = torch.device("cuda:0")
    # 计算分数
    description_list = image_description_text.split(" captures")
    comparative_description = "The image does not capture" + description_list[-1]
    # comparative_description_2 = "There is an empty house."
    comparative_description_3 = "The image captures nothing."
    text = longclip.tokenize([image_description_text, comparative_description, comparative_description_3]).to(device)
    image = long_clip_preprocess(Image.fromarray(rgb_image)).unsqueeze(0).to(device)
    with torch.no_grad():
        # image_features = long_clip_model.encode_image(image)
        # text_features = long_clip_model.encode_text(text)
        logits_per_image, logits_per_text = long_clip_model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    # print(probs)
    score = probs[0]
    save_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    save_rgb = np.array(save_rgb)
    cv2.imwrite(rgb_save_ori_path, save_rgb)
    cv2.putText(save_rgb, str(score), (5, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.imwrite(rgb_save_path, save_rgb)

    # add third view
    third_view_rgb = controller_env.last_event.third_party_camera_frames[0]
    third_view_rgb = cv2.cvtColor(third_view_rgb, cv2.COLOR_BGR2RGB)
    third_view_rgb = np.array(third_view_rgb)
    # cv2.putText(third_view_rgb, img_caption, (250, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.imwrite(third_rgb_save_path, third_view_rgb)


def nav_location(input_dict):
    controller_env = input_dict["agent"]
    sem_map = input_dict["sem_map"]
    save_id = input_dict["save_id"]
    nav_path_list = input_dict["nav_path_list"]
    object_mask = input_dict["object_mask"]
    rgb_base_save_path = input_dict["rgb_base_save_path"]
    frontiers_base_save_path = input_dict["frontiers_base_save_path"]
    vis_mask_base_save_path = input_dict["vis_mask_base_save_path"]
    target_nav_position = input_dict["target_nav_position"]
    third_rgb_base_path = input_dict["third_rgb_base_path"]

    nav_action, path = controller_env.parse_nav_action(target_nav_position)
    if len(path) < 5:
        pass
    else:
        path_distance = (len(path) - 1) * 0.25
        new_nav_action = []
        # for action_one in nav_action:
        #     if "RotateRight" in action_one or "RotateLeft" in action_one:
        #         new_nav_action.extend([action_one] * 18)
        #     else:
        #         new_nav_action.extend([action_one])
        for nav_action_id, action_one in enumerate(tqdm(nav_action[:-2])):
            rgb, depth_frame, mask_list, info_dict = controller_env.to_thor_api_exec([action_one])
            if nav_action_id % 3 == 0 or nav_action_id in [len(nav_action)-1]:
                if len(nav_action) > 3:
                    if "move" in action_one.lower():
                        controller_env.set_third_view()
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
            input_dict = {"rgb_base_save_path": rgb_base_save_path,
                          "frontiers_base_save_path": frontiers_base_save_path,
                          "vis_mask_base_save_path": vis_mask_base_save_path,
                          "rgb_image": rgb,
                          "sem_map": semantic_map,
                          "info_dict": info_dict,
                          "save_id": save_id,
                          "vis_dict": vis_dict,
                          "long_clip": input_dict["long_clip"],
                          "long_clip_preprocess": input_dict["long_clip_preprocess"],
                          "image_description_text": input_dict["image_description_text"],
                          "third_rgb_base_path": third_rgb_base_path,
                          "controller_env": controller_env
                          }
            save_step_image(input_dict)
            save_id = save_id + 1

    response_dict = {"save_id": save_id,
                     "nav_path_list": nav_path_list}
    return response_dict


def sourrding_view(input_dict):
    agent = input_dict["agent"]
    sem_map = input_dict["sem_map"]
    save_id = input_dict["save_id"]
    nav_path_list = input_dict["nav_path_list"]
    object_mask = input_dict["object_mask"]
    rgb_base_save_path = input_dict["rgb_base_save_path"]
    frontiers_base_save_path = input_dict["frontiers_base_save_path"]
    vis_mask_base_save_path = input_dict["vis_mask_base_save_path"]
    third_rgb_base_path = input_dict["third_rgb_base_path"]

    for _ in range(4):
        # 增加转圈
        rgb, depth_frame, mask_list, info_dict = agent.get_obs()
        # 如果seen_object_list有更新就启动
        # if info_dict["new_add_object"]:
        info_dict["add_robot_mask"] = True
        info_dict["region_size"] = 0.20
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

        input_dict = {"rgb_base_save_path": rgb_base_save_path,
                      "frontiers_base_save_path": frontiers_base_save_path,
                      "vis_mask_base_save_path": vis_mask_base_save_path,
                      "rgb_image": rgb,
                      "sem_map": global_sem_map,
                      "info_dict": info_dict,
                      "save_id": save_id,
                      "vis_dict": vis_dict,
                      "long_clip": input_dict["long_clip"],
                      "long_clip_preprocess": input_dict["long_clip_preprocess"],
                      "image_description_text": input_dict["image_description_text"],
                      "third_rgb_base_path": third_rgb_base_path,
                      "controller_env": agent
                      }
        save_step_image(input_dict)
        save_id = save_id + 1
        # event = agent.step(action="RotateRight", forceAction=True, smooth_nav=True)
        rgb, depth_frame, mask_list, info_dict = agent.to_thor_api_exec(["RotateRight"])
    print("end surrounding ----------------------")
    response_dict = {"save_id": save_id,
                     "nav_path_list": nav_path_list}
    return response_dict


def main():
    # load clip
    # -----------------------------------
    open_clip_model = "ViT-H-14"
    torch.autograd.set_grad_enabled(False)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Initializing model...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        open_clip_model, pretrained="./checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
    )
    clip_model.cuda("cuda:1")
    clip_model.eval()
    # -----------------------------------
    # load long clip
    # -----------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    long_clip_model, long_clip_preprocess = longclip.load("./checkpoints/longclip-B.pt", device=device)
    long_clip_model.eval()
    # -----------------------------------
    # 开始执行具体的环境配置
    args = proc_Config
    # dataset = prior.load_dataset("procthor-10k")
    dataset = load_dataset()
    dataset_train = dataset["train"]
    controller_env = ThorEnvCode(args, use_CloudRendering=True)
    # -----------------------------------
    if proc_Config.alfred_scene:
        total_cat2idx_path = proc_Config.total_cat2idx_alfred_path
    else:
        total_cat2idx_path = proc_Config.total_cat2idx_procthor_path
    total_cat2idx = json.load(open(total_cat2idx_path))
    # 处理sem_map
    # -----------------------------------
    sem_map = Semantic_Mapping(
        args=proc_Config,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        total_cat2idx=total_cat2idx
    )
    sem_map.reset()
    # -----------------------------------
    # 增加原始文件读取
    space_dataset_file_path = "./data/spaced_parse_instruction_hard_v14_val.json"
    space_dataset_file = json.load(open(space_dataset_file_path))
    house_id_dict = {}
    for space_dataset_one in space_dataset_file:
        object_list_one = []
        output_one = space_dataset_one["output"]
        house_id_one = int(space_dataset_one["house_id"])
        for action_dict_one in output_one:
            arg = action_dict_one["arg"]
            if arg != " ":
                object_list_one.append(arg)
        object_list_one = list(set(object_list_one))
        house_id_dict[house_id_one] = object_list_one
    # -----------------------------------
    image_prompt = "The image captures one of either"
    # 配置具体需要查看的文件
    # 41_house_2349
    # 37_house_2342
    pkl_file = "./log_file/llava_s1_s2_vln_parsed_response_procthor_detic_v14_abstract_small_attention/49_house_1640.pkl"
    with open(pkl_file, "rb") as f:
        info_dict = pickle.load(f)
    # 基础保存路径
    base_save_path = "./visualization/visualization_results_test_prediction_final"
    for val_index, info_dict_one in enumerate(tqdm([info_dict])):
        house_id = info_dict_one["house_id"]
        object_list = house_id_dict[int(house_id)]
        object_list_str = " "
        for object_index, object_one in enumerate(object_list):
            if object_index != len(object_list) - 1:
                object_list_str = object_list_str + "a " + object_one.lower() + ", "
            else:
                object_list_str = object_list_str + "or " + "a " + object_one.lower()
        image_description_text = image_prompt + object_list_str + "."
        print(image_description_text)
        print(house_id)
        step_action_list = info_dict_one["step_action_list"]
        success = info_dict_one["success"]
        print("是否成功：", success)
        rgb_frame, depth_frame = controller_env.load_scene(house_name=dataset_train[house_id])

        controller_env.add_third_camera()

        done_action = DoneAction(
            agent=controller_env
        )
        done_action.reset()

        # 单个房间基础路径
        rgb_base_save_path = os.path.join(base_save_path, "house-" + str(house_id) + "_" + "rgb")
        frontiers_base_save_path = os.path.join(base_save_path, "house-" + str(house_id) + "_" + "frontiers")
        vis_mask_base_save_path = os.path.join(base_save_path, "house-" + str(house_id) + "_" + "masks")
        third_rgb_base_path = os.path.join(base_save_path, "house-" + str(house_id) + "_" + "third_view")

        if not os.path.exists(rgb_base_save_path):
            os.makedirs(rgb_base_save_path)
        if not os.path.exists(frontiers_base_save_path):
            os.makedirs(frontiers_base_save_path)
        if not os.path.exists(vis_mask_base_save_path):
            os.makedirs(vis_mask_base_save_path)
        if not os.path.exists(third_rgb_base_path):
            os.makedirs(third_rgb_base_path)

        sem_map.reset()
        save_id = 0
        # 先进行环视
        for _ in range(4):
            # 增加转圈
            rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
            info_dict["add_robot_mask"] = True
            info_dict["region_size"] = 0.55
            global_sem_map, global_sem_feature_map = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
            input_dict = {"rgb_base_save_path": rgb_base_save_path,
                          "frontiers_base_save_path": frontiers_base_save_path,
                          "vis_mask_base_save_path": vis_mask_base_save_path,
                          "rgb_image": rgb,
                          "sem_map": global_sem_map,
                          "info_dict": info_dict,
                          "save_id": save_id,
                          "vis_dict": None,
                          "long_clip": long_clip_model,
                          "long_clip_preprocess": long_clip_preprocess,
                          "image_description_text": image_description_text,
                          "third_rgb_base_path": third_rgb_base_path,
                          "controller_env": controller_env}
            save_step_image(input_dict)
            save_id = save_id + 1
            # event = controller_env.step(action="RotateRight", forceAction=True, smooth_nav=True)
            rgb, depth_frame, mask_list, info_dict = controller_env.to_thor_api_exec(["RotateRight"])
        # 循环主体------------------------
        # 整体巡航路径
        nav_path_list = []
        for step_index, step_one in enumerate(tqdm(step_action_list)):
            controller_env.reset_arrival_position()
            if step_index >= 1:
                input_dict = {"agent": controller_env,
                             "sem_map": sem_map,
                             "save_id": save_id,
                             "nav_path_list": nav_path_list,
                             "object_mask": object_mask,
                             "rgb_base_save_path": rgb_base_save_path,
                             "frontiers_base_save_path": frontiers_base_save_path,
                             "vis_mask_base_save_path": vis_mask_base_save_path,
                             "long_clip": long_clip_model,
                             "long_clip_preprocess": long_clip_preprocess,
                             "image_description_text": image_description_text,
                             "third_rgb_base_path": third_rgb_base_path,
                             }
                response_dict = sourrding_view(input_dict)
                save_id = response_dict["save_id"]
                nav_path_list = response_dict["nav_path_list"]

            # 开始主体巡航动作
            rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
            select_feature_list, frontiers_label_dict_list = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="frontiers")
            action = step_one["action"]
            target = step_one["target"]
            target_nav_position = step_one["nav_position"]
            target_object = step_one["target"]
            # target_object_id = save_id["target_object_id"]
            print("action: ", action)
            print("target: ", target)

            # 根据target生成候选
            candidate_position_list, candidate_target_obj_id_list = controller_env.get_candidate_by_name(target)
            candidate_index_list = controller_env.get_candidate_id_by_object_id(candidate_target_obj_id_list)

            object_mask = sem_map.get_nav_position_mask(candidate_position_list)
            object_mask = sem_map.dilate_frontiers_mask(object_mask, kernel_size=5, padding=2)
            # 执行具体巡航动作
            if "goto" in action.lower() or target_object is None:
                # 抬头走路
                controller_env.perspective_camera_view()
                input_dict = {"agent": controller_env,
                              "sem_map": sem_map,
                              "save_id": save_id,
                              "nav_path_list": nav_path_list,
                              "object_mask": object_mask,
                              "rgb_base_save_path": rgb_base_save_path,
                              "frontiers_base_save_path": frontiers_base_save_path,
                              "vis_mask_base_save_path": vis_mask_base_save_path,
                              "target_nav_position": target_nav_position,
                              "long_clip": long_clip_model,
                              "long_clip_preprocess": long_clip_preprocess,
                              "image_description_text": image_description_text,
                              "third_rgb_base_path": third_rgb_base_path
                              }
                print("target_nav_position: ", target_nav_position)
                response_dict = nav_location(input_dict)
                save_id = response_dict["save_id"]
                save_id = save_id + 1
            else:
                is_find = False
                if controller_env.args.alfred_scene:
                    look_action_list = []
                else:
                    look_action_list = [{"action": "LookUp", "degrees": 30},
                                        {"action": "LookDown", "degrees": 60},
                                        {"action": "LookUp", "degrees": 30}]
                for i in range(4):
                    rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
                    is_find_in_frame = controller_env.check_isin_frame_alfred(target_object)
                    if is_find_in_frame:
                        is_find = True
                        break
                    for action_id, action_dict_one in enumerate(look_action_list):
                        action_name = action_dict_one["action"]
                        degrees = action_dict_one["degrees"]
                        event = controller_env.step(action=action_name, degrees=degrees)
                        rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
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
                        input_dict = {"sem_map": sem_map.global_sem_map,
                                      "save_id": save_id,
                                      "rgb_image": rgb,
                                      "rgb_base_save_path": rgb_base_save_path,
                                      "frontiers_base_save_path": frontiers_base_save_path,
                                      "vis_mask_base_save_path": vis_mask_base_save_path,
                                      "vis_dict": vis_dict,
                                      "info_dict": info_dict,
                                      "long_clip": long_clip_model,
                                      "long_clip_preprocess": long_clip_preprocess,
                                      "image_description_text": image_description_text,
                                      "third_rgb_base_path": third_rgb_base_path,
                                      "controller_env": controller_env
                                      }
                        save_step_image(input_dict)
                        save_id = save_id + 1

                        is_find_in_frame = controller_env.check_isin_frame_alfred(target_object)

                        if is_find_in_frame:
                            is_find = True
                            break
                    if is_find:
                        break
                    # 增加转圈
                    # event = controller_env.step(action="RotateRight", forceAction=True, smooth_nav=True)
                    rgb, depth_frame, mask_list, info_dict = controller_env.to_thor_api_exec(["RotateRight"])

                    rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()

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
                    input_dict = {"sem_map": sem_map.global_sem_map,
                                  "save_id": save_id,
                                  "rgb_image": rgb,
                                  "rgb_base_save_path": rgb_base_save_path,
                                  "frontiers_base_save_path": frontiers_base_save_path,
                                  "vis_mask_base_save_path": vis_mask_base_save_path,
                                  "vis_dict": vis_dict,
                                  "info_dict": info_dict,
                                  "long_clip": long_clip_model,
                                  "long_clip_preprocess": long_clip_preprocess,
                                  "image_description_text": image_description_text,
                                  "third_rgb_base_path": third_rgb_base_path,
                                  "controller_env": controller_env
                                  }
                    save_step_image(input_dict)
                    save_id = save_id + 1

                if not is_find:
                    print("not find, move other position")
                    nav_point_list = sem_map.get_sem_map_object_position_list(target)
                    for nav_point_one in nav_point_list:
                        target_nav_position_one = sem_map.pixel2world_point(nav_point_one)
                        input_dict = {"agent": controller_env,
                                      "sem_map": sem_map,
                                      "save_id": save_id,
                                      "nav_path_list": nav_path_list,
                                      "object_mask": object_mask,
                                      "rgb_base_save_path": rgb_base_save_path,
                                      "frontiers_base_save_path": frontiers_base_save_path,
                                      "vis_mask_base_save_path": vis_mask_base_save_path,
                                      "target_nav_position": target_nav_position_one,
                                      "long_clip": long_clip_model,
                                      "long_clip_preprocess": long_clip_preprocess,
                                      "image_description_text": image_description_text,
                                      "third_rgb_base_path": third_rgb_base_path
                                      }
                        print("target_nav_position: ", target_nav_position)
                        response_dict = nav_location(input_dict)
                        save_id = response_dict["save_id"]
                        nav_path_list = response_dict["nav_path_list"]

                        save_id = save_id + 1
                        is_find = controller_env.check_target_frame_alfred(target)
                        # 每次都要去掉
                        if not is_find:
                            robot_position = np.asarray(list(
                                controller_env.last_event.metadata["agent"]["position"].values()))
                            sem_map.remove_unseen_target_map(object_name=target, robot_position=robot_position)
                        if is_find:
                            break
                if not is_find:
                    # 没有找到物体, 需要更新object map
                    robot_position = np.asarray(list(controller_env.last_event.metadata["agent"]["position"].values()))
                    sem_map.remove_unseen_target_map(object_name=target, robot_position=robot_position)
                    # 同时, 上一步done action需要进行剔除
                    # done_action.reflective_action()
                    rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()

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

                    input_dict = {"rgb_base_save_path": rgb_base_save_path,
                                  "frontiers_base_save_path": frontiers_base_save_path,
                                  "vis_mask_base_save_path": vis_mask_base_save_path,
                                  "rgb_image": rgb,
                                  "sem_map": sem_map.global_sem_map,
                                  "info_dict": info_dict,
                                  "save_id": save_id,
                                  "vis_dict": vis_dict,
                                  "long_clip": long_clip_model,
                                  "long_clip_preprocess": long_clip_preprocess,
                                  "image_description_text": image_description_text,
                                  "third_rgb_base_path": third_rgb_base_path,
                                  "controller_env": controller_env
                                  }
                    save_step_image(input_dict)
                    save_id = save_id + 1
                    continue

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
                        input_dict = {"rgb_base_save_path": rgb_base_save_path,
                                      "frontiers_base_save_path": frontiers_base_save_path,
                                      "vis_mask_base_save_path": vis_mask_base_save_path,
                                      "rgb_image": rgb,
                                      "sem_map": semantic_map,
                                      "info_dict": info_dict,
                                      "save_id": save_id,
                                      "vis_dict": vis_dict,
                                      "long_clip": long_clip_model,
                                      "long_clip_preprocess": long_clip_preprocess,
                                      "image_description_text": image_description_text,
                                      "third_rgb_base_path": third_rgb_base_path,
                                      "controller_env": controller_env
                                      }
                        save_step_image(input_dict)
                        save_id = save_id + 1

                rgb, depth_frame, mask_list, info_dict = controller_env.execAction(action=[
                    action], target_arg=target, target_position=target_nav_position)
                print(controller_env.last_event)
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
                rgb, depth_frame, mask_list, info_dict = controller_env.get_obs()
                input_dict = {"rgb_base_save_path": rgb_base_save_path,
                              "frontiers_base_save_path": frontiers_base_save_path,
                              "vis_mask_base_save_path": vis_mask_base_save_path,
                              "rgb_image": rgb,
                              "sem_map": semantic_map,
                              "info_dict": info_dict,
                              "save_id": save_id,
                              "vis_dict": vis_dict,
                              "long_clip": long_clip_model,
                              "long_clip_preprocess": long_clip_preprocess,
                              "image_description_text": image_description_text,
                              "third_rgb_base_path": third_rgb_base_path,
                              "controller_env": controller_env
                              }
                save_step_image(input_dict)
                save_id = save_id + 1

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

                            input_dict = {"rgb_base_save_path": rgb_base_save_path,
                                          "frontiers_base_save_path": frontiers_base_save_path,
                                          "vis_mask_base_save_path": vis_mask_base_save_path,
                                          "rgb_image": rgb,
                                          "sem_map": semantic_map,
                                          "info_dict": info_dict,
                                          "save_id": save_id,
                                          "vis_dict": vis_dict,
                                          "long_clip": long_clip_model,
                                          "long_clip_preprocess": long_clip_preprocess,
                                          "image_description_text": image_description_text,
                                          "third_rgb_base_path": third_rgb_base_path,
                                          "controller_env": controller_env
                                          }
                            save_step_image(input_dict)
                            save_id = save_id + 1


if __name__ == "__main__":
    main()