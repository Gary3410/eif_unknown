import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import blosc
from einops import asnumpy, repeat
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
import pickle
import json
import cv2

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


def load_blosc_file(path, shape=(97, 800, 800)):
    # # Read the compressed data from the file

    with open(path, 'rb') as f:
        loaded_compressed_array = f.read()

    # # Decompress the data
    decompressed_array = np.frombuffer(blosc.decompress(loaded_compressed_array), dtype=np.float16)
    decompressed_array = decompressed_array.reshape(shape)

    return decompressed_array

def main():

    # base path
    base_path = "./visualization/visualization_results_test_prediction_final_v3/house-9817_frontiers"
    frontiers_save_path = "./visualization/visualization_results_test_prediction_final_v3/house-9817_frontiers_v2"
    space_dataset_file_path = "./data/spaced_parse_instruction_hard_v13_val.json"
    house_key_id = "92_9817"
    total_cat2idx = json.load(open("./utils/total_cat2idx.json"))
    total_idx2cat = {}
    for k, v in total_cat2idx.items():
        total_idx2cat[v] = k
    # 大物体
    static_object_list = ["CounterTop", "Sofa", "DiningTable"]

    space_dataset_file = json.load(open(space_dataset_file_path))
    house_id_dict = {}
    for index, space_dataset_one in enumerate(space_dataset_file):
        object_list_one = []
        output_one = space_dataset_one["output"]
        house_id = int(space_dataset_one["house_id"])
        house_id_one = str(index) + "_" + str(house_id)
        for action_dict_one in output_one:
            arg = action_dict_one["arg"]
            if arg != " ":
                object_list_one.append(arg)
        object_list_one = list(set(object_list_one))
        house_id_dict[house_id_one] = object_list_one

    if not os.path.exists(frontiers_save_path):
        os.makedirs(frontiers_save_path)

    # 读取sem_map字典
    map_mask_dict_list = [file_one for file_one in os.listdir(base_path) if ".pkl" in file_one]
    map_mask_dict_list.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))

    for mask_dict_one in tqdm(map_mask_dict_list):
        mask_dict_path = os.path.join(base_path, mask_dict_one)
        with open(mask_dict_path, "rb") as f:
            frontiers_dict = pickle.load(f)
        frontiers_save_path_one = os.path.join(frontiers_save_path, mask_dict_one.replace(".pkl", ".png"))

        ori_map_name = frontiers_dict["sem_map"].split("/")[-1]
        ori_map_path = os.path.join(base_path, ori_map_name)
        sem_map = load_blosc_file(ori_map_path, shape=(97, 800, 800)).copy()

        # 后处理
        object_list = house_id_dict[house_key_id]
        if object_list is not None:
            static_object_list.extend(object_list)
            save_object_list = list(set(static_object_list))
        else:
            save_object_list = static_object_list
        save_object_index = [total_cat2idx[name_one] + 1 for name_one in save_object_list]
        for map_index in range(sem_map.shape[0])[2:]:
            if map_index not in save_object_index:
                sem_map[map_index] = np.zeros([sem_map.shape[1], sem_map.shape[2]])

        robot_mask = None
        frontiers_mask = None
        nav_path_mask = None
        if "robot_mask" in frontiers_dict:
            robot_mask = frontiers_dict["robot_mask"]
            if not isinstance(robot_mask, np.ndarray):
                robot_mask = robot_mask.numpy()
        if "vis_frontiers_mask" in frontiers_dict:
            frontiers_mask = frontiers_dict["vis_frontiers_mask"]
            if not isinstance(frontiers_mask, np.ndarray):
                frontiers_mask = frontiers_mask.numpy()
        if "nav_path_mask" in frontiers_dict:
            nav_path_mask = frontiers_dict["nav_path_mask"]
            if not isinstance(nav_path_mask, np.ndarray):
                nav_path_mask = nav_path_mask.numpy()

        vis_mask = visualize_map(sem_map)

        if frontiers_mask is not None:
            vis_mask = visualize_frontiers_mask(vis_mask, frontiers_mask)
        if nav_path_mask is not None:
            vis_mask = visualize_frontiers_mask(vis_mask, nav_path_mask, color_index=180)
        if robot_mask is not None:
            vis_mask = visualize_frontiers_mask(vis_mask, robot_mask, color_index=255, channel_index=1)

        similarity_map = frontiers_dict["similarity_map"]

        heatmap = cv2.applyColorMap(similarity_map, cv2.COLORMAP_JET)
        heatmap_pil = Image.fromarray(heatmap)
        image_pil = Image.fromarray(np.uint8(image))
        result = Image.blend(image_pil, heatmap_pil, alpha=0.5)

        plt.figure()
        plt.axis('off')  # 去坐标轴
        plt.xticks([])  # 去刻度
        plt.yticks([])  # 去刻度

        plt.imshow(vis_mask)
        # plt.savefig(frontiers_save_path_one, dpi=800)
        plt.savefig(frontiers_save_path_one, dpi=800, bbox_inches='tight', pad_inches=-0.01)
        # plt.savefig(frontiers_save_path_one, dpi=500)
        # plt.savefig(frontiers_save_path)
        plt.close()

if __name__ == "__main__":
    main()