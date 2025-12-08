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

base_path = "./visualization/visualization_results_test_prediction_final_v3/house-2532_frontiers"
sim_save_path = "./visualization/visualization_results_test_prediction_final_v3/house-2532_similarity_map"
vis_base_path = "./visualization/visualization_results_test_prediction_final_v2/house-2532_frontiers_v2"
base_save_path = "./visualization/visualization_results_test_prediction_final_v3/house-2532_similarity_map_v2"
if not os.path.exists(base_save_path):
    os.makedirs(base_save_path)

# 分别读取文件
similarity_map_list = os.listdir(sim_save_path)
for similarity_map_name_one in tqdm(similarity_map_list):
    if "522" not in similarity_map_name_one:
        continue
    similarity_map_path_one = os.path.join(sim_save_path, similarity_map_name_one)
    vis_map_path_one = os.path.join(vis_base_path, similarity_map_name_one)
    save_path_one = os.path.join(base_save_path, similarity_map_name_one)
    similarity_map_path = os.path.join(base_path, similarity_map_name_one.replace(".png", ".pkl"))
    with open(similarity_map_path, "rb") as f:
        frontiers_dict = pickle.load(f)
    similarity_map = frontiers_dict["similarity_map"]

    similarity_map = ((similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min()))
    filer_index = similarity_map < 0.4
    similarity_map = similarity_map * 10

    plt.figure()
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去刻度
    plt.yticks([])  # 去刻度

    plt.imshow(similarity_map, cmap='jet')

    plt.savefig("0.png", dpi=800, bbox_inches='tight', pad_inches=-0.01)
    plt.close()

    # similarity_map = Image.fromarray(similarity_map.astype(np.uint8))
    similarity_map = Image.open("0.png")
    similarity_map = similarity_map.resize((600, 600))
    similarity_map = similarity_map.convert("RGBA")
    similarity_map = np.asarray(similarity_map).copy()
    similarity_map[filer_index, :] = [255, 255, 255, 0]
    similarity_map = Image.fromarray(similarity_map)

    similarity_map.save("11.png")

    vis_map = Image.open(vis_map_path_one)
    vis_map = vis_map.resize((600, 600))

    vis_map = vis_map.convert("RGBA")

    # width, height = similarity_map.size  # 长度和宽度
    # for i in range(0, width):  # 遍历所有长度的点
    #     for j in range(0, height):  # 遍历所有宽度的点
    #         data = similarity_map.getpixel((i, j))  # 获取一个像素
    #         if (data[2] < 205):  # RGBA都是255，改成透明色
    #             similarity_map.putpixel((i, j), (255, 255, 255, 0))

    print(vis_map.size)
    print(similarity_map.size)
    result = Image.blend(vis_map, similarity_map, alpha=0.6)

    plt.figure()
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去刻度
    plt.yticks([])  # 去刻度

    plt.imshow(result)

    plt.savefig("2.png", dpi=800, bbox_inches='tight', pad_inches=-0.01)
    plt.close()