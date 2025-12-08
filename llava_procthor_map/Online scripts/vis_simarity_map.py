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
base_save_path = "./visualization/visualization_results_test_prediction_final_v3/house-2532_similarity_map"
vis_base_path = "./visualization/visualization_results_test_prediction_final_v2/house-2532_frontiers_v2"
if not os.path.exists(base_save_path):
    os.makedirs(base_save_path)

file_list = os.listdir(base_path)
simarity_map_file_list = [file_one for file_one in file_list if ".pkl" in file_one]
save_list = [375, 376, 374]
for simarity_file_one in tqdm(simarity_map_file_list):
    file_number = int(simarity_file_one.split("_")[1].split(".")[0])
    # if file_number in save_list:
    if file_number > 0:

        simarity_one_path = os.path.join(base_path, simarity_file_one)
        save_path_one = os.path.join(base_save_path, simarity_file_one.replace(".pkl", ".png"))
        with open(simarity_one_path, "rb") as f:
            frontiers_dict = pickle.load(f)
        simarity_map = frontiers_dict["similarity_map"]

        similarity_map = frontiers_dict["similarity_map"]
        similarity_map = ((similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min()))
        filer_index = similarity_map < 0.1
        # cam = np.uint8(255 * similarity_map)
        # cam = np.uint8(cv2.resize(cam, (similarity_map.shape[0], similarity_map.shape[1])))
        # #
        # heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        # heatmap_pil = Image.fromarray(heatmap)
        # heatmap_pil = heatmap_pil.convert("RGBA")
        # heatmap_pil = np.asarray(heatmap_pil).copy()
        # heatmap_pil[filer_index, :] = [255, 255, 255, 0]
        # heatmap_pil = Image.fromarray(heatmap_pil)
        # heatmap_pil = Image.open(save_path_one)
        # vis_path_one = os.path.join(vis_base_path, simarity_file_one.replace(".pkl", ".png"))
        # image_pil = Image.open(vis_path_one)
        # result = Image.blend(image_pil, heatmap_pil, alpha=0.5)
        # print(np.max(simarity_map))
        # print(np.min(simarity_map))
        plt.figure()
        plt.axis('off')  # 去坐标轴
        plt.xticks([])  # 去刻度
        plt.yticks([])  # 去刻度

        plt.imshow(similarity_map, cmap='jet')

        plt.savefig(save_path_one, dpi=800, bbox_inches='tight', pad_inches=-0.01)
        plt.close()
        # heatmap_pil.save(save_path_one)


