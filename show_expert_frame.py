import pickle
import cv2
import os
from tqdm import tqdm

base_path = "/home/wzy/workplace/llava_procthor/data/datasets/datadump_test_vis/images/eval_hssd"
# file_list = [x for x in os.listdir(base_path) if ".mp4" not in x]
file_list = ["102816756_656", "103997586_171030669_835"]

# 定义视频编码格式和输出对象

for file_one in tqdm(file_list):
    file_one_path = os.path.join(base_path, file_one)
    image_file_list = [x for x in os.listdir(file_one_path) if "planner" not in x]
    # 进行排序

    image_file_list.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
    output_video = os.path.join(base_path, file_one + ".mp4")

    first_frame = cv2.imread(os.path.join(file_one_path, image_file_list[0]))
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10  # 视频帧率
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image_file_one in image_file_list:
        image_path = os.path.join(file_one_path, image_file_one)
        image = cv2.imread(image_path)
        video.write(image)

    # 释放视频对象
    video.release()
    # cv2.destroyAllWindows()

