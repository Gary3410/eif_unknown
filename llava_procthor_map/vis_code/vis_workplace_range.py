import os
import cv2
from tqdm import tqdm

base_save_file = "/data1/wzy/vision_dataset/test_open_find_object"
rgb_name_list = os.listdir(base_save_file)
for rgb_name_one in tqdm(rgb_name_list):
    rgb_one_path = os.path.join(base_save_file, rgb_name_one)
    image = cv2.imread(rgb_one_path)
    cv2.rectangle(image, (50, 50), (550, 550),
        color=(0, 0, 255), thickness=2)
    cv2.imwrite(os.path.join(base_save_file, rgb_one_path), image)
