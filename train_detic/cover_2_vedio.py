import cv2
import os
from tqdm import tqdm

vis_base_path = "./output_img"
video_rgb_save_path = "./vis_video/demo_task_6.mp4"
video_ori_rgb_save_path = "./vis_video/demo_task_6_ori.mp4"
rgb_image_path = "/home/wzy/workplace/auto_anno/GroundingDINO/data/task_0508/train/rgb"

rgb_file_list = os.listdir(vis_base_path)
rgb_file_list.sort(key=lambda x: int(x.split(".")[0]))
ori_file_list = os.listdir(rgb_image_path).sort(key=lambda x: int(x.split(".")[0]))



rgb_frame = cv2.imread(os.path.join(vis_base_path, rgb_file_list[0]))
height_rgb, width_rgb, layers_rgb = rgb_frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置编解码器为mp4v
video_rgb = cv2.VideoWriter(video_rgb_save_path, fourcc, 5, (width_rgb, height_rgb))
video_ori_rgb = cv2.VideoWriter(video_ori_rgb_save_path, fourcc, 5, (width_rgb, height_rgb))
for rgb_name_one in tqdm(rgb_file_list[1980:2477]):
    rgb_path_one = os.path.join(vis_base_path, rgb_name_one)
    video_rgb.write(cv2.imread(rgb_path_one))

    rgb_path_one = os.path.join(rgb_image_path, rgb_name_one)
    video_ori_rgb.write(cv2.imread(rgb_path_one))

cv2.destroyAllWindows()
video_rgb.release()
video_ori_rgb.release()


