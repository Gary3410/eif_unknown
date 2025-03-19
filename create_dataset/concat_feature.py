import cv2
import numpy as np
from tqdm import tqdm
import blosc
import os


def save_blosc_file(path, sem_feature):
    sem_feature = sem_feature.astype(np.float16)
    compressed_array = blosc.compress(sem_feature.tobytes())
    # Save the compressed data to a file
    with open(path, 'wb') as f:
        f.write(compressed_array)


def load_blosc_file(path, feature_shape=(256, 1024)):
    with open(path, 'rb') as f:
        loaded_compressed_array = f.read()
        # # Decompress the data
    decompressed_array = np.frombuffer(blosc.decompress(loaded_compressed_array), dtype=np.float16)
    decompressed_array = decompressed_array.reshape(feature_shape)
    feature = np.asarray(decompressed_array)
    return feature


def main():
    img_feature_base_path = "./vision_dataset/nav_inter_dataset/vision_feature_blosc_2D_only"
    point_feature_base_path = "./vision_dataset/nav_inter_dataset/point_clip_feature"
    save_base_feature_path = "./vision_dataset/nav_inter_dataset/vision_feature_v3"

    # main loop
    img_feature_list = os.listdir(img_feature_base_path)
    print(len(img_feature_list))
    for img_file_index, img_feature_name_one in enumerate(tqdm(img_feature_list)):
        img_feature_path_one = os.path.join(img_feature_base_path, img_feature_name_one)
        point_feature_path_one = os.path.join(point_feature_base_path, img_feature_name_one)
        save_path_one = os.path.join(save_base_feature_path, img_feature_name_one)

        # load feature
        img_feature_one = load_blosc_file(img_feature_path_one, feature_shape=(2048, 1024))
        point_feature_one = load_blosc_file(point_feature_path_one, feature_shape=(10, 1024))

        fusion_feature = np.concatenate((img_feature_one[:246, :], point_feature_one), axis=0)

        save_blosc_file(save_path_one, fusion_feature)

if __name__ == "__main__":
    main()