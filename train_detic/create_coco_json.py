import datetime
import json
import os
import re
import fnmatch
from PIL import Image
from itertools import groupby
import numpy as np
from skimage import measure
from pycococreatortools import pycococreatortools
from pycocotools import mask
from tqdm import tqdm
from object_constants import OBJECTS_LANG_PROCTHOR
import pickle

INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "zhenyu wu",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

ROOT_DIR = "/data1/wzy/vision_dataset/maskrcnn_v3"

def filter_for_jpeg(root, files):
    file_types = ['*.png', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    # contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = measure.find_contours(padded_binary_mask)
    # for contour_one in contours:
    #     print(np.asarray(contour_one).shape)
    contours_list = []
    for contour_one in contours:
        contours_list.append(np.asarray(contour_one))
    contours_list = np.concatenate(contours_list, axis=0)[None, :, :]
    # print(contours_list.shape)
    contours = np.subtract(contours_list, 1)
    # exit()
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def main():
    # 读取文件的path
    image_base_path = "./vision_dataset/maskrcnn_v3/rgb"
    label_base_path = "./vision_dataset/maskrcnn_v3/label"

    image_name_list = sorted(os.listdir(image_base_path))

    # 读取object_constants
    categories = []
    for categories_index, object_name_one in enumerate(OBJECTS_LANG_PROCTHOR):
        categories_id = categories_index + 1
        categories_one = {"id": categories_id,
                          "name": object_name_one,
                          "supercategory": "indoor"}
        categories.append(categories_one)
    assert len(categories) == len(OBJECTS_LANG_PROCTHOR)

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": categories,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # -------------- main loop ----------------------
    for image_name_one in tqdm(image_name_list[-4000:]):
        image_path_one = os.path.join(image_base_path, image_name_one)
        label_path_one = os.path.join(label_base_path, image_name_one.replace(".png", ".pkl"))
        # label_info = pickle.load(open(label_path_one))
        if not os.path.exists(label_path_one):
            continue
        with open(label_path_one, "rb") as f:
            label_info = pickle.load(f)

        image = Image.open(image_path_one)
        image_info = pycococreatortools.create_image_info(
            image_id, image_name_one, image.size)
        coco_output["images"].append(image_info)

        tolerance = 2
        # target["boxes"] = process_boxes
        # target["labels"] = process_labels
        # target["masks"] = process_masks
        # target["area"] = area
        # target["iscrowd"] = iscrowd
        bbox = label_info["boxes"]
        class_id_list = label_info["labels"]
        masks = label_info["masks"]
        area = label_info["area"]

        for label_one_index, class_id_one in enumerate(class_id_list):
            # 直接定死is_crowd
            class_id_one = class_id_one + 1
            category_info = {'id': int(class_id_one), 'is_crowd': 0}
            bbox_one = bbox[label_one_index]
            mask_one = masks[label_one_index]
            # area_one = area[label_one_index]

            x_min, y_min, x_max, y_max = bbox_one
            w = x_max - x_min
            h = y_max - y_min
            bounding_box = np.asarray([x_min, y_min, w, h])
            area_sum = np.sum(mask_one)
            if area_sum > 100:
                binary_mask_encoded = mask.encode(np.asfortranarray(mask_one.astype(np.uint8)))
                area = mask.area(binary_mask_encoded)
                segmentation = binary_mask_to_polygon(mask_one.astype(np.uint8), tolerance)
                annotation_info = {
                    "id": segmentation_id,
                    "image_id": image_id,
                    "category_id": category_info["id"],  # category_info["id"]
                    "iscrowd": category_info["is_crowd"],  # tegory_info["is_crowd"]
                    "area": area.tolist(),  # area.tolist()
                    "bbox": bounding_box.tolist(),  # bounding_box
                    "segmentation": segmentation,
                    "width": mask_one.shape[1],
                    "height": mask_one.shape[0],
                }
                coco_output["annotations"].append(annotation_info)
                segmentation_id = segmentation_id + 1

        image_id = image_id + 1

    with open('{}/instances_val_procthor.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()
