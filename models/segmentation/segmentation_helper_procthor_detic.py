import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import sys
import json
import torch
import copy
import skimage.morphology
import mss

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, './models/Detic/third_party/CenterNet2/')
from centernet.config import add_centernet_config
from models.Detic.detic.config import add_detic_config

from models.Detic.detic.predictor import VisualizationDemo
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.structures import Boxes, Instances
from collections import Counter, OrderedDict
from dataclasses import dataclass


@dataclass
class DeticConfig:
    config_file: str = "/home/wzy/workplace/llava_procthor/models/Detic/configs/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size_procthor.yaml"
    cpu: bool = False
    webcam: bool = False
    video_input: bool = False
    input: str = ""
    output: str = ""
    vocabulary: str = "procthor"
    custom_vocabulary: str = ""
    pred_all_class: bool = False
    confidence_threshold: float = 0.5
    opts = ["MODEL.WEIGHTS", "/home/wzy/workplace/llava_procthor/models/Detic/output/Detic/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size_procthor/model_final.pth"]


def setup_cfg(args, device=None, confidence_threshold=None):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    if device is not None:
        cfg.MODEL.DEVICE = device
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    if confidence_threshold is None:
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    else:
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/home/wzy/workplace/llava_procthor/models/Detic/configs/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size_procthor.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="procthor",
        choices=['lvis', 'openimages', 'objects365', 'coco', "procthor", "realscene", 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "/home/wzy/workplace/llava_procthor/models/Detic/output/Detic/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size_procthor/model_final.pth"],
        nargs=argparse.REMAINDER,
    )
    return parser


class SemgnetationHelperProcThorDetic:
    def __init__(self, agent):
        self.agent = agent
        args = self.args = self.agent.args
        total_cat2idx_path = self.args.total_cat2idx_procthor_path
        total_cat2idx = json.load(open(total_cat2idx_path))
        object_name_list = list(total_cat2idx.keys())

        if args.use_sem_seg:
            # Procthor物体
            # self.sem_seg_gpu = torch.device("cuda:" + str(args.sem_seg_gpu) if args.cuda else "cpu")
            # detic_args = get_parser().parse_args()
            detic_args = DeticConfig
            detic_cfg = setup_cfg(detic_args, device=args.sem_seg_gpu)
            self.detic_model = VisualizationDemo(detic_cfg, detic_args)

            self.objects2idx = {k: i for i, k in enumerate(object_name_list)}
            self.idx2object = {v: k for k, v in self.objects2idx.items()}

    def update_agent(self, agent):
        self.agent = agent
        self.args = self.agent.args

    # Visualize
    def get_torch_tensor_box(self):
        gt_boxes = self.agent.last_event.instance_detections2D
        array_list = []
        for k, v in gt_boxes.items():
            category = k.split('|')[0]
            if category in self.agent.total_cat2idx:
                array_list.append(v.tolist())
        return torch.tensor(array_list)

    # Visualize
    def visualize_sem(self, torch_tensor=None):

        visualizer = Visualizer(self.agent.last_event.frame, instance_mode=ColorMode.IMAGE)
        image_h, image_w, channel = self.agent.last_event.frame.shape
        if self.args.use_sem_seg:
            v_outputs = Instances(
                image_size=(image_h, image_w),
                scores=self.segmented_dict["scores"],
                pred_classes=np.array([self.idx2object[int(s.item())] for s in
                                       self.segmented_dict['classes']]),
                pred_masks=torch.tensor(self.segmented_dict['masks'].astype(bool)),
            )
            vis_output = visualizer.draw_instance_predictions(v_outputs.to("cpu"))
        else:
            instances = Instances((image_h, image_w)).to(torch.device("cpu"))
            boxes = Boxes(torch_tensor.to(torch.device("cpu")))
            instances.set('pred_boxes', boxes)
            vis_output = visualizer.draw_instance_predictions(predictions=instances)
        return vis_output.get_image()

    def visualize_sem_nav(self, rgb_image):
        visualizer = Visualizer(rgb_image, instance_mode=ColorMode.IMAGE)
        image_h, image_w, channel = rgb_image.shape
        v_outputs = Instances(
            image_size=(image_h, image_w),
            scores=self.segmented_dict["scores"],
            pred_classes=np.array([self.idx2object[int(s.item())] for s in
                                   self.segmented_dict['classes']]),
            pred_masks=torch.tensor(self.segmented_dict['masks'].astype(bool)),
        )
        vis_output = visualizer.draw_instance_predictions(v_outputs.to("cpu"))
        return vis_output.get_image()

    def segmentation_for_map(self):
        object_len = len(self.segmented_dict["scores"])
        if self.args.alfred_scene:
            semantic_seg = np.zeros((
                self.args.env_frame_height_alfred, self.args.env_frame_width_alfred,
                self.args.num_sem_categories_alfred))
        else:
            semantic_seg = np.zeros((
            self.args.env_frame_height, self.args.env_frame_width, self.args.num_sem_categories))
        for i in range(object_len):
            category = self.idx2object[self.segmented_dict["classes"][i].item()]
            v = self.segmented_dict["masks"][i]
            if not (category in self.agent.total_cat2idx):
                pass
            else:
                cat = self.agent.total_cat2idx[category]
                semantic_seg[:, :, cat] += v.astype("float")

        return semantic_seg

    def get_instance_mask_seg_alfworld_both(self):
        rgb = copy.deepcopy(self.agent.last_event.frame)
        image = rgb[:, :, [2, 1, 0]]
        predictions, visualized_output = self.detic_model.run_on_image(image)

        # 获取预测结果
        mask = predictions["instances"].pred_masks.to('cpu').numpy()
        pred_class = predictions["instances"].pred_classes.to('cpu').numpy()
        pred_boxes = predictions["instances"].pred_boxes.tensor.to("cpu").numpy()
        scores = predictions["instances"].scores.to('cpu').numpy()

        # 后处理
        desired_classes = []
        for cat_name in self.agent.total_cat2idx:
            if not (cat_name in ["None", "fake"]):
                if cat_name in self.objects2idx.keys():
                    object_class = self.objects2idx[cat_name]
                    desired_classes.append(object_class)
                else:
                    pass

        # Make the above into a dictionary
        self.segmented_dict = {"boxes": pred_boxes,
                               "classes": pred_class,
                               "masks": mask,
                               "scores": scores}

    def get_pred_dict(self, rgb):
        image = rgb[:, :, [2, 1, 0]]
        predictions, visualized_output = self.detic_model.run_on_image(image)

        # 获取预测结果
        mask = predictions["instances"].pred_masks.to('cpu').numpy()
        pred_class = predictions["instances"].pred_classes.to('cpu').numpy()
        pred_boxes = predictions["instances"].pred_boxes.tensor.to("cpu").numpy()
        scores = predictions["instances"].scores.to('cpu').numpy()

        # 后处理
        desired_classes = []
        for cat_name in self.agent.total_cat2idx:
            if not (cat_name in ["None", "fake"]):
                if cat_name in self.objects2idx.keys():
                    object_class = self.objects2idx[cat_name]
                    desired_classes.append(object_class)
                else:
                    pass

        # Make the above into a dictionary
        self.segmented_dict = {"boxes": pred_boxes,
                               "classes": pred_class,
                               "masks": mask,
                               "scores": scores}
        return self.segmented_dict

    # Get pseudo instance names as keys
    def segmentation_ground_truth(self):
        instance_mask = self.agent.last_event.instance_masks
        if self.args.alfred_scene:
            semantic_seg = np.zeros((
                self.args.env_frame_height_alfred, self.args.env_frame_width_alfred,
                self.args.num_sem_categories_alfred))
        else:
            semantic_seg = np.zeros((
                self.args.env_frame_height, self.args.env_frame_width, self.args.num_sem_categories))
        if self.args.ignore_categories:
            for k, v in instance_mask.items():
                category = k.split('|')[0]
                category_last = k.split('|')[-1]
                if 'Sliced' in category_last:
                    category = category + 'Sliced'

                if 'Sink' in category and 'SinkBasin' in category_last:
                    category = 'SinkBasin'
                if 'Bathtub' in category and 'BathtubBasin' in category_last:
                    category = 'BathtubBasin'

                if not (category in self.agent.total_cat2idx):
                    pass
                else:
                    cat = self.agent.total_cat2idx[category]
                    semantic_seg[:, :, cat] = v.astype('float')

        else:
            for k, v in instance_mask.items():
                category = k.split('|')[0]
                try:
                    cat = self.agent.total_cat2idx[category]
                    success = 1
                except:
                    success = 0

                if success == 1:
                    semantic_seg[:, :, cat] = v.astype('float')
                else:
                    pass

        return semantic_seg.astype('uint8')

    def get_sem_pred(self, rgb):
        if not (self.args.use_sem_seg):
            semantic_pred = self.segmentation_ground_truth()  # (300, 300, num_cat)
            if self.args.visualize or self.args.save_pictures:
                torch_tensor = self.get_torch_tensor_box()
                sem_vis = self.visualize_sem(torch_tensor)
                sem_vis = cv2.cvtColor(sem_vis, cv2.COLOR_RGB2BGR)

        else:
            # 1. First get dict
            self.get_instance_mask_seg_alfworld_both()

            # 2. Get segmentation for map making
            semantic_pred = self.segmentation_for_map()

            # 3. visualize (get sem_vis)
            if self.args.visualize or self.args.save_pictures:
                sem_vis = self.visualize_sem()
                sem_vis = cv2.cvtColor(sem_vis, cv2.COLOR_RGB2BGR)

        if self.args.visualize:
            sem_vis = cv2.resize(sem_vis, (500, 500))
            cv2.imshow("Sem", sem_vis)
            cv2.waitKey(1)

        if self.args.save_pictures:
            cv2.imwrite(self.agent.picture_folder_name + "Sem/" + "Sem_" + str(self.agent.steps_taken) + ".png", sem_vis)
        semantic_pred = semantic_pred.astype(np.float32)

        return semantic_pred

    def get_instance_mask_from_obj_type(self, object_type):
        mask = np.zeros((300, 300))
        for k, v in self.agent.last_event.instance_masks.items():
            category = k.split('|')[0]
            category_last = k.split('|')[-1]
            if category in self.agent.cat_equate_dict:
                category = self.agent.cat_equate_dict[category]
            if 'Sliced' in category_last:
                category = category + 'Sliced'
            if 'Sink' in category and 'SinkBasin' in category_last:
                category = 'SinkBasin'
            if 'Bathtub' in category and 'BathtubBasin' in category_last:
                category = 'BathtubBasin'
            if category == object_type:
                mask = v
        if np.sum(mask) == 0:
            mask = None
        return mask

    def get_instance_mask_from_obj_type_largest(self, object_type):
        mask = np.zeros((300, 300))
        max_area = 0
        for k, v in self.agent.last_event.instance_masks.items():
            category = k.split('|')[0]
            category_last = k.split('|')[-1]
            if category in self.agent.cat_equate_dict:
                category = self.agent.cat_equate_dict[category]
            if 'Sliced' in category_last:
                category = category + 'Sliced'
            if 'Sink' in category and 'SinkBasin' in category_last:
                category = 'SinkBasin'
            if 'Bathtub' in category and 'BathtubBasin' in category_last:
                category = 'BathtubBasin'
            if category == object_type:
                if np.sum(v) >= max_area:
                    max_area = np.sum(v)
                    mask = v
        if np.sum(mask) == 0:
            mask = None
        return mask

    def sem_seg_get_instance_mask_from_obj_type_largest_only(self, object_type):
        mask = np.zeros((300, 300))
        object_len = len(self.segmented_dict["scores"])
        max_area = -1

        if object_type in self.agent.cat_equate_dict:
            object_type = self.agent.cat_equate_dict[object_type]

        if object_type in self.objects2idx.keys():
            # Get the highest score
            for i in range(object_len):
                category = self.idx2object[self.segmented_dict["classes"][i].item()]
                if category == object_type:
                    v = self.segmented_dict["masks"][i]
                    score = self.segmented_dict["scores"][i]
                    area = np.sum(self.segmented_dict["masks"][i])
                    max_area = max(area, max_area)
                    if max_area == area:
                        mask = v.astype("float")

        if np.sum(mask) == 0:
            mask = None
        return mask

    def maskDistance(self, mask, ref_pt):
        cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ref_xy = (ref_pt[1], ref_pt[0])
        dist = cv2.pointPolygonTest(cnt[0], ref_xy, True)  # positive: inside the mask, negative: outside
        return dist

    def sem_seg_get_instance_mask_from_obj_type(self, object_type):
        mask = np.zeros((300, 300))
        object_len = len(self.segmented_dict["scores"])
        max_score = -1

        if object_type in self.agent.cat_equate_dict:
            object_type = self.agent.cat_equate_dict[object_type]

        if object_type in self.objects2idx.keys():
            # Get the highest score
            for i in range(object_len):
                category = self.idx2object[self.segmented_dict["classes"][i].item()]
                if category == object_type:
                    v = self.segmented_dict["masks"][i]
                    score = self.segmented_dict["scores"][i]
                    max_score = max(score, max_score)
                    if score == max_score:
                        mask = v.astype(np.float32)

        if np.sum(mask) == 0:
            mask = None
        return mask

    def get_target(self, interact_mask, mask_px_sample=1):
        if interact_mask is None:
            return ""
        instance_segs = np.array(self.agent.last_event.instance_segmentation_frame)
        color_to_object_id = self.agent.last_event.color_to_object_id

        # get object_id for each 1-pixel in the interact_mask
        nz_rows, nz_cols = np.nonzero(interact_mask)
        instance_counter = Counter()
        for i in range(0, len(nz_rows), mask_px_sample):
            x, y = nz_rows[i], nz_cols[i]
            instance = tuple(instance_segs[x, y])
            instance_counter[instance] += 1

        # iou scores for all instances
        iou_scores = {}
        for color_id, intersection_count in instance_counter.most_common():
            union_count = np.sum(np.logical_or(np.all(instance_segs == color_id, axis=2), interact_mask.astype(bool)))
            iou_scores[color_id] = intersection_count / float(union_count)
        iou_sorted_instance_ids = list(OrderedDict(sorted(iou_scores.items(), key=lambda x: x[1], reverse=True)))

        # get the most common object ids ignoring the object-in-hand
        inv_obj = self.agent.last_event.metadata['inventoryObjects'][0]['objectId'] \
            if len(self.agent.last_event.metadata['inventoryObjects']) > 0 else None
        all_ids = [color_to_object_id[color_id] for color_id in iou_sorted_instance_ids
                   if color_id in color_to_object_id and color_to_object_id[color_id] != inv_obj]

        # print all ids

        # print instance_ids
        instance_ids = [inst_id for inst_id in all_ids if inst_id is not None]
        instance_ids = self.agent.prune_by_any_interaction(instance_ids)
        if self.args.ground_truth_segmentation:
            pass
        else:
            instance_ids_new = self.agent.prune_by_any_interaction(instance_ids)
            for instance_id in instance_ids:
                if 'Sink' in instance_id:
                    instance_ids_new.append(instance_id)
        # prune invalid instances like floors, walls, etc.
        if len(instance_ids) == 0:
            return ""
        else:
            target_instance_id = instance_ids[0]
            return target_instance_id

    def _visible_objects(self):
        objects_meta_data = self.agent.last_event.metadata["objects"]
        objectIDs = []
        for i, obj_dict in enumerate(objects_meta_data):
            if obj_dict['visible']:
                objectID = obj_dict['objectId']
                objectIDs.append(objectID)
        return objectIDs


class HelperFunctions:
    def diff_two_frames(f1, f2):
        diff1 = np.where(f1[:, :, 0] != f2[:, :, 0])
        diff2 = np.where(f1[:, :, 1] != f2[:, :, 1])
        diff3 = np.where(f1[:, :, 2] != f2[:, :, 2])
        diff_mask = np.zeros((300, 300));
        return_mask = np.zeros((300, 300))
        diff_mask[diff1] = 1.0
        diff_mask[diff2] = 1.0
        diff_mask[diff3] = 1.0

        # Divide diff mask into 2 regions with scikit image
        connected_regions = skimage.morphology.label(diff_mask, connectivity=2)
        unique_labels = [i for i in range(0, np.max(connected_regions) + 1)]
        lab_area = {lab: 0 for lab in unique_labels}
        min_ar = 10000000
        smallest_lab = None
        for lab in unique_labels:
            wheres = np.where(connected_regions == lab)
            lab_area[lab] = np.sum(len(wheres[0]))
            if lab_area[lab] > 100:
                min_ar = min(lab_area[lab], min_ar)
                if min_ar == lab_area[lab]:
                    smallest_lab = lab

        if not (smallest_lab) is None:
            return_mask[np.where(connected_regions == smallest_lab)] = 1

        # return_mask[np.where(picked_up_mask)] = 0.0
        return return_mask

        return None, vis_output.get_image()

    def diff_two_frames_max(f1, f2):
        diff1 = np.where(f1[:, :, 0] != f2[:, :, 0])
        diff2 = np.where(f1[:, :, 1] != f2[:, :, 1])
        diff3 = np.where(f1[:, :, 2] != f2[:, :, 2])
        diff_mask = np.zeros((300, 300));
        return_mask = np.zeros((300, 300))
        diff_mask[diff1] = 1.0
        diff_mask[diff2] = 1.0
        diff_mask[diff3] = 1.0

        # Divide diff mask into 2 regions with scikit image
        connected_regions = skimage.morphology.label(diff_mask, connectivity=2)
        unique_labels = [i for i in range(0, np.max(connected_regions) + 1)]
        lab_area = {lab: 0 for lab in unique_labels}
        max_ar = 0
        largest_lab = None
        for lab in unique_labels:
            wheres = np.where(connected_regions == lab)
            lab_area[lab] = np.sum(len(wheres[0]))
            if lab_area[lab] > 100:
                max_ar = max(lab_area[lab], max_ar)
                if max_ar == lab_area[lab]:
                    largest_lab = lab

        if not (largest_lab) is None:
            return_mask[np.where(connected_regions == largest_lab)] = 1

        return return_mask