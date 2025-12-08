import os
import numpy as np
import torch
import random
import json
import argparse
from PIL import Image

import cv2

from alfworld.agents.detector.engine import train_one_epoch, evaluate
import alfworld.agents.detector.utils as utils
import torchvision
import alfworld.agents.detector.transforms as T
from alfworld.agents.detector.mrcnn import get_model_instance_segmentation, load_pretrained_model
import pickle
import sys
import alfworld.gen.constants as constants

MIN_PIXELS = 100


def get_object_classes():
    total_cat2idx_path = "./utils/total_cat2idx.json"
    total_cat2idx = json.load(open(total_cat2idx_path))
    object_classes_list = list(total_cat2idx.keys())
    return object_classes_list


class AlfredDataset(object):
    def __init__(self, root, transforms, args):
        self.root = root
        self.transforms = transforms
        self.args = args
        self.object_classes = get_object_classes()

        # load all image files, sorting them to
        # ensure that they are aligned
        self.get_data_files(root, balance_scenes=args.balance_scenes)

    def get_data_files(self, root, balance_scenes=False):
        if balance_scenes:
            image_base_path = os.path.join(root, "rgb")
            label_base_path = os.path.join(root, "label")
            image_name_list = sorted(os.listdir(image_base_path))
            image_path_list = []
            label_path_list = []
            for image_name_one in image_name_list:
                label_name_one = image_name_one.replace(".png", ".pkl")
                image_path_one = os.path.join(image_base_path, image_name_one)
                label_path_one = os.path.join(label_base_path, label_name_one)
                if not os.path.exists(label_path_one):
                    continue
                image_path_list.append(image_path_one)
                label_path_list.append(label_path_one)

            self.imgs = image_path_list
            self.labels = label_path_list

        else:
            self.imgs = [os.path.join(root, "images", f) for f in
                         list(sorted(os.listdir(os.path.join(root, "images"))))]
            self.masks = [os.path.join(root, "masks", f) for f in list(sorted(os.listdir(os.path.join(root, "masks"))))]
            self.metas = [os.path.join(root, "meta", f) for f in list(sorted(os.listdir(os.path.join(root, "meta"))))]

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]
        label_path = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        with open(label_path, "rb") as f:
            label_dict = pickle.load(f)

        masks = label_dict["masks"]
        boxes = label_dict["boxes"]
        labels = label_dict["labels"]

        iscrowd = torch.zeros(len(masks), dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main(args):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = len(get_object_classes()) + 1
    # use our dataset and defined transformations
    dataset = AlfredDataset(args.data_path, get_transform(train=True), args)
    dataset_test = AlfredDataset(args.data_path, get_transform(train=False), args)

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    indices = list(range(len(dataset)))
    dataset = torch.utils.data.Subset(dataset, indices[:-4000])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-4000:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    if args.load_model:
        model = load_pretrained_model(args.load_model)
    else:
        model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
        momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        step_size=3,
        gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)
        # save model
        model_path = os.path.join(args.save_path, "%s_%03d.pth" % (args.save_name, epoch))
        torch.save(model.state_dict(), model_path)
        print("Saving %s" % model_path)

    print("Done training!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./vision_dataset/maskrcnn_v3/")
    parser.add_argument("--save_path", type=str, default="./output_model/maskrcnn_v3")
    parser.add_argument("--object_types", choices=["objects", "receptacles", "all"], default="all")
    parser.add_argument("--save_name", type=str, default="mrcnn_alfred_objects")
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.005)

    parser.add_argument("--balance_scenes", action='store_true')
    parser.add_argument("--kitchen_factor", type=float, default=1.0)
    parser.add_argument("--living_factor", type=float, default=1.0)
    parser.add_argument("--bedroom_factor", type=float, default=1.0)
    parser.add_argument("--bathroom_factor", type=float, default=1.0)

    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)