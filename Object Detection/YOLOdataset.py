import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from PIL import Image
from YOLOmodel import *
import albumentations as alb


class YOLOdataset(Dataset):
    def __init__(self, csv_file, image_dir, label_dir, anchors,
                 image_size=416, grid_sizes=[13, 26, 52],
                 num_classes=20, transform=None):

        self.labels_list = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.num_classes = num_classes
        self.grid_sizes = grid_sizes
        self.anchors = torch.Tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.iou_threshold = 0.5

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        # Getting the label path
        label_path = os.path.join(self.label_dir, self.label_list.iloc[idx, 1])
        # We are applying roll to move class label to the last column
        # 5 columns: x, y, width, height, class_label
        bboxes = np.roll(np.loadtxt(fname=label_path,
                         delimiter=" ", ndmin=2), 4, axis=1).tolist()

        # Getting the image path
        img_path = os.path.join(self.image_dir, self.label_list.iloc[idx, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        # Albumentations augmentations
        if self.transform:
            augs = self.transform(image=image, bboxes=bboxes)
            image = augs["image"]
            bboxes = augs["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        # target : [probabilities, x, y, width, height, class_label]
        targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6))
                   for s in self.grid_sizes]

        # Identify anchor box and cell for each bounding box
        for box in bboxes:
            # Calculate iou of bounding box with anchor boxes
            iou_anchors = iou(torch.tensor(box[2:4]),
                              self.anchors,
                              is_pred=False)
            # Selecting the best anchor box
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box

            # At each scale, assigning the bounding box to the
            # best matching anchor box
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                # Identifying the grid size for the scale
                s = self.grid_sizes[scale_idx]

                # Identifying the cell to which the bounding box belongs
                i, j = int(s * y), int(s * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                # Check if the anchor box is already assigned
                if not anchor_taken and not has_anchor[scale_idx]:

                    # Set the probability to 1
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # Calculating the center of the bounding box relative
                    # to the cell
                    x_cell, y_cell = s * x - j, s * y - i

                    # Calculating the width and height of the bounding box
                    # relative to the cell
                    width_cell, height_cell = (width * s, height * s)

                    # Idnetify the box coordinates
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell,
                         height_cell]
                    )

                    # Assigning the box coordinates to the target
                    targets[scale_idx][anchor_on_scale,
                                       i, j, 1:5] = box_coordinates

                    # Assigning the class label to the target
                    targets[scale_idx][anchor_on_scale,
                                       i, j, 5] = int(class_label)

                    # Set the anchor box as assigned for the scale
                    has_anchor[scale_idx] = True

                # If the anchor box is already assigned, check if the
                # IoU is greater than the threshold
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    # Set the probability to -1 to ignore the anchor box
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        # Return the image and the target
        return image, tuple(targets)


# Transform for training
def train_transform(image_size):
    train_transform = alb.Compose(
        [
            # Rescale an image so that maximum side is equal to image_size
            alb.LongestMaxSize(max_size=image_size),
            # Pad remaining areas with zeros
            alb.PadIfNeeded(
                min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT
            ),
            # Random color jittering
            alb.ColorJitter(
                brightness=0.5, contrast=0.5,
                saturation=0.5, hue=0.5, p=0.5
            ),
            # Flip the image horizontally
            alb.HorizontalFlip(p=0.5),
            # Normalize the image
            alb.Normalize(
                mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
            ),
            # Convert the image to PyTorch tensor
            alb.ToTensorV2()
        ],
        # Augmentation for bounding boxes
        bbox_params=alb.BboxParams(
            format="yolo",
            min_visibility=0.4,
            label_fields=[]
        )
    )
    return train_transform

# Transform for testing


def test_transform(image_size):
    test_transform = alb.Compose(
        [
            # Rescale an image so that maximum side is equal to image_size
            alb.LongestMaxSize(max_size=image_size),
            # Pad remaining areas with zeros
            alb.PadIfNeeded(
                min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT
            ),
            # Normalize the image
            alb.Normalize(
                mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
            ),
            # Convert the image to PyTorch tensor
            alb.ToTensorV2()
        ],
        # Augmentation for bounding boxes
        bbox_params=alb.BboxParams(
            format="yolo",
            min_visibility=0.4,
            label_fields=[]
        )
    )
    return test_transform
