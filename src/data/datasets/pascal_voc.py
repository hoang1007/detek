from typing import Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import VOCDetection

from src.structures import DataSample

from .base_dataset import BaseDataset

VOC_CLASSES = [
    "__background__",  # always index 0
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class VOCDataset(BaseDataset):
    def __init__(
        self,
        img_size: Tuple[int, int] = (640, 640),
        min_box_size: int = 0,
        root="data",
        year="2007",
        image_set="train",
    ):
        super().__init__(VOC_CLASSES)
        if image_set == "train":
            self.transform = A.Compose(
                (
                    A.RandomBrightnessContrast(p=0.2),
                    A.HorizontalFlip(p=0.5),
                    A.LongestMaxSize(max_size=max(img_size), p=1.0),
                    A.RandomSizedBBoxSafeCrop(
                        width=img_size[0], height=img_size[1], erosion_rate=0.0, p=0.2
                    ),
                    A.PadIfNeeded(
                        min_height=img_size[1],
                        min_width=img_size[0],
                        value=0,
                        p=1.0,
                    ),
                    ToTensorV2(),
                ),
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
            )
        else:
            self.transform = A.Compose(
                (
                    A.PadIfNeeded(
                        min_height=None,
                        min_width=None,
                        pad_height_divisor=32,
                        pad_width_divisor=32,
                        value=0,
                        p=1.0,
                    ),
                    ToTensorV2(),
                ),
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
            )

        self.min_box_size = min_box_size
        self._data = VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=True,
        )

    def __getitem__(self, idx):
        img, info = self._data[idx]
        img = np.array(img)

        gt_boxes, labels = [], []

        for obj_info in info["annotation"]["object"]:
            label_name = obj_info["name"]
            bndbox = [int(k) for k in obj_info["bndbox"].values()]

            width = bndbox[2] - bndbox[0]
            height = bndbox[3] - bndbox[1]
            if width < self.min_box_size or height < self.min_box_size:
                continue

            gt_boxes.append(bndbox)
            labels.append(self.get_class_idx(label_name))

        transformed = self.transform(image=img, bboxes=gt_boxes, labels=labels)

        img = transformed["image"]

        gt_boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        labels = torch.tensor(transformed["labels"], dtype=torch.long)

        return DataSample(img, gt_boxes, labels)

    def __len__(self):
        return len(self._data)
