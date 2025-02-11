"""
COCO dataset module.

This module defines a CocoDetection dataset class along with helper functions
to convert COCO polygon annotations to segmentation masks.
"""

from pathlib import Path
import copy
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
import datasets.transforms as T
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import numpy as np
import contextlib
import os
from util.misc import all_gather


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder: str, ann_file: str, transforms, return_masks: bool):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx: int):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        # Wrap annotations into a dictionary.
        target = {'image_id': image_id, 'annotations': copy.deepcopy(target)}
        # Convert polygons to masks and adjust target (including label conversion).
        img, target = self.prepare(img, target)
        # Optionally apply additional transforms.
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        # Uncomment the following line to debug and verify that labels are now 0-indexed.
        # print("Unique target labels:", torch.unique(target.get("labels", torch.tensor([]))))
        return img, target


def convert_coco_poly_to_mask(segmentations, height: int, width: int):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if mask.ndim < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8).any(dim=2)
        masks.append(mask)
    return torch.stack(masks, dim=0) if masks else torch.zeros((0, height, width), dtype=torch.uint8)


class ConvertCocoPolysToMask:
    def __init__(self, return_masks: bool = False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size
        image_id = torch.tensor([target["image_id"]])
        # Filter annotations: only use those with iscrowd == 0.
        anno = [obj for obj in target["annotations"] if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        
        # Convert bounding boxes from [x, y, w, h] to [x_min, y_min, x_max, y_max].
        boxes = torch.tensor([obj["bbox"] for obj in anno], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        
        # *** Convert category IDs from 1-indexed to 0-indexed.
        classes = torch.tensor([obj["category_id"] - 1 for obj in anno], dtype=torch.int64)
        
        masks = (convert_coco_poly_to_mask([obj["segmentation"] for obj in anno], h, w)
                 if self.return_masks else None)

        # Keep only valid boxes (nonzero area).
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        
        target = {
            "boxes": boxes[keep],
            "labels": classes[keep],
            "image_id": image_id,
            "area": torch.tensor([obj["area"] for obj in anno])[keep],
            "iscrowd": torch.tensor([obj.get("iscrowd", 0) for obj in anno])[keep],
            "orig_size": torch.tensor([h, w]),
            "size": torch.tensor([h, w]),
        }
        
        if self.return_masks:
            target["masks"] = masks[keep]
        
        return image, target


def make_coco_transforms(image_set: str):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=1333),
            T.RandomSizeCrop(384, 600),
            normalize,
        ])
    elif image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    else:
        raise ValueError(f'Unknown dataset split: {image_set}')


def build(image_set: str, args) -> torch.utils.data.Dataset:
    root = Path(args.coco_path)
    assert root.exists(), f'Provided COCO path {root} does not exist'
    
    PATHS = {
        "train": (root / "images/train_mini2017", root / "annotations/instances_minitrain2017.json"),
        "val": (root / "images/val2017/images", root / "annotations/instances_val2017.json"),
        "val": (root / "images/val2017/images", root / "annotations/instances_val2017.json"),
       # "test": (root / "images/test2017/images", root / "annotations/instances_test2017.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(str(img_folder), str(ann_file), transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
