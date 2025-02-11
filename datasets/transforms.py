"""
Transforms and data augmentation for both image and bounding boxes.
Adapted from DETR (Facebook) and modified for Conditional DETR.
"""

import random
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate

def crop(image, target, region):
    """Crop the image and target according to the given region."""
    cropped_image = F.crop(image, *region)
    target = target.copy()
    i, j, h, w = region

    # Update target size.
    target["size"] = torch.tensor([h, w])

    # List fields to update.
    fields = []
    if "labels" in target:
        fields.append("labels")
    if "area" in target:
        fields.append("area")
    if "iscrowd" in target:
        fields.append("iscrowd")

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # Remove annotations with zero area.
    if "boxes" in target or "masks" in target:
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)
        for field in fields:
            if field in target:
                target[field] = target[field][keep]

    return cropped_image, target

def hflip(image, target):
    """Horizontally flip the image and target boxes/masks."""
    flipped_image = F.hflip(image)
    w, h = image.size
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        # Flip x-coordinates: swap x_min and x_max and subtract from width.
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes
    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)
    return flipped_image, target

def resize(image, target, size, max_size=None):
    """
    Resize the image (and update target accordingly) to a given size.
    If `size` is an int, the smaller edge is matched to it while keeping the aspect ratio,
    unless that would cause the longer edge to exceed max_size (if provided).
    If `size` is a tuple, it is taken as (width, height).
    """
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min(w, h))
            max_original_size = float(max(w, h))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            # Reverse tuple to (height, width)
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    new_size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, new_size)
    if target is None:
        return rescaled_image, None

    target = target.copy()
    ratio_width = float(rescaled_image.width) / float(image.width)
    ratio_height = float(rescaled_image.height) / float(image.height)
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes
    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area
    h_new, w_new = new_size
    target["size"] = torch.tensor([h_new, w_new])
    if "masks" in target:
        target['masks'] = interpolate(target['masks'][:, None].float(), new_size, mode="nearest")[:, 0] > 0.5
    return rescaled_image, target

def pad(image, target, padding):
    """Pad the bottom/right side of the image."""
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target

class RandomCrop(object):
    """Randomly crop the image."""
    def __init__(self, size):
        self.size = size
    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)

class RandomSizeCrop(object):
    """Crop the image to a random size."""
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        # Determine the maximum allowed width and height based on max_size
        max_w = min(img.width, self.max_size)
        max_h = min(img.height, self.max_size)
        
        # If the image is too small, use the maximum possible size
        if self.min_size > max_w or self.min_size > max_h:
            # Option 1: Use the maximum available size as both w and h
            w = max_w
            h = max_h
            # Optionally, you could also log a warning or handle this case differently.
        else:
            w = random.randint(self.min_size, max_w)
            h = random.randint(self.min_size, max_h)
        
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    """Crop the image at the center."""
    def __init__(self, size):
        self.size = size
    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))

class ColorJitter(object):
    """Randomly adjust brightness, contrast, saturation, and hue."""
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)
    def __call__(self, img, target):
        img = self.color_jitter(img)
        return img, target

class RandomHorizontalFlip(object):
    """Horizontally flip the image with probability p."""
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target

class RandomResize(object):
    """Randomly resize the image to one of the given sizes."""
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size
    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)

class RandomPad(object):
    """Randomly pad the image on right and bottom."""
    def __init__(self, max_pad):
        self.max_pad = max_pad
    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))

class RandomSelect(object):
    """
    Randomly select between two sets of transforms.
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p
    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)

class ToTensor(object):
    """Convert a PIL image to a tensor."""
    def __call__(self, img, target):
        return F.to_tensor(img), target

class RandomErasing(object):
    """Apply Random Erasing to the image."""
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)
    def __call__(self, img, target):
        return self.eraser(img), target

class Normalize(object):
    """Normalize the image (and update bounding boxes accordingly)."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target

class Compose(object):
    """Compose a list of transforms into one."""
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n    {0}".format(t)
        format_string += "\n)"
        return format_string
