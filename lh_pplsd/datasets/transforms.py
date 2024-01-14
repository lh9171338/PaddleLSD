# -*- encoding: utf-8 -*-
"""
@File    :   transforms.py
@Time    :   2023/11/26 15:19:21
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import cv2
import numpy as np
import random
import paddle.vision.transforms as T
from lh_pplsd.apis import manager


__all__ = [
    "Compose",
    "LoadImageFromFile",
    "ColorJitter",
    "RandomVerticalFlip",
    "RandomHorizontalFlip",
    "ResizeImage",
    "RandomScaleImage",
    "RandomErasingImage",
    "RandomSampleLine",
    "NormalizeImage",
    "Visualize",
]


class Compose:
    """
    Compose
    """

    def __init__(self, transforms):
        if not isinstance(transforms, list):
            raise TypeError("The transforms must be a list!")
        self.transforms = transforms

    def __call__(self, sample):
        """ """
        for t in self.transforms:
            sample = t(sample)

        return sample


@manager.TRANSFORMS.add_component
class LoadImageFromFile:
    """
    Load image from file
    """

    def __init__(self, to_float=False):
        self.to_float = to_float

    def __call__(self, sample):
        image_file = sample["img_meta"]["image_file"]
        image = cv2.imread(image_file)
        if self.to_float:
            image = image.astype("float32")

        sample["image"] = image
        sample["img_meta"].update(
            dict(
                ori_size=(image.shape[1], image.shape[0]),
                img_size=(image.shape[1], image.shape[0]),
            )
        )

        return sample


@manager.TRANSFORMS.add_component
class ColorJitter:
    """
    ColorJitter
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.transform = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        sample["image"] = self.transform(sample["image"])

        return sample


@manager.TRANSFORMS.add_component
class RandomVerticalFlip:
    """
    Random vertical flip
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            sample["image"] = sample["image"][::-1]
            height = sample["image"].shape[0]
            gt_lines = sample.get("gt_lines", [])
            if len(gt_lines):
                gt_lines[..., 1] = height - gt_lines[..., 1]
                gt_lines[..., 1] = np.clip(gt_lines[..., 1], 0, height - 1e-4)

        return sample


@manager.TRANSFORMS.add_component
class RandomHorizontalFlip:
    """
    Random horizontal flip
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            sample["image"] = sample["image"][:, ::-1]
            width = sample["image"].shape[1]
            gt_lines = sample.get("gt_lines", [])
            if len(gt_lines):
                gt_lines[..., 0] = width - gt_lines[..., 0]
                gt_lines[..., 0] = np.clip(gt_lines[..., 0], 0, width - 1e-4)

        return sample


@manager.TRANSFORMS.add_component
class ResizeImage:
    """
    Resize image
    """

    def __init__(self, size, interp=cv2.INTER_LINEAR):
        self.size = size
        self.interp = interp

    def __call__(self, sample):
        image = sample["image"]
        ori_size = (image.shape[1], image.shape[0])
        sample["image"] = cv2.resize(
            image, self.size, interpolation=self.interp
        )
        sample["img_meta"]["img_size"] = self.size
        gt_lines = sample.get("gt_lines", [])
        if len(gt_lines):
            sx = self.size[0] / ori_size[0]
            sy = self.size[1] / ori_size[1]
            gt_lines[..., 0] *= sx
            gt_lines[..., 1] *= sy

        return sample


@manager.TRANSFORMS.add_component
class RandomScaleImage:
    """
    Random scale image
    """

    def __init__(self, scales, interp=cv2.INTER_LINEAR):
        assert len(scales) == 2, "len of scales should be 2"
        self.scales = scales
        self.interp = interp

    def __call__(self, sample):
        scale = (
            random.random() * (self.scales[1] - self.scales[0])
            + self.scales[0]
        )
        image = cv2.resize(
            sample["image"],
            (0, 0),
            fx=scale,
            fy=scale,
            interpolation=self.interp,
        )
        sample["image"] = image
        sample["img_meta"]["img_size"] = (image.shape[1], image.shape[0])
        gt_lines = sample.get("gt_lines", [])
        if len(gt_lines):
            gt_lines *= scale

        return sample


@manager.TRANSFORMS.add_component
class RandomErasingImage:
    """
    Random erasing image
    """

    def __init__(self, prob=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.transform = T.RandomErasing(prob, scale, ratio, value)

    def __call__(self, sample):
        sample["image"] = self.transform(sample["image"])

        return sample


@manager.TRANSFORMS.add_component
class RandomSampleLine:
    """
    Random sample line
    """

    def __init__(self, num_stc_pos_proposals, num_stc_neg_proposals=0):
        self.num_stc_pos_proposals = num_stc_pos_proposals
        self.num_stc_neg_proposals = num_stc_neg_proposals
        assert self.num_stc_neg_proposals == 0, "Currently `num_stc_neg_proposals` must be 0"
    
    def __call__(self, sample):
        gt_lines = sample.get("gt_lines")
        gt_pos_lines = np.random.permutation(gt_lines)[:self.num_stc_pos_proposals]
        gt_samples = gt_pos_lines
        gt_labels = np.ones(len(gt_samples), dtype="int32")
        sample["gt_samples"] = gt_samples
        sample["gt_labels"] = gt_labels

        return sample


@manager.TRANSFORMS.add_component
class NormalizeImage:
    """
    Normalize image
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample["image"].astype("float32")
        image = (image - self.mean) / self.std
        sample["image"] = image.transpose((2, 0, 1)).astype("float32")

        return sample


@manager.TRANSFORMS.add_component
class Visualize:
    """
    Visualize
    """

    def __init__(self, save_path, with_label=False):
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.with_label = with_label

    def __call__(self, sample):
        image = sample["image"].copy()
        if self.with_label and "gt_lines" in sample:
            gt_lines = sample["gt_lines"]
            for pts in gt_lines:
                pts = np.round(pts).astype(np.int32)
                cv2.polylines(image, [pts], isClosed=False, color=[0, 255, 255], thickness=2)
                for pt in pts:
                    pt = tuple(pt)
                    cv2.line(image, pt, pt, color=[255, 255, 0], thickness=6)

        filename = os.path.basename(sample["img_meta"]["image_file"])
        save_file = os.path.join(self.save_path, filename)
        cv2.imwrite(save_file, image)

        return sample
