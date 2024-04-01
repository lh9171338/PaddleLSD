# -*- encoding: utf-8 -*-
"""
@File    :   wireframe_dataset.py
@Time    :   2024/01/04 19:38:09
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import pickle
import logging
import numpy as np
import paddle
from pplsd.datasets.transforms import Compose
from pplsd.apis import manager


@manager.DATASETS.add_component
class WireframeDataset(paddle.io.Dataset):
    """
    WireframeDataset
    """

    def __init__(
        self,
        data_root,
        ann_file,
        mode,
        pipeline=None,
        **kwargs,
    ):
        super().__init__()

        self.data_root = data_root
        self.ann_file = ann_file
        self.mode = mode
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None

        self.load_annotations()

    @property
    def is_train_mode(self):
        """is train mode"""
        return self.mode == "train"

    @property
    def is_test_mode(self):
        """is test mode"""
        return self.mode == "test"

    def load_annotations(self):
        """load annotations"""
        with open(self.ann_file, "rb") as f:
            self.data_infos = pickle.load(f)

    def __getitem__(self, index):
        info = self.data_infos[index]
        image_file = os.path.join(
            self.data_root, info["img_meta"]["image_file"]
        )

        sample = {
            "mode": self.mode,
            "img_meta": {
                "sample_idx": index,
                "image_file": image_file,
            },
        }
        if not self.is_test_mode:
            sample["gt_lines"] = info["lines"]

        if self.pipeline:
            sample = self.pipeline(sample)

        return sample

    def __len__(self):
        return len(self.data_infos)

    def collate_fn(self, batch):
        """collate_fn"""
        sample = batch[0]
        collated_batch = {}
        for key in sample:
            if key in ["image"]:
                collated_batch[key] = np.stack(
                    [elem[key] for elem in batch], axis=0
                )
            elif key in ["gt_lines", "gt_samples", "gt_labels", "img_meta"]:
                collated_batch[key] = [elem[key] for elem in batch]

        return collated_batch


if __name__ == "__main__":
    # set base logging config
    fmt = "[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

    from pplsd.datasets.transforms import (
        LoadImageFromFile,
        ColorJitter,
        RandomHorizontalFlip,
        RandomVerticalFlip,
        ResizeImage,
        RandomErasingImage,
        RandomScaleImage,
        Visualize,
        NormalizeImage,
    )

    dataset = {
        "type": "WireframeDataset",
        "data_root": "data/wireframe",
        "ann_file": "data/wireframe/train.pkl",
        "mode": "train",
        "pipeline": [
            LoadImageFromFile(),
            ColorJitter(0.4, 0.4, 0.4, 0.4),
            RandomHorizontalFlip(1),
            RandomVerticalFlip(1),
            ResizeImage(size=(512, 512)),
            # RandomScaleImage([0.5, 2]),
            Visualize(save_path="visualize", with_label=True),
            NormalizeImage(
                mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
            ),
        ],
    }
    dataset = WireframeDataset(**dataset)
    dataloader = paddle.io.DataLoader(
        dataset,
        batch_size=4,
        num_workers=16,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    for sample in dataloader:
        print(sample["img_meta"]["sample_idx"])
