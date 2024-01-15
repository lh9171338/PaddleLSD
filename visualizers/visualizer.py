# -*- encoding: utf-8 -*-
"""
@File    :   visualizer.py
@Time    :   2023/12/21 22:46:13
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lh_pplsd.apis import manager
import lh_pplsd.apis.utils as api_utils
from lh_tool.Iterator import SingleProcess, MultiProcess


__all__ = ["Visualizer", "BoxVisualizer"]


class Visualizer:
    """
    Visualizer
    """

    def __init__(
        self,
        score_thresh=0,
        nprocs=1,
        **kwargs,
    ):
        self.score_thresh = score_thresh
        self.nprocs = nprocs

        self.reset()

    def reset(self):
        """reset"""
        self.result_buffer = []

    def update(self, results):
        """
        update

        Args:
            results (dict|list[dict]): prediction and target
        """
        if not isinstance(results, list):
            results = [results]
        results = api_utils.tensor2numpy(results)
        self.result_buffer.extend(results)

    def visualize(self, save_dir):
        """
        visualize

        Args:
            save_dir (str): save directory

        Returns:
            None
        """
        raise NotImplementedError


@manager.VISUALIZERS.add_component
class LineVisualizer(Visualizer):
    """
    Line Visualizer
    """

    def __init__(
        self,
        fast=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.fast = fast

    def _visualize_single(self, save_dir, result):
        """
        visualize single image

        Args:
            save_dir (str): save directory
            result (dict): prediction and target

        Returns:
            None
        """
        # read image
        img_size = result["img_size"]
        ori_size = result["ori_size"]
        sx = ori_size[0] / img_size[0]
        sy = ori_size[1] / img_size[1]
        image_file = result["image_file"]
        image = cv2.imread(image_file)

        # pred
        pred_lines = result["pred_lines"]
        pred_line_scores = result["pred_line_scores"]
        mask = pred_line_scores >= self.score_thresh
        pred_lines = pred_lines[mask]
        pred_lines[..., 0] *= sx
        pred_lines[..., 1] *= sy

        # plot
        save_file = os.path.join(save_dir, os.path.basename(image_file))
        if self.fast:
            for pts in pred_lines:
                pts = np.round(pts).astype(np.int32)
                cv2.polylines(
                    image,
                    [pts],
                    isClosed=False,
                    color=[0, 255, 255],
                    thickness=2,
                )
                for pt in pts:
                    pt = tuple(pt)
                    cv2.line(image, pt, pt, color=[255, 255, 0], thickness=6)
            cv2.imwrite(save_file, image)
        else:
            fig = plt.figure()
            fig.set_size_inches(ori_size[0] / ori_size[1], 1, forward=False)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.xlim([-0.5, ori_size[0] - 0.5])
            plt.ylim([ori_size[1] - 0.5, -0.5])
            plt.imshow(image[:, :, ::-1])
            for pts in pred_lines:
                pts = pts - 0.5
                plt.plot(pts[:, 0], pts[:, 1], color="orange", linewidth=0.5)
                plt.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    color="#33FFFF",
                    s=1.2,
                    edgecolors="none",
                    zorder=5,
                )

            plt.savefig(save_file, dpi=ori_size[1], bbox_inches=0)
            plt.close()

    def visualize(self, save_dir):
        """
        visualize

        Args:
            save_dir (str): save directory

        Returns:
            None
        """
        os.makedirs(save_dir, exist_ok=True)
        process = MultiProcess if self.nprocs > 1 else SingleProcess
        process(self._visualize_single, nprocs=self.nprocs).run(
            save_dir=save_dir,
            result=self.result_buffer,
        )
