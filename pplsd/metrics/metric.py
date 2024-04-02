# -*- encoding: utf-8 -*-
"""
@File    :   metric.py
@Time    :   2023/12/21 22:32:32
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import numpy as np
import paddle
import tqdm
from paddle.metric import Metric
from pplsd.apis import manager
import pplsd.apis.utils as api_utils
import pplsd.metrics.utils as metric_utils


__all__ = [
    "ComposeMetric",
    "StructuralAPMetric",
    "JunctionAPMetric",
    "HeatmapAPMetric",
]


@manager.METRICS.add_component
class ComposeMetric(Metric):
    """
    Compose Metric
    """

    def __init__(
        self,
        metrics,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert len(metrics) > 0, "metric_dict should not be empty."
        self.metrics = metrics

        self.reset()

    def name(self):
        """
        Return name of metric instance.
        """
        return self.__name__

    def reset(self):
        """reset"""
        for metric in self.metrics:
            metric.reset()

    def update(self, results):
        """
        update

        Args:
            result (dict|list[dict]): result dict

        Return:
            None
        """
        for metric in self.metrics:
            metric.update(results)

    def accumulate(self, save_dir):
        """
        accumulate

        Args:
            save_dir (str): save dir for metric curve

        Return:
            metric_dict (dict): ap dict
        """
        metric_dict = dict()
        for metric in self.metrics:
            metric_dict.update(metric.accumulate(save_dir))

        return metric_dict


@manager.METRICS.add_component
class StructuralAPMetric(Metric):
    """
    Structural AP Metric
    """

    def __init__(
        self,
        downsample=4,
        thresh_list=[5, 10, 15],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.downsample = downsample
        self.thresh_list = thresh_list
        self.num_thresh = len(thresh_list)

        self.reset()

    def name(self):
        """
        Return name of metric instance.
        """
        return self.__name__

    def reset(self):
        """reset"""
        self.tpfp_buffer = [[] for _ in range(self.num_thresh)]
        self.pred_score_buffer = []
        self.gt_count_buffer = 0

    def update(self, results):
        """
        update

        Args:
            result (dict|list[dict]): result dict

        Return:
            None
        """
        if not isinstance(results, list):
            results = [results]

        for result in tqdm.tqdm(results):
            pred_lines = result["pred_lines"] / self.downsample
            pred_scores = result["pred_line_scores"]
            gt_lines = (
                result["gt_lines"].astype(pred_lines.dtype) / self.downsample
            )

            # record gt count
            num_gt = len(gt_lines)
            self.gt_count_buffer += num_gt

            # record pred score
            self.pred_score_buffer.append(pred_scores.numpy())
            num_pred = len(pred_scores)
            if num_pred == 0:
                continue

            # calculate distance
            if num_gt:
                dists1 = (
                    ((pred_lines[:, None] - gt_lines) ** 2)
                    .sum(axis=-1)
                    .mean(axis=-1)
                )
                dists2 = (
                    ((pred_lines[:, None] - gt_lines[:, ::-1]) ** 2)
                    .sum(axis=-1)
                    .mean(axis=-1)
                )
                dists = paddle.minimum(dists1, dists2)
                gt_indices = paddle.argmin(dists, axis=-1)

                # for each threshold
                for i, thresh in enumerate(self.thresh_list):
                    mask1 = (
                        paddle.take_along_axis(
                            dists, gt_indices[:, None], axis=1
                        )
                        < thresh
                    )
                    mask = paddle.zeros_like(dists, dtype="int32")
                    mask = paddle.put_along_axis(
                        mask,
                        gt_indices[:, None],
                        mask1.astype("int32"),
                        axis=1,
                    )
                    cumsum = paddle.cumsum(mask, axis=0)
                    mask2 = (
                        paddle.take_along_axis(
                            cumsum, gt_indices[:, None], axis=1
                        )
                        == 1
                    )
                    tpfp = (mask1 & mask2).squeeze(axis=-1)
                    self.tpfp_buffer[i].append(tpfp.numpy())
            else:
                # for each threshold
                for i, thresh in enumerate(self.thresh_list):
                    tpfp = np.zeros((num_pred))
                    self.tpfp_buffer[i].append(tpfp)

    def accumulate(self, save_dir=None) -> dict:
        """
        accumulate

        Args:
            save_dir (str): save dir for metric curve

        Return:
            metric_dict (dict): metric dict
        """
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # collect results from all ranks
        buffer = dict(
            gt_count_buffer=self.gt_count_buffer,
            tpfp_buffer=self.tpfp_buffer,
            pred_score_buffer=self.pred_score_buffer,
        )
        ret_list = api_utils.collect_object(buffer)
        self.reset()
        for ret in ret_list:
            self.gt_count_buffer += ret["gt_count_buffer"]
            self.pred_score_buffer.extend(ret["pred_score_buffer"])
            for i in range(self.num_thresh):
                self.tpfp_buffer[i].extend(ret["tpfp_buffer"][i])

        metric_dict = dict()
        gt_count = self.gt_count_buffer
        pred_score = np.concatenate(self.pred_score_buffer)
        for i, thresh in enumerate(self.thresh_list):
            if len(self.tpfp_buffer[i]) == 0:
                AP = 0
            else:
                tpfp = np.concatenate(self.tpfp_buffer[i])

                # sort
                indices = np.argsort(-pred_score, axis=0)
                tpfp = tpfp[indices]

                tp = np.cumsum(tpfp, axis=0)
                fp = np.cumsum(1 - tpfp, axis=0)
                prs = tp / np.maximum(tp + fp, 1e-9)
                rcs = tp / gt_count
                AP, _, _ = metric_utils.calc_AP(prs, rcs)

                if save_dir is not None:
                    save_file = os.path.join(
                        save_dir, "sAP{}.jpg".format(thresh)
                    )
                    metric_utils.plot_pr_curve(
                        save_file,
                        prs,
                        rcs,
                        label="sAP{}={:.1f}".format(thresh, AP),
                    )

            metric_dict["sAP{}".format(thresh)] = AP

        metric_dict["msAP"] = np.mean(list(metric_dict.values()))
        keys = ["msAP"] + [
            "sAP{}".format(thresh) for thresh in self.thresh_list
        ]
        values = [metric_dict[key] for key in keys]
        values = ["{:.1f}".format(value * 100) for value in values]
        print("| " + " | ".join(keys) + " |")
        print("|" + " :---: |" * len(keys))
        print("| " + " | ".join(values) + " |")

        return metric_dict


@manager.METRICS.add_component
class JunctionAPMetric(Metric):
    """
    Junction AP Metric
    """

    def __init__(
        self,
        downsample=4,
        thresh_list=[0.5, 1.0, 2.0],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.downsample = downsample
        self.thresh_list = thresh_list
        self.num_thresh = len(thresh_list)

        self.reset()

    def name(self):
        """
        Return name of metric instance.
        """
        return self.__name__

    def reset(self):
        """reset"""
        self.tpfp_buffer = [[] for _ in range(self.num_thresh)]
        self.pred_score_buffer = []
        self.gt_count_buffer = 0

    def update(self, results):
        """
        update

        Args:
            result (dict|list[dict]): result dict

        Return:
            None
        """
        if not isinstance(results, list):
            results = [results]

        for result in tqdm.tqdm(results):
            pred_juncs = result["pred_juncs"] / self.downsample
            pred_scores = result["pred_junc_scores"]
            gt_lines = (
                result["gt_lines"].astype(pred_juncs.dtype) / self.downsample
            )
            gt_juncs = gt_lines.reshape([-1, 2])
            gt_juncs = paddle.unique(gt_juncs, axis=0)

            # record gt count
            num_gt = len(gt_juncs)
            self.gt_count_buffer += num_gt

            # record pred score
            self.pred_score_buffer.append(pred_scores.numpy())
            num_pred = len(pred_scores)
            if num_pred == 0:
                continue

            # calculate distance
            if num_gt:
                dists = ((pred_juncs[:, None] - gt_juncs) ** 2).sum(axis=-1)
                gt_indices = paddle.argmin(dists, axis=-1)

                # for each threshold
                for i, thresh in enumerate(self.thresh_list):
                    mask1 = (
                        paddle.take_along_axis(
                            dists, gt_indices[:, None], axis=1
                        )
                        < thresh
                    )
                    mask = paddle.zeros_like(dists, dtype="int32")
                    mask = paddle.put_along_axis(
                        mask,
                        gt_indices[:, None],
                        mask1.astype("int32"),
                        axis=1,
                    )
                    cumsum = paddle.cumsum(mask, axis=0)
                    mask2 = (
                        paddle.take_along_axis(
                            cumsum, gt_indices[:, None], axis=1
                        )
                        == 1
                    )
                    tpfp = (mask1 & mask2).squeeze(axis=-1)
                    self.tpfp_buffer[i].append(tpfp.numpy())
            else:
                # for each threshold
                for i, thresh in enumerate(self.thresh_list):
                    tpfp = np.zeros((num_pred))
                    self.tpfp_buffer[i].append(tpfp)

    def accumulate(self, save_dir=None) -> dict:
        """
        accumulate

        Args:
            save_dir (str): save dir for metric curve

        Return:
            metric_dict (dict): metric dict
        """
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # collect results from all ranks
        buffer = dict(
            gt_count_buffer=self.gt_count_buffer,
            tpfp_buffer=self.tpfp_buffer,
            pred_score_buffer=self.pred_score_buffer,
        )
        ret_list = api_utils.collect_object(buffer)
        self.reset()
        for ret in ret_list:
            self.gt_count_buffer += ret["gt_count_buffer"]
            self.pred_score_buffer.extend(ret["pred_score_buffer"])
            for i in range(self.num_thresh):
                self.tpfp_buffer[i].extend(ret["tpfp_buffer"][i])

        metric_dict = dict()
        gt_count = self.gt_count_buffer
        pred_score = np.concatenate(self.pred_score_buffer)
        for i, thresh in enumerate(self.thresh_list):
            if len(self.tpfp_buffer[i]) == 0:
                AP = 0
            else:
                tpfp = np.concatenate(self.tpfp_buffer[i])

                # sort
                indices = np.argsort(-pred_score, axis=0)
                tpfp = tpfp[indices]

                tp = np.cumsum(tpfp, axis=0)
                fp = np.cumsum(1 - tpfp, axis=0)
                prs = tp / np.maximum(tp + fp, 1e-9)
                rcs = tp / gt_count
                AP, _, _ = metric_utils.calc_AP(prs, rcs)

                if save_dir is not None:
                    save_file = os.path.join(
                        save_dir, "APJ{:.1f}.jpg".format(thresh)
                    )
                    metric_utils.plot_pr_curve(
                        save_file,
                        prs,
                        rcs,
                        label="APJ{:.1f}={:.1f}".format(thresh, AP),
                    )

            metric_dict["APJ{:.1f}".format(thresh)] = AP

        metric_dict["mAPJ"] = np.mean(list(metric_dict.values()))
        keys = ["mAPJ"] + [
            "APJ{:.1f}".format(thresh) for thresh in self.thresh_list
        ]
        values = [metric_dict[key] for key in keys]
        values = ["{:.1f}".format(value * 100) for value in values]
        print("| " + " | ".join(keys) + " |")
        print("|" + " :---: |" * len(keys))
        print("| " + " | ".join(values) + " |")

        return metric_dict


@manager.METRICS.add_component
class HeatmapAPMetric(Metric):
    """
    Heatmap AP Metric
    """

    def __init__(
        self,
        downsample=4,
        thresh_list=[
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.97,
            0.99,
            0.995,
            0.999,
            0.9995,
            0.9999,
        ],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.downsample = downsample
        self.thresh_list = thresh_list
        self.num_thresh = len(thresh_list)

        self.reset()

    def name(self):
        """
        Return name of metric instance.
        """
        return self.__name__

    def reset(self):
        """reset"""
        self.tp_buffer = [0 for _ in range(self.num_thresh)]
        self.fp_buffer = [0 for _ in range(self.num_thresh)]
        self.gt_count_buffer = 0

    def _line2heatmap(self, heatmap_size, lines):
        """
        convert line to heatmap

        Args:
            heatmap_size (list[int]): [W, H]
            lines (Tensor): lines with shape (N, 2, 2)

        Return:
            edgemap (Tensor): edge map with shape (H, W)
        """
        W, H = heatmap_size
        edgemap = paddle.zeros((H, W), dtype="bool")
        if len(lines):
            num_pts = (
                paddle.norm(lines[:, 1] - lines[:, 0], 2, axis=-1)
                .max()
                .astype("int32")
            )
            t = paddle.linspace(0, 1, num_pts)
            lambda_ = paddle.stack([1 - t, t], axis=-1)  # (num_pts, 2)
            pts = paddle.matmul(lambda_[None], lines)  # (N, num_pts, 2)
            pts = pts.reshape([-1, 2])  # (N * num_pts, 2)
            pts = paddle.round(pts).astype("int32")
            pts = paddle.unique(pts, axis=0)
            xs, ys = paddle.unstack(pts, axis=-1)
            xs = xs.clip(0, W - 1)
            ys = ys.clip(0, H - 1)
            edgemap[ys, xs] = True

        return edgemap

    def update(self, results):
        """
        update

        Args:
            result (dict|list[dict]): result dict

        Return:
            None
        """
        if not isinstance(results, list):
            results = [results]

        for result in tqdm.tqdm(results):
            pred_lines = result["pred_lines"] / self.downsample
            pred_scores = result["pred_line_scores"]
            gt_lines = (
                result["gt_lines"].astype(pred_lines.dtype) / self.downsample
            )
            img_size = result["img_size"]
            heatmap_size = [
                img_size[0] // self.downsample,
                img_size[1] // self.downsample,
            ]

            # record gt count
            gt_edgemap = self._line2heatmap(heatmap_size, gt_lines)
            self.gt_count_buffer += gt_edgemap.sum().item()

            for i, thresh in enumerate(self.thresh_list):
                mask = pred_scores > thresh
                cur_pred_lines = pred_lines[mask].reshape([-1, 2, 2])
                pred_edgemap = self._line2heatmap(heatmap_size, cur_pred_lines)

                tp = (gt_edgemap & pred_edgemap).sum().item()
                self.tp_buffer[i] += tp
                self.fp_buffer[i] += pred_edgemap.sum().item() - tp

    def accumulate(self, save_dir=None) -> dict:
        """
        accumulate

        Args:
            save_dir (str): save dir for metric curve

        Return:
            metric_dict (dict): metric dict
        """
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # collect results from all ranks
        buffer = dict(
            gt_count_buffer=self.gt_count_buffer,
            tp_buffer=self.tp_buffer,
            fp_buffer=self.fp_buffer,
        )
        ret_list = api_utils.collect_object(buffer)
        self.reset()
        for ret in ret_list:
            self.gt_count_buffer += ret["gt_count_buffer"]
            for i, _ in enumerate(self.thresh_list):
                self.tp_buffer[i] += ret["tp_buffer"][i]
                self.fp_buffer[i] += ret["fp_buffer"][i]

        metric_dict = dict()
        gt_count = self.gt_count_buffer
        tp = np.array(self.tp_buffer)
        fp = np.array(self.fp_buffer)
        prs = tp / np.maximum(tp + fp, 1e-9)
        rcs = tp / gt_count
        indices = np.argsort(rcs)
        rcs = rcs[indices]
        prs = prs[indices]
        precision = np.concatenate(([0.0], prs))
        recall = np.concatenate(([0.0], rcs))

        i = np.where(recall[1:] != recall[:-1])[0]
        APH = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
        FH = (2 * prs * rcs / np.maximum(prs + rcs, 1e-9)).max()

        if save_dir is not None:
            save_file = os.path.join(save_dir, "APH.jpg")
            metric_utils.plot_pr_curve(
                save_file, prs, rcs, label="APH={:.1f}".format(APH)
            )

        metric_dict["APH"] = APH
        metric_dict["FH"] = FH
        keys = ["APH", "FH"]
        values = [metric_dict[key] for key in keys]
        values = ["{:.1f}".format(value * 100) for value in values]
        print("| " + " | ".join(keys) + " |")
        print("|" + " :---: |" * len(keys))
        print("| " + " | ".join(values) + " |")

        return metric_dict
