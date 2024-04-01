# -*- encoding: utf-8 -*-
"""
@File    :   detr_line_coder.py
@Time    :   2024/01/20 12:42:15
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn.functional as F
from pplsd.apis import manager


@manager.LINE_CODERS.add_component
class DETRLineJunctionCoder:
    """
    DETR Line and Junction Coder
    """

    def __init__(
        self,
        feat_size,
        downsample,
    ):
        self.feat_size = feat_size
        self.downsample = downsample

    def encode(self, gt_lines):
        """
        encode

        Args:
            gt_lines (list[Tensor]): list of ground truth lines, shape [N, 2, 2]

        Returns:
            target_dict (dict): target dict
        """
        B = len(gt_lines)
        W, H = self.feat_size
        jmap = paddle.zeros((B, H, W, 1), dtype="float32")
        joff = paddle.zeros((B, H, W, 2), dtype="float32")
        lmap = paddle.zeros((B, H, W, 1), dtype="float32")
        loff = paddle.zeros((B, H, W, 4), dtype="float32")
        gt_juncs, adj_mats = [], []

        for batch_id in range(B):
            # remove padding lines
            line = gt_lines[batch_id]
            if len(line) == 0:
                junc = paddle.zeros([0, 2], dtype="float32")
                adj_mat = paddle.zeros([1, 1], dtype="bool")
                gt_juncs.append(junc)
                adj_mats.append(adj_mat)
                continue

            # downsample
            line[..., 0] = paddle.clip(
                line[..., 0] / self.downsample, 0, W - 1e-4
            )
            line[..., 1] = paddle.clip(
                line[..., 1] / self.downsample, 0, H - 1e-4
            )
            mask = line[:, 0, 1] > line[:, -1, 1]
            line[mask] = line[mask][:, ::-1]

            # get line target
            center = line.mean(axis=1)
            x, y = paddle.unstack(center, axis=-1)
            x0, y0 = x.astype("int32"), y.astype("int32")
            b = paddle.full_like(x0, batch_id)
            lmap[b, y0, x0] = 1
            loff[b, y0, x0] = (
                line - center.astype("int32")[:, None] - 0.5
            ).reshape([-1, 4])

            # get junctions and link tabel
            junc = line.reshape([-1, 2])
            junc, indices = paddle.unique(junc, axis=0, return_inverse=True)
            N = len(junc)
            indices = indices.reshape([-1, 2])
            indices = paddle.concat([indices, indices[:, ::-1]])
            adj_mat = paddle.zeros([N + 1, N + 1], dtype="bool")
            adj_mat[indices[:, 0], indices[:, 1]] = True
            gt_juncs.append(junc)
            adj_mats.append(adj_mat)

            # get junction target
            x, y = paddle.unstack(junc, axis=-1)
            x0, y0 = x.astype("int32"), y.astype("int32")
            b = paddle.full_like(x0, batch_id)
            jmap[b, y0, x0] = 1
            joff[b, y0, x0] = junc - junc.astype("int32") - 0.5

        jmap = jmap.transpose([0, 3, 1, 2])
        joff = joff.transpose([0, 3, 1, 2])
        lmap = lmap.transpose([0, 3, 1, 2])
        loff = loff.transpose([0, 3, 1, 2])

        target_dict = dict(
            jmap=jmap,
            joff=joff,
            lmap=lmap,
            loff=loff,
            gt_juncs=gt_juncs,
            adj_mats=adj_mats,
        )

        return target_dict

    def decode(
        self,
        pred_dict,
    ) -> list:
        """
        decode

        Args:
            pred_dict (dict): dict of predictions

        Returns:
            result_dict (dict): dict of results
        """
        heatmap = pred_dict["heatmap"]
        B = heatmap.shape[0]

        # get junctions
        juncs, junc_scores = self._calc_junction(
            heatmap=heatmap[:, 0],
            indices=pred_dict["junc_indices"],
            outputs=pred_dict["junc_outputs"][-1],
        )

        # get lines
        lines, line_scores = self._calc_line(
            heatmap=heatmap[:, 1],
            indices=pred_dict["line_indices"],
            outputs=pred_dict["line_outputs"][-1],
        )

        pred_juncs, pred_junc_scores = [], []
        pred_lines, pred_line_scores = [], []
        for batch_id in range(B):
            pred_junc = juncs[batch_id]
            pred_junc_score = junc_scores[batch_id]
            pred_line = lines[batch_id]
            pred_line_score = line_scores[batch_id]

            # Match junctions and lines
            dist_junc_to_end1 = (
                (pred_line[:, None, 0] - pred_junc[None]) ** 2
            ).sum(axis=-1)
            dist_junc_to_end2 = (
                (pred_line[:, None, 1] - pred_junc[None]) ** 2
            ).sum(axis=-1)
            idx_junc_to_end1 = paddle.argmin(dist_junc_to_end1, axis=-1)
            idx_junc_to_end2 = paddle.argmin(dist_junc_to_end2, axis=-1)
            idx_junc_to_end_min = paddle.minimum(
                idx_junc_to_end1, idx_junc_to_end2
            )
            idx_junc_to_end_max = paddle.maximum(
                idx_junc_to_end1, idx_junc_to_end2
            )

            iskeep = idx_junc_to_end_min != idx_junc_to_end_max
            idx_junc_to_end_min = idx_junc_to_end_min[iskeep]
            idx_junc_to_end_max = idx_junc_to_end_max[iskeep]

            idx_junc = paddle.stack(
                [idx_junc_to_end_min, idx_junc_to_end_max], axis=-1
            )
            idx_junc, unique_indices = paddle.unique(
                idx_junc, return_index=True, axis=0
            )
            end1 = pred_junc[idx_junc[:, 0]].reshape([-1, 2])
            end2 = pred_junc[idx_junc[:, 1]].reshape([-1, 2])
            pred_line = paddle.stack([end1, end2], axis=1)
            pred_line_score = pred_line_score[unique_indices]

            pred_juncs.append(pred_junc)
            pred_junc_scores.append(pred_junc_score)
            pred_lines.append(pred_line)
            pred_line_scores.append(pred_line_score)

        result_dict = dict(
            pred_juncs=pred_juncs,
            pred_junc_scores=pred_junc_scores,
            pred_lines=pred_lines,
            pred_line_scores=pred_line_scores,
        )

        return result_dict

    def _calc_junction(self, heatmap, indices, outputs):
        """
        caclulate junctions

        Args:
            heatmap (Tensor): heatmap, shape: [B, H, W]
            indices (Tensor): indices, shape: [B, num_queries]
            outputs (Tensor): outputs, shape: [B, num_queries, 3]

        Returns:
            juncs (Tensor): junctions, shape: [num_queries, 2]
            scores (Tensor): scores, shape: [num_queries]
        """
        B, H, W = heatmap.shape
        heatmap = heatmap.sigmoid().flatten(1)
        scores = outputs[..., 0].sigmoid()
        offset = outputs[..., 1:]
        heatmap_scores = paddle.take_along_axis(
            heatmap, indices, axis=-1
        )  # [B, num_queries]
        scores = (scores**0) * (heatmap_scores**1)

        xs, ys = indices % W, indices // W
        centers = paddle.stack([xs, ys], axis=-1).astype(outputs.dtype) + 0.5
        juncs = centers + offset

        juncs[..., 0] = juncs[..., 0].clip(min=0, max=W - 1e-4)
        juncs[..., 1] = juncs[..., 1].clip(min=0, max=H - 1e-4)

        return juncs, scores

    def _calc_line(self, heatmap, indices, outputs):
        """
        caclulate lines

        Args:
            heatmap (Tensor): heatmap, shape: [B, H, W]
            indices (Tensor): indices, shape: [B, num_queries]
            outputs (Tensor): outputs, shape: [B, num_queries, 5]

        Returns:
            lines (Tensor): lines, shape: [num_queries, 2, 2]
            scores (Tensor): scores, shape: [num_queries]
        """
        B, H, W = heatmap.shape
        heatmap = heatmap.sigmoid().flatten(1)
        scores = outputs[..., 0].sigmoid()
        offset = outputs[..., 1:].reshape([B, -1, 2, 2])
        heatmap_scores = paddle.take_along_axis(
            heatmap, indices, axis=-1
        )  # [B, num_queries]
        scores = (scores**0) * (heatmap_scores**1)

        xs, ys = indices % W, indices // W
        centers = paddle.stack([xs, ys], axis=-1).astype(outputs.dtype) + 0.5
        lines = centers.unsqueeze(axis=-2) + offset

        lines[..., 0] = lines[..., 0].clip(min=0, max=W - 1e-4)
        lines[..., 1] = lines[..., 1].clip(min=0, max=H - 1e-4)

        return lines, scores
