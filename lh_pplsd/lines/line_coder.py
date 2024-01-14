# -*- encoding: utf-8 -*-
"""
@File    :   line_coder.py
@Time    :   2024/01/04 22:47:04
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn.functional as F
from lh_pplsd.apis import manager


@manager.LINE_CODERS.add_component
class CenterLineJuncCoder:
    """
    Center Line and Junction Coder
    """

    def __init__(
        self,
        feat_size,
        downsample,
        nms_size=3,
    ):
        self.feat_size = feat_size
        self.downsample = downsample
        self.nms_size = nms_size

    def encode(self, gt_lines):
        """
        encode

        Args:
            gt_lines (Tensor): ground truth lines, shape [B, N, 2]
        
        Returns:
            target_dict (dict): target dict
        """
        B = len(gt_lines)
        W, H = self.feat_size
        jmap = paddle.zeros((B, H, W, 1), dtype="float32")
        joff = paddle.zeros((B, H, W, 2), dtype="float32")
        lmap = paddle.zeros((B, H, W, 1), dtype="float32")
        loff = paddle.zeros((B, H, W, 4), dtype="float32")

        for batch_id in range(B):
            # remove padding lines
            line = gt_lines[batch_id]
            valid = ~(line == 0).all(axis=[-2, -1])
            line = line[valid]
            if len(line) == 0:
                continue
            
            # downsample
            line[..., 0] = paddle.clip(line[..., 0] / self.downsample, 0, W - 1e-4)
            line[..., 1] = paddle.clip(line[..., 1] / self.downsample, 0, H - 1e-4)
            mask = line[:, 0, 1] > line[:, -1, 1]
            line[mask] = line[mask][:, ::-1]

            # get line target
            center = line.mean(axis=1)
            x, y = paddle.unstack(center, axis=-1)
            x0, y0 = x.astype("int32"), y.astype("int32")
            b = paddle.full_like(x0, batch_id)
            lmap[b, y0, x0] = 1
            loff[b, y0, x0] = (line - center.astype("int32")[:, None]).reshape([-1, 4])

            # get junctions
            junc = line.reshape([-1, 2])
            junc = paddle.unique(junc, axis=0)
            
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
        )

        return target_dict

    def decode(
        self, 
        output,
        junc_max_num,
        line_max_num,
        junc_score_thresh=0,
        line_score_thresh=0,        
    ) -> list:
        """
        decode

        Args:
            output (Tensor): output of network, shape: [B, C, H, W]
            junc_max_num (int): max number of junctions
            line_max_num (int): max number of lines
            junc_score_thresh (float): threshold for junctions
            line_score_thresh (float): threshold for lines
        
        Returns:
            result_dict (dict): dict of results
        """
        B = output.shape[0]
        jmap = output[:, 0:1].sigmoid()
        joff = output[:, 1:3]
        lmap = output[:, 3:4].sigmoid()
        loff = output[:, 4:]

        with paddle.no_grad():
            # NMS
            max_heatmap = F.max_pool2d(jmap, kernel_size=self.nms_size, stride=1, padding=self.nms_size // 2)
            jmap = (jmap == max_heatmap).astype(jmap.dtype) * jmap

            pred_juncs, pred_junc_scores = [], []
            pred_lines, pred_line_scores = [], []
            for batch_id in range(B):
                # Generate junctions and lines
                pred_junc, pred_junc_score = self._calc_junction(jmap[batch_id], joff[batch_id], thresh=junc_score_thresh, top_K=junc_max_num)
                pred_line, pred_line_score = self._calc_line(lmap[batch_id], loff[batch_id], thresh=line_score_thresh, top_K=line_max_num)

                # Match junctions and lines
                dist_junc_to_end1 = ((pred_line[:, None, 0] - pred_junc[None]) ** 2).sum(axis=-1)
                dist_junc_to_end2 = ((pred_line[:, None, 1] - pred_junc[None]) ** 2).sum(axis=-1)
                idx_junc_to_end1 = paddle.argmin(dist_junc_to_end1, axis=-1)
                idx_junc_to_end2 = paddle.argmin(dist_junc_to_end2, axis=-1)
                idx_junc_to_end_min = paddle.minimum(idx_junc_to_end1, idx_junc_to_end2)
                idx_junc_to_end_max = paddle.maximum(idx_junc_to_end1, idx_junc_to_end2)

                iskeep = idx_junc_to_end_min != idx_junc_to_end_max
                idx_junc_to_end_min = idx_junc_to_end_min[iskeep]
                idx_junc_to_end_max = idx_junc_to_end_max[iskeep]

                idx_junc = paddle.stack([idx_junc_to_end_min, idx_junc_to_end_max], axis=-1)
                idx_junc, unique_indices = paddle.unique(idx_junc, return_index=True, axis=0)
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

    def _calc_junction(self, jmap, joff, thresh, top_K):
        """
        caclulate junctions

        Args:
            jmap (Tensor): junction map, shape: [1, H, W]
            joff (Tensor): junction offset, shape: [2, H, W]
            thresh (float): threshold
            top_K (int): top K junctions
        
        Returns:
            juncs (Tensor): junctions, shape: [N, 2]
            scores (Tensor): scores, shape: [N]
        """
        H, W = jmap.shape[-2:]
        scores = jmap.flatten()
        joff = joff.reshape([2, -1]).transpose([1, 0])

        num = min(int((scores >= thresh).sum().item()), top_K)
        indices = paddle.argsort(scores, descending=True)[:num]
        scores = scores[indices]
        xs, ys = indices % W, indices // W
        juncs = paddle.stack([xs, ys], axis=-1) + 0.5 + joff[indices].reshape([-1, 2])

        juncs[..., 0] = juncs[..., 0].clip(min=0, max=W - 1e-4)
        juncs[..., 1] = juncs[..., 1].clip(min=0, max=H - 1e-4)

        return juncs, scores


    def _calc_line(self, lmap, loff, thresh, top_K):
        """
        caclulate lines

        Args:
            lmap (Tensor): line map, shape: [1, H, W]
            loff (Tensor): line offset, shape: [4, H, W]
            thresh (float): threshold
            top_K (int): top K lines
        
        Returns:
            lines (Tensor): lines, shape: [N, 2, 2]
            scores (Tensor): scores, shape: [N]
        """
        H, W = lmap.shape[-2:]
        scores = lmap.flatten()
        loff = loff.reshape([2, 2, -1]).transpose([2, 0, 1])

        num = min(int((scores >= thresh).sum().item()), top_K)
        indices = paddle.argsort(scores, descending=True)[:num]
        scores = scores[indices]
        xs, ys = indices % W, indices // W

        center = paddle.stack([xs, ys], axis=-1).astype(loff.dtype)
        lines = center[:, None] + loff[indices].reshape([-1, 2, 2])

        lines[..., 0] = lines[..., 0].clip(min=0, max=W - 1e-4)
        lines[..., 1] = lines[..., 1].clip(min=0, max=H - 1e-4)

        return lines, scores
