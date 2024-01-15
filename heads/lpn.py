# -*- encoding: utf-8 -*-
"""
@File    :   lpn.py
@Time    :   2024/01/05 10:22:17
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn as nn
from lh_pplsd.apis import manager


__all__ = ["LPN"]


class NormalHead(nn.Layer):
    """
    Normal Head
    """

    expansion = 4

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        assert (
            in_channels % self.expansion == 0
        ), "in_channels must be divisible by expansion: {}".format(
            self.expansion
        )
        hidden_channels = in_channels // self.expansion

        self.head = nn.Sequential(
            nn.Conv2D(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias_attr=False,
            ),
            nn.BatchNorm2D(hidden_channels),
            nn.ReLU(),
            nn.Conv2D(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        out = self.head(x)
        return out


class MultiTaskHead(nn.Layer):
    """
    Multi Task Head
    """

    def __init__(
        self,
        in_channels,
        out_channels_list,
    ):
        super().__init__()

        self.heads = nn.LayerList()
        for out_channels in out_channels_list:
            self.heads.append(NormalHead(in_channels, out_channels))

    def forward(self, x):
        outs = [head(x) for head in self.heads]
        out = paddle.concat(outs, axis=1)

        return out


class LoI(nn.Layer):
    def __init__(self, num_pts=32):
        super().__init__()

        self.num_pts = num_pts

        t = paddle.linspace(0, 1, num_pts)
        lambda_ = paddle.stack([1 - t, t], axis=-1)
        self.register_buffer("lambda_", lambda_)

    def forward(self, feat, loi):
        """
        Forward function of LoI Head

        Args:
            feat (Tensor): The feature map, shape: [C, H, W]
            loi (Tensor): The predicted LoI, shape: [num_lines, 2, 2]

        Returns:
            loi_feat (Tensor): The LoI feature, shape: [num_lines, C, num_pts]
        """
        C, H, W = feat.shape
        feat = feat.transpose([1, 2, 0])

        pts = (self.lambda_[None, :, :, None] * loi[:, None]).sum(axis=-2)
        # [num_lines, num_pts, 2] -> [num_lines * num_pts, 2]
        pts = pts.reshape([-1, 2])
        x, y = pts.unstack(axis=-1)
        x0 = x.floor().clip(min=0, max=W - 1).astype("int32")
        y0 = y.floor().clip(min=0, max=H - 1).astype("int32")
        x1 = (x0 + 1).clip(min=0, max=W - 1).astype("int32")
        y1 = (y0 + 1).clip(min=0, max=H - 1).astype("int32")
        loi_feat = (
            feat[y0, x0].t() * (y1 - y) * (x1 - x)
            + feat[y1, x0].t() * (y - y0) * (x1 - x)
            + feat[y0, x1].t() * (y1 - y) * (x - x0)
            + feat[y1, x1].t() * (y - y0) * (x - x0)
        )
        loi_feat = loi_feat.reshape([C, -1, self.num_pts]).transpose([1, 0, 2])

        return loi_feat


class LoIHead(nn.Layer):
    """
    LoI Head
    """

    def __init__(
        self,
        num_feats,
        num_pts=32,
        hidden_channels=1024,
    ):
        super().__init__()

        self.num_feats = num_feats
        self.num_pts = num_pts

        self.loi = LoI(num_pts=num_pts)
        self.pooling = nn.MaxPool1D(kernel_size=4, stride=4)
        self.fc1 = nn.Sequential(
            nn.Conv2D(
                self.num_feats,
                self.num_feats // 2,
                kernel_size=3,
                padding=1,
                bias_attr=False,
            ),
            nn.BatchNorm2D(self.num_feats // 2),
            nn.ReLU(),
            nn.Conv2D(
                self.num_feats // 2,
                self.num_feats // 2,
                kernel_size=3,
                padding=1,
            ),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(
                (self.num_feats // 2) * (self.num_pts // 4),
                hidden_channels,
                bias_attr=False,
            ),
            nn.BatchNorm1D(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels, bias_attr=False),
            nn.BatchNorm1D(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, feat, lois):
        """
        Forward function of LoI Head

        Args:
            feat (Tensor): The feature map, shape: [B, C, H, W]
            lois (list[Tensor]): The predicted LoI

        Returns:
            loi_scores (list[Tensor]): The LoI scores
        """
        B = feat.shape[0]
        feat = self.fc1(feat)

        loi_scores = []
        for batch_id in range(B):
            loi_feat = self.loi(feat[batch_id], lois[batch_id])
            loi_feat = self.pooling(loi_feat)
            loi_feat = loi_feat.flatten(start_axis=1)
            loi_score = self.fc2(loi_feat).squeeze(axis=-1)
            loi_scores.append(loi_score)

        return loi_scores


@manager.HEADS.add_component
class LPN(nn.Layer):
    """
    Line Proposal Network
    """

    def __init__(
        self,
        num_feats,
        out_channels_list,
        downsample,
        junc_max_num=300,
        line_max_num=5000,
        junc_score_thresh=0,
        line_score_thresh=0,
        num_pos_proposals=300,
        num_neg_proposals=300,
        two_stage=True,
        use_gt_sample=True,
        nms_size=3,
        num_pts=32,
        match_thresh=1.5,
        line_coder=None,
        use_auxiliary_loss=True,
        loss_jmap=None,
        loss_joff=None,
        loss_lmap=None,
        loss_loff=None,
        loss_loi=None,
        test_cfg=dict(),
    ):
        super().__init__()

        self.downsample = downsample
        self.junc_max_num = junc_max_num
        self.line_max_num = line_max_num
        self.junc_score_thresh = junc_score_thresh
        self.line_score_thresh = line_score_thresh
        self.num_pos_proposals = num_pos_proposals
        self.num_neg_proposals = num_neg_proposals
        self.nms_size = nms_size
        self.two_stage = two_stage
        self.use_gt_sample = use_gt_sample
        self.match_thresh = match_thresh
        self.test_cfg = test_cfg

        self.line_coder = line_coder
        self.use_auxiliary_loss = use_auxiliary_loss
        self.loss_jmap = loss_jmap
        self.loss_joff = loss_joff
        self.loss_lmap = loss_lmap
        self.loss_loff = loss_loff
        self.loss_loi = loss_loi

        self.multi_task_head = MultiTaskHead(
            in_channels=num_feats, out_channels_list=out_channels_list
        )
        if self.two_stage:
            self.loi_head = LoIHead(num_feats=num_feats, num_pts=num_pts)

    def forward(
        self, feats, gt_lines=None, gt_samples=None, gt_labels=None, **kwargs
    ):
        """
        Forward function of Line Proposal Network

        Args:
            feats (List[Tensor]): The feature map, shape: [B, C, H, W]
            gt_lines (Tensor): The ground truth lines, shape: [B, num_lines, 2, 2]
            gt_samples (Tensor): The ground truth samples, shape: [B, N, 2, 2]
            gt_labels (Tensor): The ground truth labels, shape: [B, B]

        Returns:
            pred_dicts (List[Dict]): The predictions
        """
        if not isinstance(feats, (list, tuple)):
            feats = [feats]

        if not self.use_auxiliary_loss:
            feats = feats[-1:]

        pred_dicts = []
        for feat in feats:
            # Multi-task head
            output = self.multi_task_head(feat)

            pred_dict = dict(
                output=output,
            )
            if not self.training:
                result_dict = self.line_coder.decode(
                    output,
                    junc_max_num=self.junc_max_num,
                    line_max_num=self.line_max_num,
                    junc_score_thresh=self.junc_score_thresh,
                    line_score_thresh=self.line_score_thresh,
                )
                pred_dict.update(result_dict)

            if not self.two_stage:
                return pred_dict

            # decode
            if self.training:
                result_dict = self.line_coder.decode(
                    output,
                    junc_max_num=self.junc_max_num,
                    line_max_num=self.line_max_num,
                    junc_score_thresh=0,
                    line_score_thresh=0,
                )

            if self.training:
                # sample lines from GT
                result_dict = self._sample_lines(
                    result_dict, gt_lines, gt_samples, gt_labels
                )
                pred_loi = result_dict["sampled_lines"]
            else:
                pred_loi = result_dict["pred_lines"]

            # LoI head
            loi_scores = self.loi_head(feat, pred_loi)
            result_dict["loi_scores"] = loi_scores

            pred_dict.update(result_dict)
            pred_dicts.append(pred_dict)

        return pred_dicts

    def loss(self, pred_dicts, gt_lines, **kwargs) -> dict:
        """
        loss

        Args:
            pred_dicts (list[dict]): dict of predicted outputs
            gt_lines (Tensor): ground truth lines

        Returns:
            loss_dict (dict): dict of loss
        """
        # get target
        target_dict = self.line_coder.encode(gt_lines)
        jmap = target_dict["jmap"]
        joff = target_dict["joff"]
        lmap = target_dict["lmap"]
        loff = target_dict["loff"]

        loss_dict = dict()
        for i, pred_dict in enumerate(pred_dicts):
            # get pred
            output = pred_dict["output"]
            pred_jmap = output[:, 0:1]
            pred_joff = output[:, 1:3]
            pred_lmap = output[:, 3:4]
            pred_loff = output[:, 4:]

            # get loss
            avg_factor = max(1, jmap.sum())
            loss_jmap = self.loss_jmap(pred_jmap, jmap, avg_factor=avg_factor)
            loss_joff = self.loss_joff(
                pred_joff, joff, weight=jmap, avg_factor=avg_factor
            )

            avg_factor = max(1, lmap.sum())
            loss_lmap = self.loss_lmap(pred_lmap, lmap, avg_factor=avg_factor)
            loss_loff = self.loss_loff(
                pred_loff, loff, weight=lmap, avg_factor=avg_factor
            )

            loss_dict["loss_jmap_{}".format(i)] = loss_jmap
            loss_dict["loss_joff{}".format(i)] = loss_joff
            loss_dict["loss_lmap_{}".format(i)] = loss_lmap
            loss_dict["loss_loff_{}".format(i)] = loss_loff

            if self.two_stage:
                loi_labels = pred_dict["sampled_labels"]
                loi_scores = pred_dict["loi_scores"]

                loss_loi_pos, loss_loi_neg = 0, 0
                batch_size = len(loi_labels)
                for loi_score, loi_label in zip(loi_scores, loi_labels):
                    loss_loi = self.loss_loi(
                        loi_score, loi_label, reduction="none"
                    )
                    loss_loi_pos += (
                        loss_loi[loi_label == 1].mean() / batch_size
                    )
                    loss_loi_neg += (
                        loss_loi[loi_label == 0].mean() / batch_size
                    )

                loss_dict["loss_loi_pos_{}".format(i)] = loss_loi_pos
                loss_dict["loss_loi_neg_{}".format(i)] = loss_loi_neg
                loss_dict["num_sampled_pos_{}".format(i)] = pred_dict[
                    "num_sampled_pos"
                ]
                loss_dict["num_sampled_neg_{}".format(i)] = pred_dict[
                    "num_sampled_neg"
                ]

        return loss_dict

    def predict(self, pred_dicts: dict, **kwargs) -> list:
        """
        predict

        Args:
            pred_dict (list[dict]): dict of predicted outputs

        Returns:
            results (list): list of results
        """
        pred_dict = pred_dicts[-1]
        pred_juncs = pred_dict["pred_juncs"]
        pred_junc_scores = pred_dict["pred_junc_scores"]
        pred_lines = pred_dict["pred_lines"]
        if self.two_stage:
            pred_line_scores = []
            line_score_factor = self.test_cfg.get("line_score_factor", 0.5)
            for loi_score, pred_line_score in zip(
                pred_dict["loi_scores"], pred_dict["pred_line_scores"]
            ):
                pred_line_score = (
                    loi_score.sigmoid() ** (line_score_factor)
                ) * (pred_line_score ** (1.0 - line_score_factor))
                pred_line_scores.append(pred_line_score)
        else:
            pred_line_scores = pred_dict["pred_line_scores"]

        results = []
        B = len(pred_juncs)
        for batch_id in range(B):
            pred_junc = pred_juncs[batch_id] * self.downsample
            pred_junc_score = pred_junc_scores[batch_id]
            pred_line = pred_lines[batch_id] * self.downsample
            pred_line_score = pred_line_scores[batch_id]

            indices = paddle.argsort(pred_line_score, descending=True, axis=-1)
            pred_line = pred_line[indices].reshape([-1, 2, 2])
            pred_line_score = pred_line_score[indices]
            max_num = len(pred_line)
            line_score_thresh = self.test_cfg.get("line_score_thresh", 0)
            max_num = min(
                max_num, (pred_line_score >= line_score_thresh).sum()
            )
            pred_line = pred_line[:max_num]
            pred_line_score = pred_line_score[:max_num]

            result = dict(
                pred_juncs=pred_junc,
                pred_junc_scores=pred_junc_score,
                pred_lines=pred_line,
                pred_line_scores=pred_line_score,
            )
            results.append(result)

        return results

    def _sample_lines(
        self, result_dict, gt_lines, gt_samples=None, gt_labels=None
    ):
        """
        Sample proposed lines

        Args:
            result_dict (Dict): The result dict of line proposal network
            gt_lines (Tensor): The ground truth lines, shape: [B, num_lines, 2, 2]
            gt_samples (Tensor): The ground truth samples, shape: [B, N, 2, 2]
            gt_labels (Tensor): The ground truth labels, shape: [B, N]

        Returns:
            result_dict (Dict): The result dict after sampling
        """
        pred_lines = result_dict["pred_lines"]
        B = len(pred_lines)

        sampled_lines, sampled_labels = [], []
        num_sampled_pos, num_sampled_neg = 0, 0
        for batch_id in range(B):
            pred_line = pred_lines[batch_id]

            gt_line = (
                gt_lines[batch_id].astype(pred_line.dtype) / self.downsample
            )
            valid = ~(gt_line == 0).all(axis=[-2, -1])
            gt_line = gt_line[valid].reshape([-1, 2, 2])

            dists1 = (
                ((pred_line[:, None] - gt_line) ** 2)
                .sum(axis=-1)
                .mean(axis=-1)
            )
            dists2 = (
                ((pred_line[:, None] - gt_line[:, ::-1]) ** 2)
                .sum(axis=-1)
                .mean(axis=-1)
            )
            dists = paddle.minimum(dists1, dists2).min(axis=-1)
            label = (dists < self.match_thresh).astype(pred_line.dtype)
            pos_id = (label == 1).nonzero(as_tuple=False).flatten()
            neg_id = (label == 0).nonzero(as_tuple=False).flatten()

            if len(pos_id) > self.num_pos_proposals:
                idx = paddle.randperm(len(pos_id))[: self.num_pos_proposals]
                pos_id = pos_id[idx]

            if len(neg_id) > self.num_neg_proposals:
                idx = paddle.randperm(len(neg_id))[: self.num_neg_proposals]
                neg_id = neg_id[idx]

            if len(pos_id):
                keep_id = paddle.concat([pos_id, neg_id])
            else:
                keep_id = neg_id
            sampled_line = pred_line[keep_id].reshape([-1, 2, 2])
            sampled_label = label[keep_id]
            num_sampled_pos += len(pos_id) / B
            num_sampled_neg += len(neg_id) / B

            # add GT samples
            if self.use_gt_sample:
                gt_sample = gt_samples[batch_id].astype(sampled_line.dtype)
                gt_label = gt_labels[batch_id].astype(sampled_label.dtype)
                valid = gt_label >= 0
                gt_sample = gt_sample[valid].reshape([-1, 2, 2])
                gt_label = gt_label[valid]

                sampled_line = paddle.concat([sampled_line, gt_sample])
                sampled_label = paddle.concat([sampled_label, gt_label])

            # shuflle
            idx = paddle.randperm(len(sampled_line))
            sampled_line = sampled_line[idx].reshape([-1, 2, 2])
            sampled_label = sampled_label[idx]

            mask = paddle.randint_like(sampled_label, 2) < 0.5
            sampled_line[mask] = sampled_line[mask][:, ::-1]

            sampled_lines.append(sampled_line)
            sampled_labels.append(sampled_label)

        result_dict["sampled_lines"] = sampled_lines
        result_dict["sampled_labels"] = sampled_labels
        result_dict["num_sampled_pos"] = num_sampled_pos
        result_dict["num_sampled_neg"] = num_sampled_neg

        return result_dict
