# -*- encoding: utf-8 -*-
"""
@File    :   replpn.py
@Time    :   2024/03/04 15:17:45
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn as nn
from pplsd.apis import manager
from pplsd.layers import RepLayer, RepLargeKernelConv, DCN, DDCN


__all__ = ["RepLPN"]


class RepNormalHead(RepLayer):
    """
    Re-parameterization Normal Head
    """

    expansion = 4

    def __init__(
        self,
        in_channels,
        out_channels,
        use_dcn=False,
        use_ddcn=False,
    ):
        super().__init__()
        assert (
            in_channels % self.expansion == 0
        ), "in_channels must be divisible by expansion: {}".format(
            self.expansion
        )
        hidden_channels = in_channels // self.expansion

        if use_dcn:
            self.head = nn.Sequential(
                DCN(
                    in_channels,
                    hidden_channels,
                    kernel_size=3,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(hidden_channels),
                nn.ReLU(),
                DCN(hidden_channels, out_channels, kernel_size=3),
            )
        elif use_ddcn:
            self.head = nn.Sequential(
                DDCN(
                    in_channels,
                    hidden_channels,
                    kernel_size=3,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(hidden_channels),
                nn.ReLU(),
                DDCN(hidden_channels, out_channels, kernel_size=3),
            )
        else:
            self.head = nn.Sequential(
                RepLargeKernelConv(
                    in_channels, hidden_channels, kernel_size=3
                ),
                nn.ReLU(),
                nn.Conv2D(
                    hidden_channels, out_channels, kernel_size=3, padding=1
                ),
            )

    def forward(self, x):
        out = self.head(x)
        return out


class RepMultiTaskHead(RepLayer):
    """
    Re-parameterization Multi Task Head
    """

    def __init__(
        self,
        in_channels,
        out_channels_list,
        use_dcn=False,
        use_ddcn=False,
    ):
        super().__init__()

        self.heads = nn.LayerList()
        for out_channels in out_channels_list:
            self.heads.append(
                RepNormalHead(in_channels, out_channels, use_dcn, use_ddcn)
            )

    def forward(self, x):
        outs = [head(x) for head in self.heads]
        out = paddle.concat(outs, axis=1)

        return out


@manager.HEADS.add_component
class RepLPN(RepLayer):
    """
    Re-parameterization Line Proposal Network
    """

    def __init__(
        self,
        num_feats,
        out_channels_list,
        downsample,
        use_dcn=False,
        use_ddcn=False,
        junc_max_num=300,
        line_max_num=5000,
        junc_score_thresh=0,
        line_score_thresh=0,
        num_pos_proposals=300,
        num_neg_proposals=300,
        nms_size=3,
        match_thresh=1.5,
        line_coder=None,
        use_auxiliary_loss=True,
        loss_jmap=None,
        loss_joff=None,
        loss_lmap=None,
        loss_loff=None,
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
        self.match_thresh = match_thresh
        self.test_cfg = test_cfg

        self.line_coder = line_coder
        self.use_auxiliary_loss = use_auxiliary_loss
        self.loss_jmap = loss_jmap
        self.loss_joff = loss_joff
        self.loss_lmap = loss_lmap
        self.loss_loff = loss_loff

        self.multi_task_head = RepMultiTaskHead(
            in_channels=num_feats,
            out_channels_list=out_channels_list,
            use_dcn=use_dcn,
            use_ddcn=use_ddcn,
        )

    def forward(self, feats, **kwargs):
        """
        Forward function of Line Proposal Network

        Args:
            feats (List[Tensor]): The feature map, shape: [B, C, H, W]

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
        result_dict = self.line_coder.decode(
            pred_dict["output"],
            junc_max_num=self.junc_max_num,
            line_max_num=self.line_max_num,
            junc_score_thresh=self.junc_score_thresh,
            line_score_thresh=self.line_score_thresh,
        )

        pred_juncs = result_dict["pred_juncs"]
        pred_junc_scores = result_dict["pred_junc_scores"]
        pred_lines = result_dict["pred_lines"]
        pred_line_scores = result_dict["pred_line_scores"]

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
