# -*- encoding: utf-8 -*-
"""
@File    :   deformable_detr_head.py
@Time    :   2024/01/17 09:54:09
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from pplsd.apis import manager
from pplsd.layers import build_conv_layer, build_linear_layer, build_norm_layer
from pplsd.layers.param_init import (
    constant_init,
    xavier_uniform_init,
)


__all__ = ["DeformableDETRHead"]


class PredictionHead1D(nn.Layer):
    """
    Prediction Head 1D
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=64,
        num_layers=2,
        norm_cfg=dict(type_name="BatchNorm1D", data_format="NLC"),
        init_bias=0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.norm_cfg = norm_cfg
        self.init_bias = init_bias

        layers = []
        for _ in range(num_layers - 1):
            layers.append(
                nn.Sequential(
                    build_linear_layer(
                        in_channels, hidden_channels, bias=True
                    ),
                    build_norm_layer(self.norm_cfg, hidden_channels),
                    nn.ReLU(),
                )
            )
        layers.append(
            build_linear_layer(hidden_channels, out_channels, bias=True)
        )
        self.head = nn.Sequential(*layers)

        self._reset_parameters()

    def _reset_parameters(self):
        """reset parameters"""
        constant_init(self.head[-1].bias, value=self.init_bias)

    def forward(self, x):
        """Forward function for PredictionHead.

        Args:
            x (Tensor): Input feature map with the shape of [B, L, C_{in}]

        Returns:
            out (Tensor): Prediction head output with the shape of [B, L, C_{out}]

        """
        out = self.head(x)

        return out


class PredictionHead2D(nn.Layer):
    """
    Prediction Head 2D
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=64,
        num_layers=2,
        conv_cfg=None,
        norm_cfg=dict(type_name="BatchNorm2D"),
        init_bias=0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.norm_cfg = norm_cfg
        self.init_bias = init_bias

        layers = []
        for _ in range(num_layers - 1):
            layers.append(
                nn.Sequential(
                    build_conv_layer(
                        conv_cfg,
                        hidden_channels,
                        hidden_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias_attr=True,
                    ),
                    build_norm_layer(self.norm_cfg, hidden_channels),
                    nn.ReLU(),
                )
            )
        layers.append(
            build_conv_layer(
                conv_cfg,
                hidden_channels,
                2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=True,
            )
        )
        self.head = nn.Sequential(*layers)

        self._reset_parameters()

    def _reset_parameters(self):
        """reset parameters"""
        constant_init(self.head[-1].bias, value=self.init_bias)

    def forward(self, x):
        """Forward function for PredictionHead.

        Args:
            x (Tensor): Input feature map with the shape of [B, C_{in}, H, W]

        Returns:
            out (Tensor): Prediction head output with the shape of [B, C_{out}, H, W]

        """
        out = self.head(x)

        return out


@manager.HEADS.add_component
class DeformableDETRHead(nn.Layer):
    """
    Deformable DETR Head
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        downsample,
        num_junc_proposals=300,
        num_line_proposals=300,
        junc_nms_kernel_size=1,
        line_nms_kernel_size=1,
        decoder=None,
        line_coder=None,
        use_auxiliary_loss=True,
        loss_jmap=None,
        loss_jcls=None,
        loss_joff=None,
        loss_jmat=None,
        loss_lmap=None,
        loss_lcls=None,
        loss_loff=None,
        conv_cfg=None,
        test_cfg=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.downsample = downsample
        self.num_junc_proposals = num_junc_proposals
        self.num_line_proposals = num_line_proposals
        self.use_line_proposals = num_line_proposals > 0
        self.junc_nms_kernel_size = junc_nms_kernel_size
        self.line_nms_kernel_size = line_nms_kernel_size
        self.decoder = decoder
        self.line_coder = line_coder
        self.use_auxiliary_loss = use_auxiliary_loss
        self.loss_jmap = loss_jmap
        self.loss_joff = loss_joff
        self.loss_jcls = loss_jcls
        self.loss_jmat = loss_jmat
        self.loss_lmap = loss_lmap
        self.loss_lcls = loss_lcls
        self.loss_loff = loss_loff
        self.test_cfg = test_cfg

        # input project layer
        self.input_proj = build_conv_layer(
            conv_cfg,
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=True,
        )

        # heatmap prediction for junction and line
        self.headmap_head = PredictionHead2D(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=2,
            num_layers=2,
            init_bias=-2.19,
        )

        # position embedding
        self.query_pos_embedding = nn.Linear(
            3, hidden_channels
        )  # (x, y, junc or line)

        # prediction head
        self.junc_head = PredictionHead1D(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=3,
            num_layers=2,
        )
        self.line_head = PredictionHead1D(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=5,
            num_layers=2,
        )

        # mask the lower triangle region
        xs = ys = paddle.arange(self.num_junc_proposals)
        ys, xs = paddle.meshgrid([ys, xs])
        link_mask = xs > ys
        self.register_buffer("link_mask", link_mask)

        self._reset_parameters()

    def _reset_parameters(self):
        """reset parameters"""
        for p in self.decoder.parameters():
            if p.dim() > 1:
                xavier_uniform_init(p)

        xavier_uniform_init(self.input_proj.weight)
        constant_init(self.input_proj.bias)

    def _get_init_query(
        self, heatmap, feat_flatten, num_queries, nms_kernel_size=1
    ):
        """
        get initial query from heatmap

        Args:
            heatmap (Tensor): The heatmap, shape: [B, 1, H, W]
            feat_flatten (Tensor): The feature map, shape: [B, H * W, C]
            num_queries (int): The number of query
            nms_kernel_size (int): The kernel size of non maximum suppression

        Returns:
            query (Tensor): The initial query, shape: [B, num_queries, C]
            query_pos (Tensor): The initial query position, shape: [B, num_queries, 2]
            indices (Tensor): The indices of the initial query, shape: [B, num_queries]
        """
        # non maximum suppression
        if nms_kernel_size > 1:
            local_max = F.max_pool2d(
                heatmap,
                kernel_size=nms_kernel_size,
                padding=nms_kernel_size // 2,
            )
            heatmap = heatmap * (heatmap == local_max).astype(heatmap.dtype)

        # get topk
        B, _, H, W = heatmap.shape
        heatmap = heatmap.reshape([B, -1])
        indices = heatmap.argsort(axis=1, descending=True)[:, :num_queries]
        query = paddle.take_along_axis(
            feat_flatten, indices[..., None], axis=1
        )
        ys, xs = indices // W, indices % W
        query_pos = (
            paddle.stack([xs, ys], axis=-1).astype(feat_flatten.dtype) + 0.5
        )  # [B, num_queries, 2]

        return query, query_pos, indices

    def forward(
        self,
        feats,
        **kwargs,
    ):
        """
        Forward function

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
            # input project
            feat = self.input_proj(feat)
            B, C, H, W = feat.shape

            # multi-scale features
            feats_flatten = feat.flatten(2).transpose([0, 2, 1])
            spatial_shapes = paddle.to_tensor([feat.shape[2:]])

            # headmap head
            heatmap_outout = self.headmap_head(feat)
            heatmap = F.sigmoid(heatmap_outout.detach())

            # get query and position
            query, query_pos, junc_indices = self._get_init_query(
                heatmap[:, 0:1],
                feats_flatten,
                self.num_junc_proposals,
                self.junc_nms_kernel_size,
            )
            query_pos = paddle.concat(
                [query_pos, paddle.full_like(query_pos[..., :1], 0)], axis=-1
            )

            if self.use_line_proposals:
                (
                    line_query,
                    line_query_pos,
                    line_indices,
                ) = self._get_init_query(
                    heatmap[:, 1:2],
                    feats_flatten,
                    self.num_line_proposals,
                    self.line_nms_kernel_size,
                )
                line_query_pos = paddle.concat(
                    [
                        line_query_pos,
                        paddle.full_like(line_query_pos[..., :1], 1),
                    ],
                    axis=-1,
                )
                query = paddle.concat([query, line_query], axis=1)
                query_pos = paddle.concat([query_pos, line_query_pos], axis=1)

            # get pos embedding
            query_pos_embed = self.query_pos_embedding(query_pos)

            # transformer decoder
            reference_points = query_pos[..., :2].unsqueeze(2)
            reference_points[..., 0] /= W
            reference_points[..., 1] /= H
            # intermediates = self.decoder(
            #     query=query,
            #     reference_points=reference_points,
            #     value_flatten=feats_flatten,
            #     spatial_shapes=spatial_shapes,
            #     query_pos_embed=query_pos_embed,
            # )
            intermediates = query[None]

            # predict
            L = len(intermediates)  # [L, B, num_queries, C]
            intermediates = intermediates.reshape([L * B, -1, C])
            junc_queries = intermediates[:, : self.num_junc_proposals]
            junc_outputs = self.junc_head(junc_queries)
            junc_outputs = junc_outputs.reshape(
                [L, B, self.num_junc_proposals, -1]
            )
            if self.use_line_proposals:
                line_queries = intermediates[:, self.num_junc_proposals :]
                line_outputs = self.line_head(line_queries)
                line_outputs = line_outputs.reshape(
                    [L, B, self.num_line_proposals, -1]
                )

            pred_dict = dict(
                heatmap=heatmap_outout,
                junc_indices=junc_indices,
                junc_outputs=junc_outputs,
            )
            if self.use_line_proposals:
                pred_dict.update(
                    dict(
                        line_indices=line_indices,
                        line_outputs=line_outputs,
                    )
                )

            pred_dicts.append(pred_dict)

        return pred_dicts

    def loss(self, pred_dicts, gt_lines, **kwargs) -> dict:
        """
        loss

        Args:
            pred_dicts (list[dict]): dict of predicted outputs
            gt_lines (list[Tensor]): list of ground truth lines, with shape [num_lines, 2, 2]

        Returns:
            loss_dict (dict): dict of loss
        """
        # get target
        target_dict = self.line_coder.encode(gt_lines)
        jmap = target_dict["jmap"]
        joff = target_dict["joff"]
        lmap = target_dict["lmap"]
        loff = target_dict["loff"]
        flatten_jmap = jmap.flatten(2).transpose([0, 2, 1])
        flatten_joff = joff.flatten(2).transpose([0, 2, 1])
        flatten_lmap = lmap.flatten(2).transpose([0, 2, 1])
        flatten_loff = loff.flatten(2).transpose([0, 2, 1])
        gt_juncs = target_dict["gt_juncs"]
        adj_mats = target_dict["adj_mats"]
        B, _, H, W = jmap.shape

        loss_dict = dict()
        for i, pred_dict in enumerate(pred_dicts):
            # heatmap loss
            output = pred_dict["heatmap"]
            pred_jmap = output[:, 0:1]
            pred_lmap = output[:, 1:2]
            loss_jmap = self.loss_jmap(
                pred_jmap, jmap, avg_factor=max(1, jmap.sum())
            )
            loss_lmap = self.loss_lmap(
                pred_lmap, lmap, avg_factor=max(1, lmap.sum())
            )
            loss_dict["loss_jmap_{}".format(i)] = loss_jmap
            loss_dict["loss_lmap_{}".format(i)] = loss_lmap

            # junction loss
            indices = pred_dict["junc_indices"]  # [B, num_junc_proposals]
            outputs = pred_dict[
                "junc_outputs"
            ]  # [L, B, num_junc_proposals, C]
            pred_jcls = outputs[..., 0:1]
            pred_joff = outputs[..., 1:3]
            # pred_jmat = outputs[..., 3:]
            # pred_jmat = paddle.matmul(pred_jmat, pred_jmat.transpose([0, 1, 3, 2]))
            L = len(outputs)
            jcls_target = paddle.take_along_axis(
                flatten_jmap, indices.unsqueeze(-1), axis=1
            )
            joff_target = paddle.take_along_axis(
                flatten_joff, indices.unsqueeze(-1), axis=1
            )

            avg_factor = max(1, jcls_target.sum())
            loss_jcls = self.loss_jcls(
                pred_jcls,
                jcls_target.tile([L, 1, 1, 1]),
                avg_factor=avg_factor,
            )
            loss_joff = self.loss_joff(
                pred_joff,
                joff_target.tile([L, 1, 1, 1]),
                weight=jcls_target,
                avg_factor=avg_factor,
            )
            loss_dict["loss_jcls_{}".format(i)] = loss_jcls
            loss_dict["loss_joff_{}".format(i)] = loss_joff

            # xs, ys = junc_indices % W, junc_indices // W
            # centers = paddle.stack([xs, ys], axis=-1).astype(output.dtype) + 0.5

            # # match junction
            # jmat_target = paddle.zeros_like(pred_jmat[0])
            # for batch_id in range(B):
            #     center = centers[batch_id]
            #     gt_junc = gt_juncs[batch_id]
            #     N = len(gt_junc)
            #     adj_mat = adj_mats[batch_id].astype(pred_jmat.dtype)
            #     if N == 0:
            #         continue
            #     dists = ((center[:, None] - gt_junc) ** 2).sum(axis=-1)
            #     match = paddle.argmin(dists, axis=-1)
            #     dists = paddle.take_along_axis(dists, match.unsqueeze(-1), axis=-1).squeeze(-1)
            #     match[dists > 2.25] = N
            #     idx1, idx2 = paddle.meshgrid(match, match)
            #     idx1, idx2 = idx1.flatten(), idx2.flatten()
            #     jmat_target[batch_id] = adj_mat[idx1, idx2].reshape(pred_jmat.shape[-2:])

            # # get loss
            # loss_jmat = self.loss_jmat(pred_jmat, jmat_target.tile([L, 1, 1, 1]), avg_factor=max(1, jmat_target.sum()))
            # loss_dict["loss_jmat_{}".format(i)] = loss_jmat
            # loss_dict["num_jmat_{}".format(i)] = jmat_target.sum(axis=[-2, -1]).mean()
            # pred_jmat = pred_jmat.sigmoid()
            # pred_jmat = pred_jmat[0, 0].numpy()
            # jmat_target = jmat_target[0].numpy()
            # image = (np.hstack([pred_jmat, jmat_target]) * 255).astype("uint8")
            # import cv2
            # cv2.imwrite("image.png", image)

            if self.use_line_proposals:
                indices = pred_dict["line_indices"]
                outputs = pred_dict["line_outputs"]
                pred_lcls = outputs[..., 0:1]
                pred_loff = outputs[..., 1:5]
                lcls_target = paddle.take_along_axis(
                    flatten_lmap, indices.unsqueeze(-1), axis=1
                )
                loff_target = paddle.take_along_axis(
                    flatten_loff, indices.unsqueeze(-1), axis=1
                )

                avg_factor = max(1, lcls_target.sum())
                loss_lcls = self.loss_lcls(
                    pred_lcls,
                    lcls_target.tile([L, 1, 1, 1]),
                    avg_factor=avg_factor,
                )
                loss_loff = self.loss_loff(
                    pred_loff,
                    loff_target.tile([L, 1, 1, 1]),
                    weight=lcls_target,
                    avg_factor=avg_factor,
                )
                loss_dict["loss_lcls_{}".format(i)] = loss_lcls
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
        result_dict = self.line_coder.decode(pred_dict)
        pred_juncs = result_dict["pred_juncs"]
        pred_junc_scores = result_dict["pred_junc_scores"]
        pred_lines = result_dict["pred_lines"]
        pred_line_scores = result_dict["pred_line_scores"]

        results = []
        B = len(pred_juncs)
        for batch_id in range(B):
            # junctions
            pred_junc = pred_juncs[batch_id] * self.downsample
            pred_junc_score = pred_junc_scores[batch_id]
            indices = paddle.argsort(pred_junc_score, descending=True)
            max_num = self.test_cfg.get("junc_max_num", len(pred_junc_score))
            junc_score_thresh = self.test_cfg.get("junc_score_thresh", 0)
            max_num = min(
                max_num, (pred_junc_score >= junc_score_thresh).sum()
            )
            indices = indices[:max_num]
            pred_junc = pred_junc[indices].reshape([-1, 2])
            pred_junc_score = pred_junc_score[indices]

            # lines
            pred_line = pred_lines[batch_id] * self.downsample
            pred_line_score = pred_line_scores[batch_id]
            indices = paddle.argsort(pred_line_score, descending=True)
            max_num = self.test_cfg.get("line_max_num", len(pred_line_score))
            line_score_thresh = self.test_cfg.get("line_score_thresh", 0)
            max_num = min(
                max_num, (pred_line_score >= line_score_thresh).sum()
            )
            indices = indices[:max_num]
            pred_line = pred_line[indices].reshape([-1, 2, 2])
            pred_line_score = pred_line_score[indices]

            result = dict(
                pred_juncs=pred_junc,
                pred_junc_scores=pred_junc_score,
                pred_lines=pred_line,
                pred_line_scores=pred_line_score,
            )
            results.append(result)

        return results
