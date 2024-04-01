# -*- encoding: utf-8 -*-
"""
@File    :   deformable_transformer.py
@Time    :   2024/01/16 12:42:58
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from pplsd.apis import manager

from pplsd.layers import (
    PositionEmbedding,
    FFN,
    MultiHeadAttention,
    MSDeformableAttention,
    DirectionalMSDeformableAttention,
)
from pplsd.layers.utils import _get_clones
from pplsd.layers.param_init import (
    linear_init,
    constant_init,
    xavier_uniform_init,
    normal_init,
)


__all__ = [
    "DeformableTransformer",
    "DeformableTransformerEncoder",
    "DeformableTransformerDecoder",
    "DeformableTransformerEncoderLayer",
    "DeformableTransformerDecoderLayer",
]


@manager.TRANSFORMER_ENCODER_LAYERS.add_component
class DeformableTransformerEncoderLayer(nn.Layer):
    """
    Deformable Transformer Encoder Layer
    """

    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_points=4,
        weight_attr=None,
        bias_attr=None,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(
            d_model, dim_feedforward, weight_attr, bias_attr
        )
        self.activation = getattr(F, activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(
            dim_feedforward, d_model, weight_attr, bias_attr
        )
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """reset parameters"""
        linear_init(self.linear1)
        linear_init(self.linear2)
        xavier_uniform_init(self.linear1.weight)
        xavier_uniform_init(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        """add pos embed"""
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        reference_points,
        spatial_shapes,
        src_mask=None,
        pos_embed=None,
    ):
        """
        forward function

        Args:
            src (Tensor): [B, query_length, C]
            reference_points (Tensor): [B, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1)
            spatial_shapes (Tensor): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            src_mask (Tensor): [B, query_length], True for non-padding elements, False for padding elements
            pos_embed (Tensor): [B, query_length, C]
        Returns:
            output (Tensor): [B, query_length, C]
        """
        # self attention
        query = self.with_pos_embed(src, pos_embed)
        src2 = self.self_attn(
            query=query,
            reference_points=reference_points,
            value=src,
            value_spatial_shapes=spatial_shapes,
            value_mask=src_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)

        return src


@manager.TRANSFORMER_ENCODERS.add_component
class DeformableTransformerEncoder(nn.Layer):
    """
    Deformable Transformer Encoder
    """

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios):
        """
        get reference points

        Args:
            spatial_shapes (Tensor): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            valid_ratios (Tensor): valid ratios, with shape [B, n_levels, 2]

        Returns:
            reference_points (Tensor): [B, query_length, n_levels, 2], range in [0, 1]
        """
        valid_ratios = valid_ratios.unsqueeze(1)  # [B, 1, n_levels, 2]
        reference_points = []
        for i, (H, W) in enumerate(spatial_shapes.tolist()):
            ref_y, ref_x = paddle.meshgrid(
                paddle.linspace(0.5, H - 0.5, H),
                paddle.linspace(0.5, W - 0.5, W),
            )
            ref_y = ref_y.flatten().unsqueeze(0) / (
                valid_ratios[:, :, i, 1] * H
            )
            ref_x = ref_x.flatten().unsqueeze(0) / (
                valid_ratios[:, :, i, 0] * W
            )
            reference_points.append(paddle.stack([ref_x, ref_y], axis=-1))
        reference_points = paddle.concat(reference_points, 1).unsqueeze(
            2
        )  # [B, query_length, 1, 2]
        reference_points = reference_points * valid_ratios
        return reference_points

    def forward(
        self,
        src,
        spatial_shapes,
        src_mask=None,
        pos_embed=None,
        valid_ratios=None,
    ):
        """
        Foward function

        Args:
            src (Tensor): [B, query_length, C]
            spatial_shapes (Tensor): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            src_mask (Tensor): [B, query_length], True for non-padding elements, False for padding elements
            pos_embed (Tensor): [B, query_length, C]
            valid_ratios (Tensor): valid ratios, with shape [B, n_levels, 2]

        Returns:
            output (Tensor): [B, query_length, C]
        """
        output = src
        if valid_ratios is None:
            valid_ratios = paddle.ones(
                [src.shape[0], spatial_shapes.shape[0], 2]
            )
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios
        )
        for layer in self.layers:
            output = layer(
                output, reference_points, spatial_shapes, src_mask, pos_embed
            )

        return output


@manager.TRANSFORMER_DECODER_LAYERS.add_component
class DeformableTransformerDecoderLayer(nn.Layer):
    """
    Deformable Transformer Decoder Layer
    """

    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_points=4,
        weight_attr=None,
        bias_attr=None,
    ):
        super().__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(
            d_model, dim_feedforward, weight_attr, bias_attr
        )
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(
            dim_feedforward, d_model, weight_attr, bias_attr
        )
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        """reset parameters"""
        linear_init(self.linear1)
        linear_init(self.linear2)
        xavier_uniform_init(self.linear1.weight)
        xavier_uniform_init(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        """add pos embed"""
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        query,
        reference_points,
        value_flatten,
        spatial_shapes,
        value_mask=None,
        query_pos_embed=None,
    ):
        """
        Foward function

        Args:
            query (Tensor): [B, query_length, C]
            reference_points (Tensor): [B, query_length, n_levels, 2]
            value_flatten (Tensor): [B, value_length, C]
            spatial_shapes (Tensor): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [B, value_length], True for non-padding elements, False for padding elements
            query_pos_embed (Tensor): [B, query_length, C]

        Returns:
            query (Tensor): [B, query_length, C]
        """
        # self attention
        q = k = self.with_pos_embed(query, query_pos_embed)
        query2 = self.self_attn(q, k, value=query)
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # cross attention
        q = self.with_pos_embed(query, query_pos_embed)
        query2 = self.cross_attn(
            q, reference_points, value_flatten, spatial_shapes, value_mask
        )
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # ffn
        query2 = self.linear2(
            self.dropout3(self.activation(self.linear1(query)))
        )
        query = query + self.dropout4(query2)
        query = self.norm3(query)

        return query


@manager.TRANSFORMER_DECODERS.add_component
class DeformableTransformerDecoder(nn.Layer):
    """
    Deformable Transformer Decoder
    """

    def __init__(
        self,
        decoder_layer,
        num_layers,
        return_intermediate=False,
    ):
        super().__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(
        self,
        query,
        reference_points,
        value_flatten,
        spatial_shapes,
        value_mask=None,
        query_pos_embed=None,
    ):
        """
        Foward function

        Args:
            query (Tensor): [B, query_length, C]
            reference_points (Tensor): [B, query_length, n_levels, 2]
            value_flatten (Tensor): [B, value_length, C]
            spatial_shapes (Tensor): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [B, value_length], True for non-padding elements, False for padding elements
            query_pos_embed (Tensor): [B, query_length, C]

        Returns:
            query (Tensor): [L, B, query_length, C]
        """
        intermediate = []
        for layer in self.layers:
            query = layer(
                query=query,
                reference_points=reference_points,
                value_flatten=value_flatten,
                spatial_shapes=spatial_shapes,
                value_mask=value_mask,
                query_pos_embed=query_pos_embed,
            )

            if self.return_intermediate:
                intermediate.append(query)

        if self.return_intermediate:
            return paddle.stack(intermediate)

        return query.unsqueeze(0)


@manager.TRANSFORMERS.add_component
class DeformableTransformer(nn.Layer):
    """
    Deformable Transformer
    """

    def __init__(
        self,
        num_queries=300,
        position_embed_type="sine",
        return_intermediate_dec=True,
        backbone_num_channels=[512, 1024, 2048],
        num_feature_levels=4,
        num_encoder_points=4,
        num_decoder_points=4,
        hidden_dim=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        lr_mult=0.1,
        weight_attr=None,
        bias_attr=None,
    ):
        super().__init__()
        assert position_embed_type in [
            "sine",
            "learned",
        ], f"ValueError: position_embed_type not supported {position_embed_type}!"
        assert len(backbone_num_channels) <= num_feature_levels

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels

        encoder_layer = DeformableTransformerEncoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            num_encoder_points,
            weight_attr,
            bias_attr,
        )
        self.encoder = DeformableTransformerEncoder(
            encoder_layer, num_encoder_layers
        )

        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            num_decoder_points,
            weight_attr,
            bias_attr,
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, return_intermediate_dec
        )

        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_embed = nn.Embedding(num_queries, hidden_dim)

        self.reference_points = nn.Linear(
            hidden_dim,
            2,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult),
        )

        self.input_proj = nn.LayerList()
        for in_channels in backbone_num_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels,
                        hidden_dim,
                        kernel_size=1,
                        weight_attr=weight_attr,
                        bias_attr=bias_attr,
                    ),
                    nn.GroupNorm(32, hidden_dim),
                )
            )
        in_channels = backbone_num_channels[-1]
        for _ in range(num_feature_levels - len(backbone_num_channels)):
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels,
                        hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        weight_attr=weight_attr,
                        bias_attr=bias_attr,
                    ),
                    nn.GroupNorm(32, hidden_dim),
                )
            )
            in_channels = hidden_dim

        self.position_embedding = PositionEmbedding(
            hidden_dim // 2,
            normalize=True if position_embed_type == "sine" else False,
            embed_type=position_embed_type,
            offset=-0.5,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """reset parameters"""
        normal_init(self.level_embed.weight)
        normal_init(self.tgt_embed.weight)
        normal_init(self.query_pos_embed.weight)
        xavier_uniform_init(self.reference_points.weight)
        constant_init(self.reference_points.bias)
        for l in self.input_proj:
            xavier_uniform_init(l[0].weight)
            constant_init(l[0].bias)

    def _get_valid_ratio(self, mask):
        """
        get valie ratio

        Args:
            mask (Tensor): [B, H, W]

        Returns:
            valid_ratio (Tensor): [B, 2]
        """
        mask = mask.astype(paddle.float32)
        _, H, W = mask.shape
        valid_ratio_h = paddle.sum(mask[:, :, 0], 1) / H  # [B]
        valid_ratio_w = paddle.sum(mask[:, 0, :], 1) / W  # [B]
        valid_ratio = paddle.stack(
            [valid_ratio_w, valid_ratio_h], -1
        )  # [B, 2]
        return valid_ratio

    def forward(
        self,
        src_feats,
        src_mask=None,
    ):
        """
        Forward function

        Args:
            src_feats (List(Tensor)): Backbone feature maps with shape [[B, C, H, W]]
            src_mask (Tensor, optional): Feature map mask with shape [B, H, W]

        Returns:
            output (Tensor): [L, B, query_length, C]
            memory (Tensor): [B, value_length, C]
            reference_points (Tensor): reference points, with shape [B, query_length, 2]
        """
        srcs = []
        for i in range(len(src_feats)):
            srcs.append(self.input_proj[i](src_feats[i]))
        if self.num_feature_levels > len(srcs):
            len_srcs = len(srcs)
            for i in range(len_srcs, self.num_feature_levels):
                if i == len_srcs:
                    srcs.append(self.input_proj[i](src_feats[-1]))
                else:
                    srcs.append(self.input_proj[i](srcs[-1]))

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = []
        for level, src in enumerate(srcs):
            B, C, H, W = src.shape
            spatial_shapes.append([H, W])
            src = src.flatten(2).transpose([0, 2, 1])  # [B, H * W, C]
            src_flatten.append(src)
            if src_mask is not None:
                mask = F.interpolate(
                    src_mask.unsqueeze(0).astype(src.dtype), size=(H, W)
                )[0].astype("bool")
            else:
                mask = paddle.ones([B, H, W], dtype="bool")
            valid_ratios.append(self._get_valid_ratio(mask))  # [B, 2]
            pos_embed = (
                self.position_embedding(mask).flatten(2).transpose([0, 2, 1])
            )  # [B, H * W, C]
            lvl_pos_embed = pos_embed + self.level_embed.weight[level].reshape(
                [1, 1, -1]
            )  # [B, H * W, C]
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask = mask.astype(src.dtype).flatten(1)  # [B, H * W]
            mask_flatten.append(mask)
        src_flatten = paddle.concat(src_flatten, 1)  # [B, n_levels * H * W, C]
        mask_flatten = paddle.concat(mask_flatten, 1)  # [B, n_levels, H * W]
        lvl_pos_embed_flatten = paddle.concat(
            lvl_pos_embed_flatten, 1
        )  # [B, n_levels, H * W, C]
        spatial_shapes = paddle.to_tensor(
            spatial_shapes, dtype="int64"
        )  # [n_levels, 2]
        valid_ratios = paddle.stack(valid_ratios, 1)  # [B, n_levels, 2]

        # encoder
        memory = self.encoder(
            src=src_flatten,
            spatial_shapes=spatial_shapes,
            src_mask=mask_flatten,
            pos_embed=lvl_pos_embed_flatten,
            valid_ratios=valid_ratios,
        )

        # prepare input for decoder
        query_embed = self.query_pos_embed.weight.unsqueeze(0).tile(
            [B, 1, 1]
        )  # [B, query_length, C]
        tgt = self.tgt_embed.weight.unsqueeze(0).tile(
            [B, 1, 1]
        )  # [B, query_length, C]
        reference_points = F.sigmoid(
            self.reference_points(query_embed)
        )  # [B, query_length, 2]
        reference_points_input = reference_points.unsqueeze(
            2
        ) * valid_ratios.unsqueeze(
            1
        )  # [B, query_length, n_levels, 2]

        # decoder
        hs = self.decoder(
            tgt,
            reference_points_input,
            memory,
            spatial_shapes,
            mask_flatten,
            query_embed,
        )

        return hs, memory, reference_points


if __name__ == "__main__":
    transformer = DeformableTransformer(
        backbone_num_channels=[256, 512, 1024],
    )
    src_feats = [
        paddle.randn([4, 256, 32, 32]),
        paddle.randn([4, 512, 16, 16]),
        paddle.randn([4, 1024, 8, 8]),
    ]
    outs = transformer(src_feats)
    for out in outs:
        print(out.shape)
