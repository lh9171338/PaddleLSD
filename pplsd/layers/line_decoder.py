# -*- encoding: utf-8 -*-
"""
@File    :   line_decoder.py
@Time    :   2024/01/20 17:30:57
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn as nn
from pplsd.apis import manager

from pplsd.layers import FFN, MultiHeadAttention, MSDeformableAttention
from pplsd.layers.utils import _get_clones


__all__ = [
    "LineTransformerDecoderLayer",
    "LineTransformerDecoder",
]


@manager.TRANSFORMER_DECODER_LAYERS.add_component
class LineTransformerDecoderLayer(nn.Layer):
    """
    Line Transformer Decoder Layer
    """

    def __init__(
        self,
        num_junc_query=300,
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

        self.num_junc_query = num_junc_query

        # self attention
        self.junc_self_attn = MultiHeadAttention(
            d_model, n_head, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.line_self_attn = MultiHeadAttention(
            d_model, n_head, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(d_model)

        # cross attention
        self.junc_cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points
        )
        self.norm3 = nn.LayerNorm(d_model)

        self.line_cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points
        )
        self.norm4 = nn.LayerNorm(d_model)

        self.junc_to_line_cross_attn = MultiHeadAttention(
            d_model, n_head, dropout=dropout
        )
        self.norm5 = nn.LayerNorm(d_model)

        self.line_to_junc_cross_attn = MultiHeadAttention(
            d_model, n_head, dropout=dropout
        )
        self.norm6 = nn.LayerNorm(d_model)

        # ffn
        self.junc_ffn = FFN(
            d_model,
            dim_feedforward,
            dropout=dropout,
            activation=activation,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
        )
        self.norm7 = nn.LayerNorm(d_model)

        self.line_ffn = FFN(
            d_model,
            dim_feedforward,
            dropout=dropout,
            activation=activation,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
        )
        self.norm8 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

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
            line_query (Tensor): [B, query_length, C]

        Returns:
            query (Tensor): [B, query_length, C]
        """
        junc_query = query[:, : self.num_junc_query]
        line_query = query[:, self.num_junc_query :]
        junc_query_pos_embed = query_pos_embed[:, : self.num_junc_query]
        line_query_pos_embed = query_pos_embed[:, self.num_junc_query :]
        junc_reference_points = reference_points[:, : self.num_junc_query]
        line_reference_points = reference_points[:, self.num_junc_query :]

        # self attention
        q = k = self.with_pos_embed(junc_query, junc_query_pos_embed)
        junc_query2 = self.junc_self_attn(q, k, value=junc_query)
        junc_query = self.norm1(junc_query + self.dropout(junc_query2))

        q = k = self.with_pos_embed(line_query, line_query_pos_embed)
        line_query2 = self.line_self_attn(q, k, value=line_query)
        line_query = self.norm2(line_query + self.dropout(line_query2))

        # cross attention
        q = self.with_pos_embed(junc_query, junc_query_pos_embed)
        junc_query2 = self.junc_cross_attn(
            q, junc_reference_points, value_flatten, spatial_shapes, value_mask
        )
        junc_query = self.norm3(junc_query + self.dropout(junc_query2))

        q = self.with_pos_embed(line_query, line_query_pos_embed)
        line_query2 = self.line_cross_attn(
            q, line_reference_points, value_flatten, spatial_shapes, value_mask
        )
        line_query = self.norm4(line_query + self.dropout(line_query2))

        q = self.with_pos_embed(junc_query, junc_query_pos_embed)
        k = self.with_pos_embed(line_query, line_query_pos_embed)
        v = junc_query
        junc_query2 = self.junc_to_line_cross_attn(q, k, value=line_query)
        junc_query = self.norm5(junc_query + self.dropout(junc_query2))

        line_query2 = self.line_to_junc_cross_attn(k, q, value=v)
        line_query = self.norm6(line_query + self.dropout(line_query2))

        # ffn
        junc_query = self.norm7(
            junc_query + self.dropout(self.junc_ffn(junc_query))
        )
        line_query = self.norm8(
            line_query + self.dropout(self.line_ffn(line_query))
        )

        query = paddle.concat([junc_query, line_query], axis=1)

        return query


@manager.TRANSFORMER_DECODERS.add_component
class LineTransformerDecoder(nn.Layer):
    """
    Line Transformer Decoder
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
        query_pos_embed=None,
    ):
        """
        Foward function

        Args:
            query (Tensor): [B, query_length, C]
            query_pos_embed (Tensor): [B, query_length, C]

        Returns:
            query (Tensor): [L, B, query_length, C]
        """
        intermediate = []
        for layer in self.layers:
            query = layer(
                query=query,
                query_pos_embed=query_pos_embed,
            )

            if self.return_intermediate:
                intermediate.append(query)

        if self.return_intermediate:
            return paddle.stack(intermediate)

        return query.unsqueeze(0)
