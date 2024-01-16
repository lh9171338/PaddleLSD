# -*- encoding: utf-8 -*-
"""
@File    :   attention.py
@Time    :   2024/01/16 15:47:55
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from lh_pplsd.layers.param_init import xavier_uniform_init, constant_init


__all__ = [
    "MultiHeadAttention",
    "MSDeformableAttention",
]


class MultiHeadAttention(nn.Layer):
    """
    Attention maps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.
    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.
    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout (float, optional): The dropout probability used on attention
            weights to drop some attention targets. 0 for no dropout. Default 0
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Default False.
    Examples:
        .. code-block:: python
            import paddle
            # encoder input: [batch_size, sequence_length, d_model]
            query = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, num_heads, query_len, query_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            multi_head_attn = paddle.nn.MultiHeadAttention(128, 2)
            output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        kdim=None,
        vdim=None,
        need_weights=False,
    ):
        super(MultiHeadAttention, self).__init__()
        nn.MultiHeadAttention
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = (
            self.kdim == embed_dim and self.vdim == embed_dim
        )

        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim:
            self.in_proj_weight = self.create_parameter(
                shape=[embed_dim, 3 * embed_dim],
                attr=None,
                dtype=self._dtype,
                is_bias=False,
            )
            self.in_proj_bias = self.create_parameter(
                shape=[3 * embed_dim],
                attr=None,
                dtype=self._dtype,
                is_bias=True,
            )
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(self.kdim, embed_dim)
            self.v_proj = nn.Linear(self.vdim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._type_list = ("q_proj", "k_proj", "v_proj")

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_init(p)
            else:
                constant_init(p)

    def compute_qkv(self, tensor, index):
        """compute qkv"""
        if self._qkv_same_embed_dim:
            tensor = F.linear(
                x=tensor,
                weight=self.in_proj_weight[
                    :, index * self.embed_dim : (index + 1) * self.embed_dim
                ],
                bias=self.in_proj_bias[
                    index * self.embed_dim : (index + 1) * self.embed_dim
                ]
                if self.in_proj_bias is not None
                else None,
            )
        else:
            tensor = getattr(self, self._type_list[index])(tensor)
        tensor = tensor.reshape(
            [0, 0, self.num_heads, self.head_dim]
        ).transpose([0, 2, 1, 3])
        return tensor

    def forward(self, query, key=None, value=None, attn_mask=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.
        Parameters:
            query (Tensor): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Tensor, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`. Default None.
            value (Tensor, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`. Default None.
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.
        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `query`, representing attention output. Or a tuple if \
                `need_weights` is True or `cache` is not None. If `need_weights` \
                is True, except for attention output, the tuple also includes \
                the attention weights tensor shaped `[batch_size, num_heads, query_length, key_length]`. \
                If `cache` is not None, the tuple then includes the new cache \
                having the same type as `cache`, and if it is `StaticCache`, it \
                is same as the input `cache`, if it is `Cache`, the new cache \
                reserves tensors concatanating raw tensors with intermediate \
                results of current query.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        q, k, v = (
            self.compute_qkv(t, i) for i, t in enumerate([query, key, value])
        )

        # scale dot product attention
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        scaling = float(self.head_dim) ** -0.5
        product = product * scaling

        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = nn.layer.transformer._convert_attention_mask(
                attn_mask, product.dtype
            )
            product = product + attn_mask
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train",
            )

        out = paddle.matmul(weights, v)

        # combine heads
        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        return out if len(outs) == 1 else tuple(outs)


def deformable_attention_core_func(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor): [n_levels, 2]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]
    Returns:
        output (Tensor): [bs, query_length, C]
    """
    bs, Len_v, n_head, c = value.shape
    _, Len_q, n_head, n_levels, n_points, _ = sampling_locations.shape

    value_list = value.split(value_spatial_shapes.prod(1).tolist(), axis=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes.tolist()):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose([0, 2, 1])
            .reshape([bs * n_head, c, h, w])
        )
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = (
            sampling_grids[:, :, :, level]
            .transpose([0, 2, 1, 3, 4])
            .flatten(0, 1)
        )
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose([0, 2, 1, 3, 4]).reshape(
        [bs * n_head, 1, Len_q, n_levels * n_points]
    )
    output = (
        (
            paddle.stack(sampling_value_list, axis=-2).flatten(-2)
            * attention_weights
        )
        .sum(-1)
        .reshape([bs, n_head * c, Len_q])
    )

    return output.transpose([0, 2, 1])


class MSDeformableAttention(nn.Layer):
    """
    Multi-Scale Deformable Attention Module
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        lr_mult=0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(
            embed_dim,
            self.total_points * 2,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult),
        )

        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """reset parameters"""
        # sampling_offsets
        constant_init(self.sampling_offsets.weight)
        thetas = paddle.arange(self.num_heads, dtype=paddle.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = paddle.stack([thetas.cos(), thetas.sin()], axis=-1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)
        grid_init = grid_init.reshape([self.num_heads, 1, 1, 2]).tile(
            [1, self.num_levels, self.num_points, 1]
        )
        scaling = paddle.arange(
            1, self.num_points + 1, dtype=paddle.float32
        ).reshape([1, 1, -1, 1])
        grid_init *= scaling
        self.sampling_offsets.bias.set_value(grid_init.flatten())

        # attention_weights
        constant_init(self.attention_weights.weight)
        constant_init(self.attention_weights.bias)

        # proj
        xavier_uniform_init(self.value_proj.weight)
        constant_init(self.value_proj.bias)
        xavier_uniform_init(self.output_proj.weight)
        constant_init(self.output_proj.bias)

    def forward(
        self,
        query,
        reference_points,
        value,
        value_spatial_shapes,
        value_mask=None,
    ):
        """
        Args:
            query (Tensor): [B, query_length, C]
            reference_points (Tensor): [B, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [B, value_length, C]
            value_spatial_shapes (Tensor): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [B, value_length], True for non-padding elements, False for padding elements
        Returns:
            output (Tensor): [B, query_length, C]
        """
        B, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        assert int(value_spatial_shapes.prod(1).sum()) == Len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape([B, Len_v, self.num_heads, self.head_dim])

        sampling_offsets = self.sampling_offsets(query).reshape(
            [B, Len_q, self.num_heads, self.num_levels, self.num_points, 2]
        )
        attention_weights = self.attention_weights(query).reshape(
            [B, Len_q, self.num_heads, self.num_levels * self.num_points]
        )
        attention_weights = F.softmax(attention_weights, -1).reshape(
            [B, Len_q, self.num_heads, self.num_levels, self.num_points]
        )

        offset_normalizer = value_spatial_shapes.flip([1]).reshape(
            [1, 1, 1, self.num_levels, 1, 2]
        )
        sampling_locations = (
            reference_points.reshape([B, Len_q, 1, self.num_levels, 1, 2])
            + sampling_offsets / offset_normalizer
        )

        output = deformable_attention_core_func(
            value, value_spatial_shapes, sampling_locations, attention_weights
        )
        output = self.output_proj(output)

        return output
