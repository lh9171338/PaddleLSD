# -*- encoding: utf-8 -*-
"""
@File    :   position_encoding.py
@Time    :   2024/01/16 12:44:32
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import math
import paddle
import paddle.nn as nn
from pplsd.apis import manager


__all__ = ["PositionEmbedding"]


@manager.POSITIONAL_ENCODINGS.add_component
class PositionEmbedding(nn.Layer):
    """
    Position Embedding
    """

    def __init__(
        self,
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        scale=None,
        embed_type="sine",
        num_embeddings=50,
        offset=0.0,
    ):
        super().__init__()
        assert embed_type in ["sine", "learned"]

        self.embed_type = embed_type
        self.offset = offset
        self.eps = 1e-6
        if self.embed_type == "sine":
            self.num_pos_feats = num_pos_feats
            self.temperature = temperature
            self.normalize = normalize
            if scale is not None and normalize is False:
                raise ValueError("normalize should be True if scale is passed")
            if scale is None:
                scale = 2 * math.pi
            self.scale = scale
        elif self.embed_type == "learned":
            self.row_embed = nn.Embedding(num_embeddings, num_pos_feats)
            self.col_embed = nn.Embedding(num_embeddings, num_pos_feats)
        else:
            raise ValueError(f"not supported {self.embed_type}")

    def forward(self, mask):
        """
        Args:
            mask (Tensor): [B, H, W]
        Returns:
            pos (Tensor): [B, C, H, W]
        """
        assert mask.dtype == paddle.bool
        if self.embed_type == "sine":
            mask = mask.astype("float32")
            y_embed = mask.cumsum(axis=-2, dtype="float32")  # [B, H, W]
            x_embed = mask.cumsum(axis=-1, dtype="float32")  # [B, H, W]
            if self.normalize:
                y_embed = (
                    (y_embed + self.offset)
                    / (y_embed[:, -1:, :] + self.eps)
                    * self.scale
                )
                x_embed = (
                    (x_embed + self.offset)
                    / (x_embed[:, :, -1:] + self.eps)
                    * self.scale
                )

            dim_t = 2 * (paddle.arange(self.num_pos_feats) // 2).astype(
                "float32"
            )
            dim_t = self.temperature ** (dim_t / self.num_pos_feats)

            pos_x = x_embed.unsqueeze(-1) / dim_t  # [B, H, W, C]
            pos_y = y_embed.unsqueeze(-1) / dim_t  # [B, H, W, C]
            pos_x = paddle.stack(
                [pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()],
                axis=-1,
            ).flatten(
                3
            )  # [B, H, W, C]
            pos_y = paddle.stack(
                [pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()],
                axis=-1,
            ).flatten(
                3
            )  # [B, H, W, C]
            pos = paddle.concat([pos_y, pos_x], axis=-1).transpose(
                [0, 3, 1, 2]
            )  # [B, 2C, H, W]
            return pos
        elif self.embed_type == "learned":
            H, W = mask.shape[-2:]
            x = paddle.arange(W)
            y = paddle.arange(H)
            x_embed = self.col_embed(x)  # [W, C]
            y_embed = self.row_embed(y)  # [H, C]
            pos = paddle.concat(
                [
                    x_embed.unsqueeze(0).tile([H, 1, 1]),
                    y_embed.unsqueeze(1).tile([1, W, 1]),
                ],
                axis=-1,
            )  # [H, W, C]
            pos = (
                pos.transpose([2, 0, 1])
                .unsqueeze(0)
                .tile([mask.shape[0], 1, 1, 1])
            )  # [B, C, H, W]
            return pos
        else:
            raise ValueError(f"not supported {self.embed_type}")
