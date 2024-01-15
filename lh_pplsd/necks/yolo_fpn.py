# -*- encoding: utf-8 -*-
"""
@File    :   yolo_fpn.py
@Time    :   2023/12/21 19:15:13
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from lh_pplsd.layers import ConvBNLayer
from lh_pplsd.apis import manager


__all__ = ["YOLOv3FPN"]


class YOLOv3DetBlock(nn.Layer):
    """
    YOLODetBlock layer for yolov3, see https://arxiv.org/abs/1804.02767
    """

    def __init__(
        self,
        ch_in,
        channel,
        norm_type="bn",
    ):
        super().__init__()

        conv_def = [
            ("conv0", ch_in, channel, 1),
            ("conv1", channel, channel * 2, 3),
            ("conv2", channel * 2, channel, 1),
            ("conv3", channel, channel * 2, 3),
            ("route", channel * 2, channel, 1),
        ]

        self.conv_module = nn.Sequential()
        for conv_name, ch_in, ch_out, kernel_size in conv_def:
            self.conv_module.add_sublayer(
                conv_name,
                ConvBNLayer(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                    norm_type=norm_type,
                ),
            )

        self.tip = ConvBNLayer(
            ch_in=channel,
            ch_out=channel * 2,
            kernel_size=3,
            padding=1,
            norm_type=norm_type,
        )

    def forward(self, x):
        route = self.conv_module(x)
        tip = self.tip(route)

        return route, tip


@manager.NECKS.add_component
class YOLOv3FPN(nn.Layer):
    """
    YOLOv3FPN layer
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        norm_type="bn",
        out_indices=None,
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        assert len(in_channels) == len(out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = len(in_channels)
        self.out_indices = (
            list(range(len(out_channels)))
            if out_indices is None
            else out_indices
        )

        self.blocks = nn.LayerList()
        self.routes = nn.LayerList()
        for i in range(self.num_blocks):
            ch_in = in_channels[-1 - i]
            if i > 0:
                ch_in += out_channels[0 - i] // 4

            ch_out = out_channels[-1 - i]
            assert ch_out % 4 == 0, "ch_out must be divisible by 4"
            block = YOLOv3DetBlock(
                ch_in,
                channel=ch_out // 2,
                norm_type=norm_type,
            )
            self.blocks.append(block)

            if i < self.num_blocks - 1:
                route = ConvBNLayer(
                    ch_in=ch_out // 2,
                    ch_out=ch_out // 4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_type=norm_type,
                )
                self.routes.append(route)

    def forward(self, feats):
        assert len(feats) == self.num_blocks

        outs = []
        for i, x in enumerate(feats[::-1]):
            if i > 0:
                x = paddle.concat([route, x], axis=1)

            route, tip = self.blocks[i](x)
            outs.append(tip)

            if i < self.num_blocks - 1:
                route = self.routes[i](route)
                route = F.interpolate(
                    route, scale_factor=2, mode="bilinear", align_corners=True
                )

        outs = [
            out for i, out in enumerate(outs[::-1]) if i in self.out_indices
        ]

        return outs


if __name__ == "__main__":
    neck = YOLOv3FPN(
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
    )
    feats = [
        paddle.randn([4, 256, 640, 640]),
        paddle.randn([4, 512, 320, 320]),
        paddle.randn([4, 1024, 160, 160]),
    ]
    outs = neck(feats)
    for out in outs:
        print(out.shape)
