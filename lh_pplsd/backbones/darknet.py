# -*- encoding: utf-8 -*-
"""
@File    :   darknet.py
@Time    :   2023/11/26 18:31:04
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn as nn
from lh_pplsd.layers import ConvBNLayer
from lh_pplsd.apis import manager


__all__ = ["DarkNet"]


class DownSample(nn.Layer):
    """
    Down Sample
    """

    def __init__(
        self, 
        ch_in,
        ch_out,
        kernel_size=3, 
        stride=2, 
        padding=1,
        norm_type="bn",
    ):
        super().__init__()

        self.conv_bn_layer = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_type=norm_type,
        )
        self.ch_out = ch_out

    def forward(self, x):
        out = self.conv_bn_layer(x)

        return out


class BasicBlock(nn.Layer):
    """
    BasicBlock layer of DarkNet
    """

    def __init__(
        self,
        ch_in,
        ch_out,
        norm_type='bn',
    ):
        super().__init__()

        assert ch_in == ch_out and (ch_in % 2) == 0, \
            f"ch_in and ch_out should be the same even int, but the input \'ch_in is {ch_in}, \'ch_out is {ch_out}"
        self.conv1 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type,
        )
        self.conv2 = ConvBNLayer(
            ch_in=ch_out // 2,
            ch_out=ch_out,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_type=norm_type,
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x

        return out


class Blocks(nn.Layer):
    """
    Blocks layer, which consist of some BaickBlock layers
    """

    def __init__(
        self,
        ch_in,
        ch_out,
        depth,
        norm_type="bn",
    ):
        super().__init__()

        self.blocks = nn.LayerList()
        for i in range(depth):
            if i == 0:
                block = BasicBlock(ch_in, ch_out)
            else:
                block = BasicBlock(ch_out, ch_out)
            self.blocks.append(block)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)

        return out


@manager.HEADS.add_component
class DarkNet(nn.Layer):
    """
    DarkNet, see https://pjreddie.com/darknet/yolo/
    """

    def __init__(
        self,
        ch_in=3,
        ch_out=32,
        depths=[1, 2, 8, 8, 4],
        out_indices=None,
        norm_type="bn",
        pretrained=None,
    ):
        super().__init__()

        self.depths = depths
        if out_indices is None:
            self.out_indices = list(range(len(depths)))
        else:
            self.out_indices = out_indices

        self.conv0 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_type=norm_type,
        )

        self.stages = nn.LayerList()
        for i, depth in enumerate(self.depths):
            downsample = DownSample(
                ch_in=ch_out * (2 ** i),
                ch_out=ch_out * (2 ** (i + 1)),
            )
            stage = Blocks(
                ch_in=ch_out * (2 ** (i + 1)),
                ch_out=ch_out * (2 ** (i + 1)),
                depth=depth,
            )
            self.stages.append(nn.Sequential(downsample, stage))

        if pretrained is not None:
            self.load_dict(paddle.load(pretrained))

    def forward(self, x):
        out = self.conv0(x)

        outs = []
        for i, stage in enumerate(self.stages):
            out = stage(out)
            if i in self.out_indices:
                outs.append(out)

        return outs


if __name__ == "__main__":
    net = DarkNet(
        ch_in=3,
        ch_out=32,
        depths=[1, 2, 8, 8, 4],
        out_indices=[2, 3, 4],
    )
    x = paddle.randn([4, 3, 640, 640])
    outs = net(x)
    for out in outs:
        print(out.shape)
