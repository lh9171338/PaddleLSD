# -*- encoding: utf-8 -*-
"""
@File    :   repshnet.py
@Time    :   2024/03/04 10:46:11
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn as nn
from pplsd.apis import manager
from pplsd.layers import RepLayer, RepLargeKernelConv, RepResidualBlock


__all__ = ["RepSHNet"]


class RepHourglass(RepLayer):
    """
    Re-parameterization Hourglass block
    """

    def __init__(self, depth, num_blocks, num_feats):
        super().__init__()

        self.depth = depth
        self.num_blocks = num_blocks
        self.num_feats = num_feats

        self.downsample = nn.MaxPool2D(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.up1 = nn.Sequential(
            *[RepResidualBlock(self.num_feats) for _ in range(self.num_blocks)]
        )
        self.low1 = nn.Sequential(
            *[RepResidualBlock(self.num_feats) for _ in range(self.num_blocks)]
        )
        if self.depth > 1:
            self.low2 = RepHourglass(
                self.depth - 1, self.num_blocks, self.num_feats
            )
        else:
            self.low2 = nn.Sequential(
                *[
                    RepResidualBlock(self.num_feats)
                    for _ in range(self.num_blocks)
                ]
            )
        self.low3 = nn.Sequential(
            *[RepResidualBlock(self.num_feats) for _ in range(self.num_blocks)]
        )

    def forward(self, x):
        up1 = self.up1(x)
        low1 = self.downsample(x)
        low1 = self.low1(low1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.upsample(low3)
        out = up1 + up2

        return out


@manager.BACKBONES.add_component
class RepSHNet(RepLayer):
    """
    Re-parameterization Stacked Hourglass Network
    """

    def __init__(
        self,
        in_channels=3,
        stem_channels=64,
        out_channels=256,
        depth=4,
        num_stacks=2,
        num_blocks=2,
        out_indices=[0, 1],
    ):
        super().__init__()

        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.num_feats = out_channels
        self.depth = depth
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.out_indices = out_indices

        expansion = RepResidualBlock.expansion
        self.stem = nn.Sequential(
            RepLargeKernelConv(
                in_channels, stem_channels, kernel_size=7, stride=2
            ),
            nn.ReLU(),
            RepResidualBlock(stem_channels, stem_channels),
            nn.MaxPool2D(kernel_size=2, stride=2),
            RepResidualBlock(stem_channels * expansion),
            RepResidualBlock(
                stem_channels * expansion, self.num_feats // expansion
            ),
        )

        hourglasses, residuals, fcs, fcs_ = [], [], [], []
        for i in range(self.num_stacks):
            hourglasses.append(
                RepHourglass(self.depth, self.num_blocks, self.num_feats)
            )
            residual = nn.Sequential(
                *[
                    RepResidualBlock(self.num_feats)
                    for _ in range(self.num_blocks)
                ]
            )
            residuals.append(residual)
            fcs.append(
                nn.Sequential(
                    RepLargeKernelConv(
                        self.num_feats, self.num_feats, kernel_size=1
                    ),
                    nn.ReLU(),
                )
            )
            if i < self.num_stacks - 1:
                fcs_.append(
                    nn.Conv2D(self.num_feats, self.num_feats, kernel_size=1)
                )

        self.hourglasses = nn.LayerList(hourglasses)
        self.residuals = nn.LayerList(residuals)
        self.fcs = nn.LayerList(fcs)
        self.fcs_ = nn.LayerList(fcs_)

    def forward(self, x):
        x = self.stem(x)

        outs = []
        for i in range(self.num_stacks):
            y = self.hourglasses[i](x)
            y = self.residuals[i](y)
            y = self.fcs[i](y)
            if i in self.out_indices:
                outs.append(y)
            if i < self.num_stacks - 1:
                y = self.fcs_[i](y)
                x = x + y

        return outs


if __name__ == "__main__":
    import time

    model = RepSHNet(
        in_channels=3,
        stem_channels=64,
        out_channels=256,
        depth=4,
        num_stacks=2,
        num_blocks=2,
    )
    x = paddle.randn([4, 3, 512, 512])
    t1 = time.time()
    outs = model(x)
    t2 = time.time()
    print("time: ", t2 - t1)
    t1 = time.time()
    with paddle.no_grad():
        outs = model(x)
    t2 = time.time()
    print("time: ", t2 - t1)

    model.convert_to_deploy()
    t1 = time.time()
    with paddle.no_grad():
        outs = model(x)
    t2 = time.time()
    print("time: ", t2 - t1)
    for out in outs:
        print(out.shape)
