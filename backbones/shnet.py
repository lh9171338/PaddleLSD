# -*- encoding: utf-8 -*-
"""
@File    :   shnet.py
@Time    :   2024/01/04 15:27:58
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn as nn
from lh_pplsd.apis import manager


__all__ = ["SHNet"]


class Residual(nn.Layer):
    """
    Residual block
    """

    expansion = 2

    def __init__(
        self,
        inplanes,
        planes=None,
        stride=1,
    ):
        super().__init__()
        planes = planes or inplanes // self.expansion

        self.bn1 = nn.BatchNorm2D(inplanes)
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(
            planes, planes, kernel_size=3, stride=stride, padding=1
        )
        self.bn3 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * self.expansion, kernel_size=1)
        self.relu = nn.ReLU()

        if inplanes != planes * self.expansion:
            self.downsample = nn.Conv2D(
                inplanes, planes * self.expansion, kernel_size=1
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity

        return out


class Hourglass(nn.Layer):
    """
    Hourglass block
    """

    def __init__(self, depth, num_blocks, num_feats):
        super().__init__()

        self.depth = depth
        self.num_blocks = num_blocks
        self.num_feats = num_feats

        self.downsample = nn.MaxPool2D(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.up1 = nn.Sequential(
            *[Residual(self.num_feats) for _ in range(self.num_blocks)]
        )
        self.low1 = nn.Sequential(
            *[Residual(self.num_feats) for _ in range(self.num_blocks)]
        )
        if self.depth > 1:
            self.low2 = Hourglass(
                self.depth - 1, self.num_blocks, self.num_feats
            )
        else:
            self.low2 = nn.Sequential(
                *[Residual(self.num_feats) for _ in range(self.num_blocks)]
            )
        self.low3 = nn.Sequential(
            *[Residual(self.num_feats) for _ in range(self.num_blocks)]
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
class SHNet(nn.Layer):
    """
    Stacked Hourglass Network
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

        expansion = Residual.expansion
        self.stem = nn.Sequential(
            nn.Conv2D(
                in_channels, stem_channels, kernel_size=7, stride=2, padding=3
            ),
            nn.BatchNorm2D(stem_channels),
            nn.ReLU(),
            Residual(stem_channels, stem_channels),
            nn.MaxPool2D(kernel_size=2, stride=2),
            Residual(stem_channels * expansion),
            Residual(stem_channels * expansion, self.num_feats // expansion),
        )

        hourglasses, residuals, fcs, fcs_ = [], [], [], []
        for i in range(self.num_stacks):
            hourglasses.append(
                Hourglass(self.depth, self.num_blocks, self.num_feats)
            )
            residual = nn.Sequential(
                *[Residual(self.num_feats) for _ in range(self.num_blocks)]
            )
            residuals.append(residual)
            fcs.append(
                nn.Sequential(
                    nn.Conv2D(
                        self.num_feats, self.num_feats, kernel_size=1, stride=1
                    ),
                    nn.BatchNorm2D(self.num_feats),
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
    net = SHNet(
        in_channels=3,
        stem_channels=64,
        out_channels=256,
        depth=4,
        num_stacks=2,
        num_blocks=2,
    )
    x = paddle.randn([4, 3, 512, 512])
    outs = net(x)
    for out in outs:
        print(out.shape)
