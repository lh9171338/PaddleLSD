# -*- encoding: utf-8 -*-
"""
@File    :   resnet.py
@Time    :   2023/12/29 13:41:05
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn as nn
from paddle.distributed.fleet.utils import recompute
from lh_pplsd.layers import build_conv_layer, build_norm_layer
from lh_pplsd.apis import manager


__all__ = ["ResNet"]


class BasicBlock(nn.Layer):
    """
    Basic Block for ResNet
    """
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type_name="BatchNorm2D"),
        **kwargs,
    ):
        super().__init__()

        self.norm1 = build_norm_layer(norm_cfg, planes)
        self.norm2 = build_norm_layer(norm_cfg, planes)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias_attr=False,
        )
        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            3,
            padding=1,
            bias_attr=False,
        )
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    def forward(self, x):
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = recompute(_inner_forward, x) if self.with_cp else _inner_forward(x)
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    """
    Bottleneck block for ResNet
    """
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        style="pytorch",
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type_name="BatchNorm2D"),
    ):
        super().__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.style = style
        if self.style == "pytorch":
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1 = build_norm_layer(norm_cfg, planes)
        self.norm2 = build_norm_layer(norm_cfg, planes)
        self.norm3 = build_norm_layer(norm_cfg, planes * self.expansion)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias_attr=False,
        )
        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias_attr=False,
        )
        self.conv3 = build_conv_layer(
            conv_cfg, 
            planes, 
            planes * self.expansion, 
            kernel_size=1, 
            bias_attr=False,
        )

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):

        def _inner_forward(x):
            identity = x
            out = self.relu(self.norm1(self.conv1(x)))
            out = self.relu(self.norm2(self.conv2(out)))
            out = self.norm3(self.conv3(out))

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = recompute(_inner_forward, x) if self.with_cp else _inner_forward(x)
        out = self.relu(out)

        return out


class ResLayer(nn.Sequential):
    """
    ResLayer to build ResNet style backbone
    """

    def __init__(
        self,
        block,
        inplanes,
        planes,
        num_blocks,
        stride=1,
        avg_down=False,
        conv_cfg=None,
        norm_cfg=dict(type_name="BatchNorm2D"),
        downsample_first=True,
        **kwargs,
    ):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2D(kernel_size=stride, stride=stride, ceil_mode=True)
                )
            downsample.extend(
                [
                    build_conv_layer(
                        conv_cfg,
                        inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=conv_stride,
                        bias_attr=False,
                    ),
                    build_norm_layer(norm_cfg, planes * block.expansion),
                ]
            )
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs,
                )
            )
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs,
                    )
                )

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs,
                    )
                )
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs,
                )
            )
        super().__init__(*layers)


@manager.BACKBONES.add_component
class ResNet(nn.Layer):
    """
    ResNet backbone
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        style="pytorch",
        deep_stem=False,
        avg_down=False,
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type_name="BatchNorm2D", requires_grad=True),
        norm_eval=True,
        with_cp=False,
        zero_init_residual=True,
        pretrained=None,
    ):
        super().__init__()

        if depth not in self.arch_settings:
            raise KeyError(f"invalid depth {depth} for resnet")
        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f"layer{i + 1}"
            self.add_sublayer(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        if pretrained is not None:
            self.load_dict(paddle.load(pretrained))

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias_attr=False,
                ),
                build_norm_layer(self.norm_cfg, stem_channels // 2),
                nn.ReLU(),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias_attr=False,
                ),
                build_norm_layer(self.norm_cfg, stem_channels // 2),
                nn.ReLU(),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias_attr=False,
                ),
                build_norm_layer(self.norm_cfg, stem_channels),
                nn.ReLU(),
            )
        else:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias_attr=False,
                ),
                build_norm_layer(
                    self.norm_cfg,
                    stem_channels,
                ),
                nn.ReLU(),
            )

        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.trainable = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.trainable = False

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)

        return outs


if __name__ == "__main__":
    net = ResNet(
        depth=50,
        in_channels=3,
        base_channels=32,
        num_stages=4,
        strides=(1, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
    )
    x = paddle.randn([4, 3, 640, 640])
    outs = net(x)
    for out in outs:
        print(out.shape)
