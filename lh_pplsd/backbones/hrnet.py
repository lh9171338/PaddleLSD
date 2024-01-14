# -*- encoding: utf-8 -*-
"""
@File    :   hrnet.py
@Time    :   2023/12/29 17:07:52
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from lh_pplsd.layers import build_conv_layer, build_norm_layer, kaiming_normal_init, constant_init
from lh_pplsd.apis import manager


__all__ = ["HRNet"]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias_attr=False
    )


class Upsample(nn.Layer):
    """
    Upsample
    """

    def __init__(
        self, 
        scale_factor, 
        mode
    ):
        super().__init__()

        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input_tensor):
        h = int(self.scale_factor * input_tensor.shape[2])
        w = int(self.scale_factor * input_tensor.shape[3])
        input_tensor = F.interpolate(
            input_tensor, size=(h, w), mode=self.mode, align_corners=False
        )
        return input_tensor


class BasicBlock(nn.Layer):
    """
    Basic Block
    """
    expansion = 1

    def __init__(
        self, 
        inplanes, 
        planes, 
        stride=1, 
        downsample=None, 
        norm_cfg=None
    ):
        super().__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        if norm_cfg == None:
            self.bn1 = nn.BatchNorm2D(planes, momentum=0.99)
        else:
            self.bn1 = build_norm_layer(norm_cfg, planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        if norm_cfg == None:
            self.bn2 = nn.BatchNorm2D(planes, momentum=0.99)
        else:
            self.bn2 = build_norm_layer(norm_cfg, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    """
    Bottleneck
    """
    expansion = 2

    def __init__(
        self, 
        inplanes, 
        planes, 
        stride=1, 
        downsample=None, 
        norm_cfg=None
    ):
        super().__init__()

        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        if norm_cfg == None:
            self.bn1 = nn.BatchNorm2D(planes, momentum=0.99)
        else:
            self.bn1 = build_norm_layer(norm_cfg, planes)
        self.conv2 = nn.Conv2D(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False
        )
        if norm_cfg == None:
            self.bn2 = nn.BatchNorm2D(planes, momentum=0.99)
        else:
            self.bn2 = build_norm_layer(norm_cfg, planes)
        self.conv3 = nn.Conv2D(
            planes, planes * self.expansion, kernel_size=1, bias_attr=False
        )
        if norm_cfg == None:
            self.bn3 = nn.BatchNorm2D(planes * self.expansion, momentum=0.99)
        else:
            self.bn3 = build_norm_layer(norm_cfg, planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Layer):
    """
    High Resolution Module
    """

    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        fuse_method,
        multi_scale_output=True,
        norm_cfg=None,
    ):
        super().__init__()

        self.norm_cfg = norm_cfg
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU()

    def _check_branches(
        self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            if self.norm_cfg == None:
                downsample = nn.Sequential(
                    nn.Conv2D(
                        self.num_inchannels[branch_index],
                        num_channels[branch_index] * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias_attr=False,
                    ),
                    nn.BatchNorm2D(
                        num_channels[branch_index] * block.expansion,
                        momentum=0.99,
                    ),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2D(
                        self.num_inchannels[branch_index],
                        num_channels[branch_index] * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias_attr=False,
                    ),
                    build_norm_layer(
                        self.norm_cfg, num_channels[branch_index] * block.expansion
                    ),
                )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
                norm_cfg=self.norm_cfg,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    norm_cfg=self.norm_cfg,
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.LayerList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    if self.norm_cfg == None:
                        fuse_layer.append(
                            nn.Sequential(
                                nn.Conv2D(
                                    num_inchannels[j],
                                    num_inchannels[i],
                                    1,
                                    1,
                                    0,
                                    bias_attr=False,
                                ),
                                nn.BatchNorm2D(num_inchannels[i], momentum=0.99),
                                Upsample(scale_factor=2 ** (j - i), mode="bilinear"),
                            )
                        )
                    else:
                        fuse_layer.append(
                            nn.Sequential(
                                nn.Conv2D(
                                    num_inchannels[j],
                                    num_inchannels[i],
                                    1,
                                    1,
                                    0,
                                    bias_attr=False,
                                ),
                                build_norm_layer(self.norm_cfg, num_inchannels[i]),
                                Upsample(scale_factor=2 ** (j - i), mode="bilinear"),
                            )
                        )

                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            if self.norm_cfg == None:
                                conv3x3s.append(
                                    nn.Sequential(
                                        nn.Conv2D(
                                            num_inchannels[j],
                                            num_outchannels_conv3x3,
                                            3,
                                            2,
                                            1,
                                            bias_attr=False,
                                        ),
                                        nn.BatchNorm2D(
                                            num_outchannels_conv3x3,
                                            momentum=0.99,
                                        ),
                                    )
                                )
                            else:
                                conv3x3s.append(
                                    nn.Sequential(
                                        nn.Conv2D(
                                            num_inchannels[j],
                                            num_outchannels_conv3x3,
                                            3,
                                            2,
                                            1,
                                            bias_attr=False,
                                        ),
                                        build_norm_layer(
                                            self.norm_cfg, num_outchannels_conv3x3
                                        ),
                                    )
                                )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            if self.norm_cfg == None:
                                conv3x3s.append(
                                    nn.Sequential(
                                        nn.Conv2D(
                                            num_inchannels[j],
                                            num_outchannels_conv3x3,
                                            3,
                                            2,
                                            1,
                                            bias_attr=False,
                                        ),
                                        nn.BatchNorm2D(
                                            num_outchannels_conv3x3,
                                            momentum=0.99,
                                        ),
                                        nn.ReLU(),
                                    )
                                )
                            else:
                                conv3x3s.append(
                                    nn.Sequential(
                                        nn.Conv2D(
                                            num_inchannels[j],
                                            num_outchannels_conv3x3,
                                            3,
                                            2,
                                            1,
                                            bias_attr=False,
                                        ),
                                        build_norm_layer(
                                            self.norm_cfg, num_outchannels_conv3x3
                                        ),
                                        nn.ReLU(),
                                    )
                                )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.LayerList(fuse_layer))

        return nn.LayerList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            if i == 0:
                y = x[0]
            else:
                y = self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


@manager.BACKBONES.add_component
class HRNet(nn.Layer):
    """
    High Resolution Net
    """

    def __init__(
        self,
        extra,
        ds_layer_strides=[2, 2, 2, 2],
        us_layer_strides=[1, 2, 4, 8],
        in_channels=3,
        base_channels=32,
        out_channels=256,
        zero_init_residual=False,
        frozen_stages=-1,
        pretrained=None,
        norm_cfg=None,
        **kwargs
    ):
        super().__init__()

        self.norm_cfg = norm_cfg
        self.frozen_stages = frozen_stages
        self.zero_init_residual = zero_init_residual
        # for
        self._in_channels = in_channels
        self.extra = extra
        self._ds_layer_strides = ds_layer_strides
        self._us_layer_strides = us_layer_strides
        # stem network
        # stem net
        self.conv1 = nn.Conv2D(
            self._in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False,
        )
        if norm_cfg == None:
            self.bn1 = nn.BatchNorm2D(base_channels, momentum=0.99)
        else:
            self.bn1 = build_norm_layer(norm_cfg, base_channels)
        self.conv2 = nn.Conv2D(
            base_channels,
            base_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias_attr=False,
        )
        if norm_cfg == None:
            self.bn2 = nn.BatchNorm2D(base_channels, momentum=0.99)
        else:
            self.bn2 = build_norm_layer(norm_cfg, base_channels)
        self.relu = nn.ReLU()

        # stage 1
        self.stage1_cfg = self.extra["stage1"]
        num_channels = self.stage1_cfg["num_channels"][0]
        block_type = self.stage1_cfg["block"]
        num_blocks = self.stage1_cfg["num_blocks"][0]

        block = blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(
            block, base_channels, num_channels, num_blocks
        )

        # stage 2
        self.stage2_cfg = self.extra["stage2"]
        num_channels = self.stage2_cfg["num_channels"]
        block_type = self.stage2_cfg["block"]

        block = blocks_dict[block_type]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channels], num_channels
        )
        # num_modules, num_branches, num_blocks, num_channels, block, fuse_method, num_inchannels
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels
        )

        # stage 3
        self.stage3_cfg = self.extra["stage3"]
        num_channels = self.stage3_cfg["num_channels"]
        block_type = self.stage3_cfg["block"]

        block = blocks_dict[block_type]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels
        )

        # stage 4
        self.stage4_cfg = self.extra["stage4"]
        num_channels = self.stage4_cfg["num_channels"]
        block_type = self.stage4_cfg["block"]

        block = blocks_dict[block_type]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels
        )

        # segmentation
        last_inp_channels = int(np.sum(pre_stage_channels))
        if norm_cfg == None:
            self.last_layer = nn.Sequential(
                nn.Conv2D(
                    in_channels=last_inp_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(out_channels, momentum=0.99),
                nn.ReLU(),
            )
        else:
            self.last_layer = nn.Sequential(
                nn.Conv2D(
                    in_channels=last_inp_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias_attr=False,
                ),
                build_norm_layer(norm_cfg, out_channels),
                nn.ReLU(),
            )

        self.init_weights(pretrained)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    if self.norm_cfg == None:
                        transition_layers.append(
                            nn.Sequential(
                                nn.Conv2D(
                                    num_channels_pre_layer[i],
                                    num_channels_cur_layer[i],
                                    3,
                                    1,
                                    1,
                                    bias_attr=False,
                                ),
                                nn.BatchNorm2D(
                                    num_channels_cur_layer[i], momentum=0.99
                                ),
                                nn.ReLU(),
                            )
                        )
                    else:
                        transition_layers.append(
                            nn.Sequential(
                                nn.Conv2D(
                                    num_channels_pre_layer[i],
                                    num_channels_cur_layer[i],
                                    3,
                                    1,
                                    1,
                                    bias_attr=False,
                                ),
                                build_norm_layer(
                                    self.norm_cfg, num_channels_cur_layer[i]
                                ),
                                nn.ReLU(),
                            )
                        )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    if self.norm_cfg == None:
                        conv3x3s.append(
                            nn.Sequential(
                                nn.Conv2D(
                                    inchannels, outchannels, 3, 2, 1, bias_attr=False
                                ),
                                nn.BatchNorm2D(outchannels, momentum=0.99),
                                nn.ReLU(),
                            )
                        )
                    else:
                        conv3x3s.append(
                            nn.Sequential(
                                nn.Conv2D(
                                    inchannels, outchannels, 3, 2, 1, bias_attr=False
                                ),
                                build_norm_layer(self.norm_cfg, outchannels),
                                nn.ReLU(),
                            )
                        )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.LayerList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            if self.norm_cfg == None:
                downsample = nn.Sequential(
                    nn.Conv2D(
                        inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias_attr=False,
                    ),
                    nn.BatchNorm2D(planes * block.expansion, momentum=0.99),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2D(
                        inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias_attr=False,
                    ),
                    build_norm_layer(self.norm_cfg, planes * block.expansion),
                )

        layers = []
        layers.append(
            block(inplanes, planes, stride, downsample, norm_cfg=self.norm_cfg)
        )
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_cfg=self.norm_cfg))

        return nn.Sequential(*layers)

    def _frozen_stages(self):
        # frozen stage  1 or stem networks
        if self.frozen_stages >= 0:
            for m in [self.conv1, self.bn1, self.conv2, self.bn2]:
                for param in m.parameters():
                    param.trainable = False
        if self.frozen_stages == 1:
            for param in self.layer1.parameters():
                param.trainable = False

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config["num_modules"]
        num_branches = layer_config["num_branches"]
        num_blocks = layer_config["num_blocks"]
        num_channels = layer_config["num_channels"]
        block = blocks_dict[layer_config["block"]]
        fuse_method = layer_config["fuse_method"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                    norm_cfg=self.norm_cfg,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.load_dict(paddle.load(pretrained))
        elif pretrained is None:
            for m in self.sublayers():
                if isinstance(m, nn.Conv2D):
                    kaiming_normal_init(m.weight)
                    if m.bias is not None:
                        constant_init(m.bias, value=0)
                elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
                    constant_init(m.weight, value=1.0)
                    constant_init(m.bias, value=0.0)

            if self.zero_init_residual:
                for m in self.sublayers():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3.weight, value=0)
                        constant_init(m.norm3.bias, value=0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2.weight, value=0)
                        constant_init(m.norm2.bias, value=0)
        else:
            raise TypeError("pretrained must be a str or None")

    @property
    def downsample_factor(self):
        factor = np.prod(self._ds_layer_strides)
        if len(self.us_layer_strides) > 0:
            factor /= self.us_layer_strides[-1]
        return factor

    def forward(self, x):
        x = self.conv1(x)  # downsample 2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["num_branches"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["num_branches"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["num_branches"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # Upsampling
        y0_h, y0_w = y_list[0].shape[2:]
        y1 = F.interpolate(
            y_list[1], size=(y0_h, y0_w), mode="bilinear", align_corners=False
        )
        y2 = F.interpolate(
            y_list[2], size=(y0_h, y0_w), mode="bilinear", align_corners=False
        )
        y3 = F.interpolate(
            y_list[3], size=(y0_h, y0_w), mode="bilinear", align_corners=False
        )

        x = paddle.concat([y_list[0], y1, y2, y3], 1)
        if self.last_layer is not None:
            x = self.last_layer(x)

        return x


if __name__ == "__main__":
    net = HRNet(
        in_channels=3,
        base_channels=64,
        out_channels=121, # int(sum([18, 32, 64, 128]) / 2)
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=[2,],
                num_channels=[64,],
                fuse_method='SUM'
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=[2, 2],
                num_channels=[18, 32],
                fuse_method='SUM'
            ),
            stage3=dict(
                num_modules=1,
                num_branches=3,
                block='BASIC',
                num_blocks=[2, 2, 2],
                num_channels=[18, 32, 64],
                fuse_method='SUM'
            ),
            stage4=dict(
                num_modules=1,
                num_branches=4,
                block='BASIC',
                num_blocks=[2, 2, 2, 2],
                num_channels=[18, 32, 64, 128],
                fuse_method='SUM'
            )
        )
    )
    x = paddle.randn([4, 3, 640, 640])
    out = net(x)
    print(out.shape)
