# -*- encoding: utf-8 -*-
"""
@File    :   layer.py
@Time    :   2023/11/26 18:11:13
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.ops import DeformConv2D
from paddle.nn.initializer import Constant
from paddle import ParamAttr
from pplsd.layers.param_init import (
    linear_init,
    xavier_uniform_init,
)


__all__ = [
    "BatchChannelNorm",
    "ConvBNLayer",
    "ConvModule",
    "FFN",
    "DCN",
    "DDCN",
    "build_linear_layer",
    "build_conv_layer",
    "build_norm_layer",
    "build_activation_layer",
]


class BatchChannelNorm(nn.Layer):
    """
    Batch Channel Norm
    """

    def __init__(
        self,
        num_channels,
        momentum=0.9,
        epsilon=1e-5,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.momentum = momentum
        self.epsilon = epsilon

        self.bn = nn.BatchNorm2D(
            num_features=num_channels,
            momentum=momentum,
            epsilon=epsilon,
            weight_attr=False,
            bias_attr=False,
        )
        self.ln = nn.GroupNorm(
            num_groups=1,
            num_channels=num_channels,
            epsilon=epsilon,
            weight_attr=False,
            bias_attr=False,
        )

        self.alpha = self.create_parameter(
            shape=[num_channels],
            default_initializer=nn.initializer.Constant(value=1.0),
        )
        self.gamma = self.create_parameter(
            shape=[num_channels],
            default_initializer=nn.initializer.Constant(value=1.0),
        )
        self.beta = self.create_parameter(
            shape=[num_channels],
            default_initializer=nn.initializer.Constant(value=0.0),
        )

    def forward(self, x):
        bn_out = self.bn(x)
        ln_out = self.ln(x)
        out = (
            self.alpha.reshape([1, -1, 1, 1]) * bn_out
            + (1 - self.alpha.reshape([1, -1, 1, 1])) * ln_out
        )
        out = out * self.gamma.reshape([1, -1, 1, 1]) + self.beta.reshape(
            [1, -1, 1, 1]
        )

        return out


class ConvBNLayer(nn.Layer):
    """
    ConvBNLayer: conv + bn + activation layer
    """

    def __init__(
        self,
        ch_in,
        ch_out,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        norm_type="bn",
        act="leaky",
    ):
        super().__init__()

        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False,
        )
        if norm_type == "bcn":
            self.norm = BatchChannelNorm(num_channels=ch_out)
        else:
            self.norm = nn.BatchNorm2D(num_features=ch_out)

        self.act = act

        self.init_weights()

    def init_weights(self):
        """init weights"""
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                nn.initializer.Normal(0.0, 0.02)(layer.weight)
                if layer.bias is not None:
                    nn.initializer.Constant(0.0)(layer.bias)
            elif isinstance(layer, nn.BatchNorm2D):
                nn.initializer.Normal(0.0, 0.02)(layer.weight)
                nn.initializer.Constant(0.0)(layer.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act == "leaky":
            out = F.leaky_relu(out, 0.1)
        else:
            out = getattr(F, self.act)(out)

        return out


class FFN(nn.Layer):
    """
    Feed-Forward Network
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        dropout=0.1,
        activation="relu",
        weight_attr=None,
        bias_attr=None,
    ):
        super().__init__()

        self.linear1 = nn.Linear(
            in_channels, hidden_channels, weight_attr, bias_attr
        )
        self.activation = getattr(F, activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(
            hidden_channels, in_channels, weight_attr, bias_attr
        )
        self._reset_parameters()

    def _reset_parameters(self):
        """reset parameters"""
        linear_init(self.linear1)
        linear_init(self.linear2)
        xavier_uniform_init(self.linear1.weight)
        xavier_uniform_init(self.linear2.weight)

    def forward(self, x):
        """Forward fuction"""
        out = self.linear2(self.dropout(self.activation(self.linear1(x))))

        return out


class DCN(nn.Layer):
    """
    DCN
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        bias_attr=None,
    ):
        super().__init__()

        self.offset_channel = 2 * kernel_size**2
        self.mask_channel = kernel_size**2

        self.conv_offset = nn.Conv2D(
            in_channels=in_channels,
            out_channels=3 * kernel_size**2,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            weight_attr=ParamAttr(initializer=Constant(0.0)),
            bias_attr=ParamAttr(initializer=Constant(0.0)),
        )
        self.conv = DeformConv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            dilation=1,
            groups=groups,
            bias_attr=bias_attr,
        )

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        offset, mask = paddle.split(
            offset_mask,
            num_or_sections=[self.offset_channel, self.mask_channel],
            axis=1,
        )
        mask = F.sigmoid(mask)
        out = self.conv(x, offset, mask=mask)

        return out


class DDCN(nn.Layer):
    """
    Directional DCN
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        bias_attr=None,
    ):
        super().__init__()

        self.dir_channel = 2
        self.dist_channel = kernel_size**2
        self.mask_channel = kernel_size**2
        half_kernel_size = (kernel_size - 1) // 2
        pos_y, pos_x = paddle.meshgrid(
            paddle.arange(-half_kernel_size, half_kernel_size + 1),
            paddle.arange(-half_kernel_size, half_kernel_size + 1),
        )
        self.pos = paddle.concat([pos_y.flatten(), pos_x.flatten()], 0)

        self.conv_offset = nn.Conv2D(
            in_channels=in_channels,
            out_channels=2 * kernel_size**2 + 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            weight_attr=ParamAttr(initializer=Constant(0.0)),
            bias_attr=ParamAttr(initializer=Constant(0.0)),
        )
        self.conv = DeformConv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            dilation=1,
            groups=groups,
            bias_attr=bias_attr,
        )

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        dir, dist, mask = paddle.split(
            offset_mask,
            num_or_sections=[
                self.dir_channel,
                self.dist_channel,
                self.mask_channel,
            ],
            axis=1,
        )
        mask = F.sigmoid(mask)
        offset = (dir[:, :, None] * dist[:, None]).flatten(1, 2) - self.pos[
            ..., None, None
        ]
        out = self.conv(x, offset, mask=mask)

        return out


def build_linear_layer(in_channels, out_channels, bias=True):
    """Build linear layer"""
    bound = 1 / math.sqrt(in_channels)
    param_attr = ParamAttr(initializer=nn.initializer.Uniform(-bound, bound))
    bias_attr = False
    if bias:
        bias_attr = ParamAttr(
            initializer=nn.initializer.Uniform(-bound, bound)
        )
    return nn.Linear(
        in_channels, out_channels, weight_attr=param_attr, bias_attr=bias_attr
    )


def build_conv_layer(cfg, *args, **kwargs):
    """build conv layer"""
    if cfg is None:
        cfg_ = dict(type_name="Conv2D")
    else:
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type_name")
    conv_layer = getattr(nn, layer_type)
    layer = conv_layer(*args, **kwargs, **cfg_)
    return layer


def build_norm_layer(cfg, num_features):
    """build norm layer"""
    if cfg is None:
        cfg_ = dict(type_name="BatchNorm2D")
    else:
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type_name")
    if layer_type == "BatchChannelNorm":
        norm_layer = BatchChannelNorm
    else:
        norm_layer = getattr(nn, layer_type)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("epsilon", 1e-5)
    layer = norm_layer(num_features, **cfg_)

    for n, p in layer.named_parameters():
        if "_mean" in n or "_variance" in n:
            continue
        p.trainable = requires_grad

    return layer


def build_activation_layer(cfg):
    """build activation layer"""
    if cfg is None:
        cfg_ = dict(type_name="ReLU")
    else:
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type_name")
    act_layer = getattr(nn, layer_type)
    return act_layer()


class ConvModule(nn.Layer):
    def __init__(
        self,
        ch_in,
        ch_out,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type_name="ReLU"),
        order=("conv", "norm", "act"),
    ):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(["conv", "norm", "act"])

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            ch_in,
            ch_out,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias,
        )

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index("norm") > order.index("conv"):
                norm_channels = ch_out
            else:
                norm_channels = ch_in
            self.bn = build_norm_layer(norm_cfg, norm_channels)

        # build activation layer
        if self.with_activation:
            self.activate = build_activation_layer(act_cfg)

    def forward(self, x):
        for layer in self.order:
            if layer == "conv":
                x = self.conv(x)
            elif layer == "norm" and self.with_norm:
                x = self.bn(x)
            elif layer == "act" and self.with_activation:
                x = self.activate(x)
        return x


if __name__ == "__main__":
    model = DDCNV2(1, 1, 3)
    x = paddle.randn((1, 1, 4, 4))
    y = model(x)
    a = 1
