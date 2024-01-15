# -*- encoding: utf-8 -*-
"""
@File    :   layer.py
@Time    :   2023/11/26 18:11:13
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle.nn as nn
import paddle.nn.functional as F


__all__ = [
    "BatchChannelNorm",
    "ConvBNLayer",
    "ConvModule",
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
