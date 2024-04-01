# -*- encoding: utf-8 -*-
"""
@File    :   reparam.py
@Time    :   2024/03/04 10:39:51
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


__all__ = [
    "ConvBN",
    "RepLayer",
    "RepVGGBlock",
    "RepLargeKernelConv",
    "RepResidualBlock",
]


class ConvBN(nn.Layer):
    """
    Conv BN Layer
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
    ):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False,
        )
        self.bn = nn.BatchNorm2D(num_features=out_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return y


class RepLayer(nn.Layer):
    """
    Re-parameterization Layer
    """

    def __init__(self):
        super().__init__()

        self.inference_mode = False

    def convert_to_deploy(self):
        """convert to deploy"""
        if not self.inference_mode:
            for layer in self.sublayers():
                if hasattr(layer, "convert_to_deploy"):
                    layer.convert_to_deploy()

            self.inference_mode = True

    def convert_to_train(self):
        """convert to train"""
        if self.inference_mode:
            for layer in self.sublayers():
                if hasattr(layer, "convert_to_train"):
                    layer.convert_to_train()

            self.inference_mode = False


class RepVGGBlock(RepLayer):
    """
    Re-parameterization VGG Block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        self.rbr_identity = (
            nn.BatchNorm2D(
                num_features=in_channels,
            )
            if out_channels == in_channels and stride == 1
            else None
        )
        self.rbr_dense = ConvBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.rbr_1x1 = ConvBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding_11,
            groups=groups,
        )

    def forward(self, x):
        if self.inference_mode:
            return self.nonlinearity(self.rbr_reparam(x))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)
        return self.nonlinearity(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)

    def convert_to_deploy(self):
        """convert to deploy"""
        if not hasattr(self, "rbr_reparam"):
            self.rbr_reparam = nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups,
            )
        if not self.inference_mode:
            kernel, bias = self.get_equivalent_kernel_bias()
            self.rbr_reparam.weight.set_value(kernel)
            self.rbr_reparam.bias.set_value(bias)
            self.inference_mode = True

    def get_equivalent_kernel_bias(self):
        """get the equivalent kernel and bias"""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid
        bias = bias3x3 + bias1x1 + biasid
        return kernel, bias

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """pad 1x1 kernel to 3x3 kernel"""
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """fuse bn tensor into conv"""
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvBN):
            kernel = branch.conv.weight
            running_mean = branch.bn._mean
            running_var = branch.bn._variance
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = paddle.to_tensor(kernel_value)
            kernel = self.id_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


class RepLargeKernelConv(RepLayer):
    """
    Re-parameterization LargeKernel Conv Block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        small_kernel=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.small_kernel = small_kernel

        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        self.lkb_origin = ConvBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
        )
        if small_kernel is not None:
            assert (
                small_kernel <= kernel_size
            ), "The kernel size for re-param cannot be larger than the large kernel!"

            self.small_conv = ConvBN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=small_kernel,
                stride=stride,
                padding=small_kernel // 2,
                groups=groups,
            )

    def forward(self, x):
        if self.inference_mode:
            return self.rbr_reparam(x)

        out = self.lkb_origin(x)
        if hasattr(self, "small_conv"):
            out += self.small_conv(x)

        return out

    def convert_to_deploy(self):
        """convert to deploy"""
        if not hasattr(self, "rbr_reparam"):
            self.rbr_reparam = nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size // 2,
                groups=self.groups,
            )
        if not self.inference_mode:
            kernel, bias = self.get_equivalent_kernel_bias()
            self.rbr_reparam.weight.set_value(kernel)
            self.rbr_reparam.bias.set_value(bias)
            self.inference_mode = True

    def get_equivalent_kernel_bias(self):
        """get the equivalent kernel and bias"""
        eq_k, eq_b = self._fuse_bn_tensor(self.lkb_origin)
        if hasattr(self, "small_conv"):
            small_k, small_b = self._fuse_bn_tensor(self.small_conv)
            eq_b += small_b
            #   add to the central part
            eq_k += F.pad(
                small_k, [(self.kernel_size - self.small_kernel) // 2] * 4
            )

        return eq_k, eq_b

    def _fuse_bn_tensor(self, branch):
        """fuse bn tensor into conv"""
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvBN):
            kernel = branch.conv.weight
            running_mean = branch.bn._mean
            running_var = branch.bn._variance
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = paddle.to_tensor(kernel_value)
            kernel = self.id_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


class RepResidualBlock(RepLayer):
    """
    Re-parameterization Residual Block
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
        outplanes = planes * self.expansion
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.stride = stride

        self.nonlinearity = nn.ReLU()

        self.rbr_identity = (
            nn.BatchNorm2D(
                num_features=inplanes,
            )
            if outplanes == inplanes and stride == 1
            else None
        )
        self.rbr_dense = ConvBN(
            in_channels=inplanes,
            out_channels=outplanes,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.rbr_1x1 = ConvBN(
            in_channels=inplanes,
            out_channels=outplanes,
            kernel_size=1,
            stride=stride,
            padding=0,
        )

    def forward(self, x):
        if self.inference_mode:
            return self.nonlinearity(self.rbr_reparam(x))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)
        return self.nonlinearity(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)

    def convert_to_deploy(self):
        """convert to deploy"""
        if not hasattr(self, "rbr_reparam"):
            self.rbr_reparam = nn.Conv2D(
                in_channels=self.inplanes,
                out_channels=self.outplanes,
                kernel_size=3,
                stride=self.stride,
                padding=1,
            )
        if not self.inference_mode:
            kernel, bias = self.get_equivalent_kernel_bias()
            self.rbr_reparam.weight.set_value(kernel)
            self.rbr_reparam.bias.set_value(bias)
            self.inference_mode = True

    def get_equivalent_kernel_bias(self):
        """get the equivalent kernel and bias"""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid
        bias = bias3x3 + bias1x1 + biasid
        return kernel, bias

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """pad 1x1 kernel to 3x3 kernel"""
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """fuse bn tensor into conv"""
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvBN):
            kernel = branch.conv.weight
            running_mean = branch.bn._mean
            running_var = branch.bn._variance
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            if not hasattr(self, "id_tensor"):
                kernel_value = np.zeros(
                    (self.inplanes, self.inplanes, 3, 3), dtype=np.float32
                )
                for i in range(self.inplanes):
                    kernel_value[i, i, 1, 1] = 1
                self.id_tensor = paddle.to_tensor(kernel_value)
            kernel = self.id_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


if __name__ == "__main__":
    model = RepVGGBlock(
        in_channels=256,
        out_channels=128,
        kernel_size=3,
    )
    # model = RepLargeKernelConv(
    #     in_channels=256,
    #     out_channels=256,
    #     kernel_size=3,
    # )
    # model = RepResidualBlock(
    #     inplanes=256,
    # )

    x = paddle.randn([4, 256, 64, 64])
    model.eval()
    with paddle.no_grad():
        out1 = model(x)
    model.convert_to_deploy()
    with paddle.no_grad():
        out2 = model(x)
    diff = out1 - out2
    print(diff.abs().max())
