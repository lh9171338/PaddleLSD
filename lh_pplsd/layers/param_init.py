# -*- encoding: utf-8 -*-
"""
@File    :   param_init.py
@Time    :   2023/12/31 20:58:10
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import math
import numpy as np
import paddle
import paddle.nn as nn
import logging


def constant_init(param, **kwargs):
    """
    Initialize the `param` with constants
    """
    initializer = nn.initializer.Constant(**kwargs)
    initializer(param, param.block)


def normal_init(param, **kwargs):
    """
    Initialize the `param` with a Normal distribution
    """
    initializer = nn.initializer.Normal(**kwargs)
    initializer(param, param.block)


def uniform_init(param, a, b):
    """
    Initialize the `param` with a uniform distribution
    """
    return _no_grad_uniform_(param, a, b)


def xavier_normal_init(tensor, gain=1, reverse=False):
    """
    Initialize the `param` with xavier_normal method
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor, reverse=reverse)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    return _no_grad_normal_(tensor, 0.0, std)


def xavier_uniform_init(param, gain=1.0, reverse=False):
    """
    Initialize the `param` with xavier_uniform method
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(param, reverse=reverse)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = (
        math.sqrt(3.0) * std
    )  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(param, -a, a)


def kaiming_normal_init(
    tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", reverse=False
):
    """
    Initialize the `param` with kaiming_normal method
    """
    if 0 in tensor.shape:
        logging.warning("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode, reverse)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with paddle.no_grad():
        initializer = paddle.nn.initializer.Normal(mean=0, std=std)
        initializer(tensor)


def kaiming_uniform_init(
    param, a=0, mode="fan_in", nonlinearity="leaky_relu", reverse=False
):
    """
    Modified tensor inspace using kaiming_uniform method
    Args:
        param (paddle.Tensor): paddle Tensor
        mode (str): ['fan_in', 'fan_out'], 'fin_in' defalut
        nonlinearity (str): nonlinearity method name
        reverse (bool):  reverse (bool: False): tensor data format order, False by default as [fout, fin, ...].
    Return:
        tensor
    """
    fan = _calculate_correct_fan(param, mode, reverse)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    k = math.sqrt(3.0) * std
    return _no_grad_uniform_(param, -k, k)


def _calculate_fan_in_and_fan_out(tensor, reverse=False):
    """
    Calculate (fan_in, _fan_out) for tensor
    Args:
        tensor (Tensor): paddle.Tensor
        reverse (bool: False): tensor data format order, False by default as [fout, fin, ...].
            e.g. : conv.weight [cout, cin, kh, kw] is False; linear.weight [cin, cout] is True
    Return:
        Tuple[fan_in, fan_out]
    """
    if tensor.ndim < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    if reverse:
        num_input_fmaps, num_output_fmaps = tensor.shape[0], tensor.shape[1]
    else:
        num_input_fmaps, num_output_fmaps = tensor.shape[1], tensor.shape[0]

    receptive_field_size = 1
    if tensor.ndim > 2:
        receptive_field_size = np.prod(tensor.shape[2:])

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


# reference: https://pytorch.org/docs/stable/_modules/torch/nn/init.html
def _calculate_correct_fan(tensor, mode, reverse=False):
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(
            "Mode {} not supported, please use one of {}".format(
                mode, valid_modes
            )
        )

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor, reverse)

    return fan_in if mode == "fan_in" else fan_out


def _calculate_gain(nonlinearity, param=None):
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (
            not isinstance(param, bool)
            and isinstance(param, int)
            or isinstance(param, float)
        ):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(
                "negative_slope {} not a valid number".format(param)
            )
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return 3.0 / 4
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def _no_grad_uniform_(tensor, a, b):
    with paddle.no_grad():
        tensor.set_value(
            paddle.uniform(
                shape=tensor.shape, dtype=tensor.dtype, min=a, max=b
            )
        )
    return tensor


def _no_grad_normal_(tensor, mean, std):
    with paddle.no_grad():
        tensor.set_value(paddle.normal(mean, std, shape=tensor.shape))
    return tensor


def reset_parameters(m, reverse=False):
    if not hasattr(m, "weight"):
        return
    if m.weight.ndim < 2:
        return

    if isinstance(m, nn.Linear):
        reverse = True

    kaiming_uniform_init(m.weight, a=math.sqrt(5), reverse=reverse)
    if m.bias is not None:
        fan_in, _ = _calculate_fan_in_and_fan_out(m.weight, reverse=reverse)
        bound = 1 / math.sqrt(fan_in)
        _no_grad_uniform_(m.bias, -bound, bound)


def init_bias_by_prob(prob):
    bias_val = float(-np.log((1 - prob) / prob))
    return bias_val
