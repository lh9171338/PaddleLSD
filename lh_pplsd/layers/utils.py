# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/01/16 15:43:40
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import copy
import paddle
import paddle.nn as nn


__all__ = [
    "_get_clones",
    "inverse_sigmoid",
]


def _get_clones(module, N):
    """get clones"""
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])


def inverse_sigmoid(x, eps=1e-6):
    """inverse sigmoid"""
    x = x.clip(min=0.0, max=1.0)
    return paddle.log(x / (1 - x + eps) + eps)
