# -*- encoding: utf-8 -*-
"""
@File    :   loss.py
@Time    :   2023/12/18 19:28:38
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn.functional as F
from lh_pplsd.apis import manager
from lh_pplsd.losses.utils import weighted_loss


class Loss:
    """
    Loss
    """

    def __init__(
        self,
        weight=None,
        **kwargs,
    ):
        if isinstance(weight, (list, tuple)):
            weight = paddle.to_tensor(weight)

        self.weight = weight

    def __call__(self, pred, target):
        raise NotImplementedError


@manager.LOSSES.add_component
class BCELoss(Loss):
    """
    Binary Coss Entropy Loss
    """

    def __init__(
        self,
        with_logits=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.with_logits = with_logits

    @weighted_loss
    def __call__(self, pred, target):
        if self.with_logits:
            loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            loss = F.binary_cross_entropy(pred, target, reduction="none")

        if self.weight is not None:
            loss = loss * self.weight

        return loss


@manager.LOSSES.add_component
class CELoss(Loss):
    """
    Coss Entropy Loss
    """

    def __init__(
        self,
        with_logits=False,
        soft_label=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.with_logits = with_logits
        self.soft_label = soft_label

    @weighted_loss
    def __call__(self, pred, target):
        if self.with_logits:
            loss = F.softmax_with_cross_entropy(pred, target, soft_label=self.soft_label)
        else:
            loss = F.cross_entropy(pred, target, soft_label=self.soft_label, reduction="none")

        if self.weight is not None:
            loss = loss * self.weight

        return loss


@manager.LOSSES.add_component
class L1Loss(Loss):
    """
    L1 Loss
    """

    @weighted_loss
    def __call__(self, pred, target):
        loss = F.l1_loss(pred, target, reduction="none")
        if self.weight is not None:
            loss = loss * self.weight

        return loss


@manager.LOSSES.add_component
class SmoothL1Loss(Loss):
    """
    Smooth L1 Loss
    """
    
    def __init__(
        self, 
        delta=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.delta = delta

    @weighted_loss
    def __call__(self, pred, target):
        loss = F.smooth_l1_loss(pred, target, delta=self.delta, reduction="none")
        if self.weight is not None:
            loss = loss * self.weight

        return loss
