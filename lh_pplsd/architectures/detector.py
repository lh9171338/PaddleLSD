# -*- encoding: utf-8 -*-
"""
@File    :   detector.py
@Time    :   2023/12/16 23:07:58
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn as nn


class Detector(nn.Layer):
    """
    Base Detector
    """

    def __init__(
        self,
        pretrained=None,
        backbone=None,
        neck=None,
        head=None,
        **kwargs,
    ):
        super().__init__()

        self.backbone = backbone
        self.neck = neck
        self.head = head

        self.pretrained = pretrained
        self.load_pretrained_model()

    def load_pretrained_model(self):
        """load pretrained model"""
        if self.pretrained is not None:
            state_dict = paddle.load(self.pretrained)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.set_state_dict(state_dict)

    @property
    def with_backbone(self):
        """Whether the detector has a backbone"""
        return hasattr(self, "backbone") and self.backbone is not None

    @property
    def with_neck(self):
        """Whether the detector has a neck"""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_head(self):
        """Whether the detector has a head"""
        return hasattr(self, "head") and self.head is not None

    def forward(self, sample: dict) -> dict:
        if self.training:
            loss_dict = self.forward_train(**sample)
            return {"loss": loss_dict}
        else:
            pred_dict = self.forward_test(**sample)
            return {"pred": pred_dict}

    def forward_train(self, **kwargs) -> dict:
        """froward function for training"""
        raise NotImplementedError()

    def forward_test(self, **kwargs) -> dict:
        """froward function for testing"""
        raise NotImplementedError()
