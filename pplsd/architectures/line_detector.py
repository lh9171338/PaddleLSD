# -*- encoding: utf-8 -*-
"""
@File    :   line_detector.py
@Time    :   2024/01/05 16:50:01
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


from pplsd.architectures import Detector
from pplsd.apis import manager


@manager.MODELS.add_component
class LineDetector(Detector):
    """
    Line Detector
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_train(self, image, **kwargs) -> dict:
        """foward function for training"""
        feats = self.backbone(image)
        if self.with_neck:
            feats = self.neck(feats)
        pred_dict = self.head(feats, **kwargs)
        loss_dict = self.head.loss(pred_dict, **kwargs)

        return loss_dict

    def forward_test(self, image, **kwargs) -> list:
        """foward function for testing"""
        feats = self.backbone(image)
        if self.with_neck:
            feats = self.neck(feats)
        pred_dict = self.head(feats, **kwargs)
        results = self.head.predict(pred_dict, **kwargs)

        return results
