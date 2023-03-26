from typing import Dict, List

import torch
from torch import nn

from src.models.heads import RoIHead, RPNHead
from src.structures import DetResult, ImageInfo

from .base_detector import BaseDetector


class FasterRCNN(BaseDetector):
    def __init__(self, backbone: nn.Module, rpn_head: RPNHead, roi_head: RoIHead, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.rpn_head = rpn_head
        self.roi_head = roi_head

    def forward_train(
        self,
        images: torch.Tensor,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        im_info = ImageInfo(images.size(3), images.size(2))
        features = self.backbone(images)

        rpn_losses, proposals = self.rpn_head.forward_train(features, gt_boxes, im_info)
        roi_losses = self.roi_head.forward_train(features, proposals, gt_boxes, gt_labels, im_info)

        losses = dict()
        losses.update(rpn_losses)
        losses.update(roi_losses)
        return losses

    def forward_test(self, images: torch.Tensor) -> List[DetResult]:
        im_info = ImageInfo(images.size(3), images.size(2))
        features = self.backbone(images)
        proposals, _, _ = self.rpn_head.forward_test(features, im_info)
        return self.roi_head.forward_test(features, proposals, im_info)
