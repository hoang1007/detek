from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from structures import ImageInfo
from utils.functional import init_weight

from .anchor_generator import AnchorGenerator
from .anchor_target import AnchorTargetGenerator
from .proposal import ProposalLayer


class RPNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feat_channels: int,
        feat_stride: int,
        proposal_layer: ProposalLayer,
        anchor_generator: AnchorGenerator,
        anchor_target: AnchorTargetGenerator,
    ):
        super().__init__()

        self.feat_channels = feat_channels
        self.feat_stride = feat_stride

        self.anchor_generator = anchor_generator
        self.proposal_layer = proposal_layer
        self.anchor_target = anchor_target

        self.conv = nn.Conv2d(in_channels, self.feat_channels, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(
            self.feat_channels, self.anchor_generator.num_base_anchors * 2, kernel_size=1
        )
        self.regressor = nn.Conv2d(
            self.feat_channels, self.anchor_generator.num_base_anchors * 4, kernel_size=1
        )

    def init_weight(self):
        init_weight(self.conv, std=0.01)
        init_weight(self.classifier, std=0.01)
        init_weight(self.regressor, std=0.01)

    def forward(self, feature_map: torch.Tensor, iminfo: ImageInfo):
        """
        Args:
            feature_map: Shape (B, C, H, W)
            iminfo: ImageInfo
        """
        anchors = self.anchor_generator(feature_map)

        fm = torch.relu(self.conv(feature_map))

        cls_scores = self.classifier(fm)  # shape (B, A * 2, H, W)
        bbox_pred = self.regressor(fm)  # shape (B, A * 4, H, W)

        cls_scores = cls_scores.permute(0, 2, 3, 1).reshape(-1, 2)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)

        cls_probs = torch.softmax(cls_scores.detach(), dim=1)
        objectness_scores = cls_probs[:, 1]

        rois = self.proposal_layer(bbox_pred, objectness_scores, anchors, iminfo)

        return bbox_pred, cls_scores, rois, anchors

    def forward_train(
        self,
        feature_map: torch.Tensor,
        gt_bboxes: torch.Tensor,
        metadata: ImageInfo,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            feature_map: Shape (B, C, H, W)
            gt_bboxes: Shape (B, N, 4)
            metadata: ImageInfo
        """
        beta = 10

        gt_bboxes = gt_bboxes[0]  # only support batch size 1
        bbox_pred, cls_scores, rois, anchors = self.forward(feature_map, metadata)
        rpn_bbox_targets, rpn_labels = self.anchor_target(anchors, gt_bboxes, metadata)

        cls_loss = F.cross_entropy(cls_scores, rpn_labels, ignore_index=-1)

        sampled_mask = rpn_labels >= 0
        objectness_masks = rpn_labels > 0

        reg_loss = F.smooth_l1_loss(
            bbox_pred[objectness_masks],
            rpn_bbox_targets[objectness_masks],
            beta=1,
            reduction="sum",
        )

        reg_loss = beta * reg_loss / sampled_mask.sum()

        return {"loss_rpn_cls": cls_loss, "loss_rpn_reg": reg_loss}, rois
