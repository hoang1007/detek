from typing import Dict, List, Tuple

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

    def forward(self, feature_map: torch.Tensor, img_info: ImageInfo):
        """
        Args:
            feature_map: Shape (B, C, H, W)
            img_info: ImageInfo

        Returns:
            bbox_pred: Shape (B, N, 4)
            cls_scores: Shape (B, N, 2)
            proposals: List of proposals per batch. List of shape (M, 4)
        """
        batch_size = feature_map.size(0)

        anchors = self.anchor_generator(feature_map)  # shape (A, 4)

        fm = F.relu(self.conv(feature_map), inplace=True)

        cls_scores = self.classifier(fm)  # shape (B, A * 2, H, W)
        bbox_pred = self.regressor(fm)  # shape (B, A * 4, H, W)

        cls_scores = cls_scores.permute(0, 2, 3, 1).reshape(batch_size, -1, 2)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

        cls_probs = F.softmax(cls_scores.detach(), dim=-1)
        objectness_scores = cls_probs[:, :, 1]

        proposals = self.proposal_layer(bbox_pred, objectness_scores, anchors, img_info)

        return bbox_pred, cls_scores, proposals, anchors

    def forward_train(
        self,
        feature_map: torch.Tensor,
        gt_bboxes: List[torch.Tensor],
        metadata: ImageInfo,
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            feature_map: Shape (B, C, H, W)
            gt_bboxes: List of shape (N, 4)
            metadata: ImageInfo

        Returns:
            losses: Dict of losses
            proposals: List of proposals per batch. List of shape (M, 4)
        """
        beta = 10

        bbox_pred, cls_scores, proposals, anchors = self.forward(feature_map, metadata)
        rpn_bbox_targets, rpn_labels = self.anchor_target(anchors, gt_bboxes, metadata)

        bbox_pred = bbox_pred.view(-1, 4)
        cls_scores = cls_scores.view(-1, 2)
        rpn_bbox_targets = rpn_bbox_targets.view(-1, 4)
        rpn_labels = rpn_labels.view(-1)

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

        return {"loss_rpn_cls": cls_loss, "loss_rpn_reg": reg_loss}, proposals
