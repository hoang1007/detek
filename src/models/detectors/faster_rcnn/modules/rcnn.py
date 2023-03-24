from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import ops

from utils.functional import init_weight

from .proposal_target import ProposalTargetGenerator


class RCNN(nn.Module):
    def __init__(
        self,
        roi_size: int,
        num_channels: int,
        hidden_channels: int,
        hidden_dim: int,
        num_classes: int,
        spatial_scale: float,
        proposal_target: ProposalTargetGenerator,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.fc = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=1),
            nn.Flatten(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * roi_size**2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.roi_cls = nn.Linear(hidden_dim, self.num_classes)
        self.roi_reg = nn.Linear(hidden_dim, self.num_classes * 4)

        self.roi_pooling = ops.RoIAlign((roi_size, roi_size), spatial_scale, sampling_ratio=-1)
        # self.roi_align = ops.RoIPool(roi_size, spatial_scale=spatial_scale)
        self.proposal_target = proposal_target(self.num_classes)  # type: ignore

    def forward(
        self, feature_map: torch.Tensor, proposals: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature_map: Shape (B, C, H, W)
            proposals: List of shape (N, 4)

        Returns:
            roi_bbox_pred: Shape (B, N, 4 * num_classes)
            roi_cls_scores: Shape (B, N, num_classes)
        """

        batch_size = feature_map.size(0)

        pooled = self.roi_pooling(feature_map, proposals)
        fc_out = self.fc(pooled)

        roi_bbox_pred = self.roi_reg(fc_out).view(batch_size, -1, 4 * self.num_classes)
        roi_cls_scores = self.roi_cls(fc_out).view(batch_size, -1, self.num_classes)

        return roi_bbox_pred, roi_cls_scores

    def forward_train(
        self,
        feature_map: torch.Tensor,
        proposals: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ):
        """
        Args:
            feature_map: Shape (B, C, H, W)
            proposals: List of shape (N, 4)
            gt_boxes: List of shape (M, 4)
            gt_labels: List of shape (M,)

        """
        beta = 10

        roi_bbox_targets, sampled_rois, roi_labels = self.proposal_target(
            proposals, gt_boxes, gt_labels
        )
        roi_bbox_pred, roi_cls_scores = self.forward(feature_map, sampled_rois)

        roi_bbox_targets = roi_bbox_targets.view(-1, self.num_classes * 4)
        roi_labels = roi_labels.view(-1)
        roi_bbox_pred = roi_bbox_pred.view(-1, self.num_classes * 4)
        roi_cls_scores = roi_cls_scores.view(-1, self.num_classes)

        cls_loss = F.cross_entropy(roi_cls_scores, roi_labels, ignore_index=-1)

        sampled_mask = roi_labels >= 0
        objectness_masks = roi_labels > 0

        reg_loss = F.smooth_l1_loss(
            roi_bbox_pred[objectness_masks],
            roi_bbox_targets[objectness_masks],
            beta=1,
            reduction="sum",
        )
        reg_loss = beta * reg_loss / sampled_mask.sum()

        return {"roi_cls_loss": cls_loss, "roi_reg_loss": reg_loss}

    def init_weight(self):
        init_weight(self.roi_cls, std=0.01)
        init_weight(self.roi_reg, std=0.01)

        for model in self.fc:
            init_weight(model, std=0.01)
