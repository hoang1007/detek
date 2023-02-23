from typing import Any, Dict, Tuple

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
        num_classes: int,
        spatial_scale: float,
        proposal_target: ProposalTargetGenerator,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_classes = num_classes
        hidden_dim = 4096

        self.fc = nn.Sequential(
            nn.Linear(num_channels * roi_size * roi_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.roi_cls = nn.Linear(4096, self.num_classes)
        self.roi_reg = nn.Linear(4096, self.num_classes * 4)

        self.roi_align = ops.RoIAlign((roi_size, roi_size), spatial_scale, sampling_ratio=-1)
        # self.roi_align = ops.RoIPool(roi_size, spatial_scale=spatial_scale)
        self.proposal_target = proposal_target(self.num_classes)  # type: ignore

    def forward(
        self, feature_map: torch.Tensor, rois: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feature_map.shape = (B, C, H, W)
        # rois.shape = (N, 4)
        assert len(rois.shape) == 2 and rois.size(1) == 4

        # pooled.shape = (N, C, roi_size, roi_size)
        pooled = self.roi_align(feature_map, [rois])

        pooled = pooled.flatten(start_dim=1)
        fc_out = self.fc(pooled)

        roi_bbox_pred = self.roi_reg(fc_out)
        roi_cls_scores = self.roi_cls(fc_out)

        return roi_bbox_pred, roi_cls_scores

    def forward_train(
        self,
        feature_map: torch.Tensor,
        rois: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ):
        """
        Args:

        """
        gt_boxes = gt_boxes[0]
        gt_labels = gt_labels[0]
        beta = 10

        roi_bbox_targets, sampled_rois, roi_labels = self.proposal_target(
            rois, gt_boxes, gt_labels
        )
        roi_bbox_pred, roi_cls_scores = self.forward(feature_map, sampled_rois)

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
