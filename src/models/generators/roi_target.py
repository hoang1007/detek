from typing import List

import torch

from src.utils.box_utils import bbox_iou, bbox_transform
from src.utils.functional import random_choice


class RoITargetGenerator:
    def __init__(
        self,
        num_samples: int,
        fg_fraction: float = 0.5,
        positive_iou_thr: float = 0.5,
        negative_iou_thr_low: float = 0.3,
        negative_iou_thr_high: float = 0.5,
        use_gt: bool = True,
    ):
        """Produce ROI targets for training.

        Args:
            num_samples: Number of samples to generate.
            fg_fraction: Number of foreground samples to generate.
            positive_iou_thr: IoU threshold for positive samples.
            negative_iou_thr_low, negative_iou_thr_high: IoU threshold for negative samples.
            use_gt: Whether to use ground-truth boxes as training samples.
        """

        self.num_samples = num_samples
        self.fg_fraction = fg_fraction
        self.positive_iou_thr = positive_iou_thr
        self.negative_iou_thr = (negative_iou_thr_low, negative_iou_thr_high)
        self.use_gt = use_gt

    def __call__(
        self,
        batch_proposals: List[torch.Tensor],
        batch_gt_boxes: List[torch.Tensor],
        batch_gt_labels: List[torch.Tensor],
    ):
        """
        Args:
            batch_proposals (list[Tensor]): Proposals of each image.
            batch_gt_boxes (list[Tensor]): Ground-truth bboxes of each image.
            batch_gt_labels (list[Tensor]): Ground-truth labels of each image.

        Returns:
            batch_reg_targets (list[Tensor]): Regression targets of shape (N, 4) per batch.
            batch_labels (list[Tensor]): Labels of shape (N, ) per batch.
            batch_roi_samples (list[Tensor]): Sampled proposals of shape (N, 4) per batch.
        """
        if self.use_gt:
            batch_proposals = [
                torch.cat((proposals, gt_boxes), dim=0)
                for proposals, gt_boxes in zip(batch_proposals, batch_gt_boxes)
            ]

        batch_reg_targets: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []
        batch_roi_samples: List[torch.Tensor] = []

        for proposals, gt_boxes, gt_labels in zip(
            batch_proposals, batch_gt_boxes, batch_gt_labels
        ):
            reg_targets, labels, roi_samples = self.sample(proposals, gt_boxes, gt_labels)
            batch_reg_targets.append(reg_targets)
            batch_labels.append(labels)
            batch_roi_samples.append(roi_samples)

        return batch_reg_targets, batch_labels, batch_roi_samples

    def sample(self, proposals: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor):
        ious = bbox_iou(proposals, gt_boxes)

        # For each proposal, which gt best overlaps with it
        max_ious, argmax_ious = ious.max(dim=1)

        num_fg = int(self.fg_fraction * self.num_samples)
        fg_ids = torch.nonzero(max_ious >= self.positive_iou_thr).squeeze_(1)

        if fg_ids.numel() > num_fg:
            fg_ids = fg_ids[random_choice(fg_ids, num_fg)]
        else:
            num_fg = fg_ids.numel()

        num_bg = self.num_samples - num_fg
        bg_ids = torch.nonzero(
            (max_ious >= self.negative_iou_thr[0]) & (max_ious < self.negative_iou_thr[1])
        ).squeeze_(1)

        if bg_ids.numel() > num_bg:
            bg_ids = bg_ids[random_choice(bg_ids, num_bg)]
        else:
            num_bg = bg_ids.numel()

        if self.use_gt:
            assert num_fg > 0, "No foreground samples found"

        keep_ids = torch.hstack((fg_ids, bg_ids))
        roi_samples = proposals[keep_ids]
        reg_targets = bbox_transform(roi_samples, gt_boxes[argmax_ious[keep_ids]])
        roi_labels = gt_labels[argmax_ious[keep_ids]]
        roi_labels[num_fg:] = 0  # Assign background labels

        return reg_targets, roi_labels, roi_samples
