from typing import List

import torch

from src.structures import ImageInfo
from src.utils.box_utils import bbox_iou, bbox_transform
from src.utils.functional import random_choice


class RPNTargetGenerator:
    def __init__(
        self,
        num_samples: int,
        fg_fraction: float = 0.5,
        positive_iou_thr: float = 0.7,
        negative_iou_thr: float = 0.3,
        allowed_border: int = 0,
    ):
        """Produce RPN targets for training.

        Args:
            num_samples: Number of samples to generate.
            fg_fraction: Number of foreground samples to generate.
            positive_iou_thr: IoU threshold for positive samples.
            negative_iou_thr: IoU threshold for negative samples.
            allowed_border: Whether to allow anchors are outside the image.
        """
        self.num_samples = num_samples
        self.fg_fraction = fg_fraction
        self.positive_iou_thr = positive_iou_thr
        self.negative_iou_thr = negative_iou_thr
        self.allowed_border = allowed_border

    def __call__(
        self,
        anchors: torch.Tensor,
        batch_gt_bboxes: List[torch.Tensor],
        im_info: ImageInfo,
    ):
        """
        Args:
            anchors (Tensor): Anchors of shape (N, 4).
            gt_bboxes (list[Tensor]): Ground-truth bboxes of each image.
            im_info (ImageInfo): Image information.

        Returns:
            reg_targets (Tensor): Regression targets of shape (B, N, 4).
            labels (Tensor): Labels of shape (B, N,).
        """
        num_anchors = anchors.size(0)

        anchors, keep = self._get_inside_anchors(anchors, im_info.height, im_info.width)
        batch_reg_targets = []
        batch_labels = []

        for gt_boxes in batch_gt_bboxes:
            reg_targets, labels = self.sample(anchors, gt_boxes)
            reg_targets = self._unmap(reg_targets, num_anchors, keep, fill=0)
            labels = self._unmap(labels, num_anchors, keep, fill=-1)

            batch_reg_targets.append(reg_targets)
            batch_labels.append(labels)

        batch_reg_targets = torch.stack(batch_reg_targets, dim=0)
        batch_labels = torch.stack(batch_labels, dim=0)

        return batch_reg_targets, batch_labels

    def sample(self, anchors: torch.Tensor, gt_boxes: torch.Tensor):
        """
        Args:
            anchors (Tensor):bg_ids.numel() > 0:
            num_fg = 0
            num_bg = self.num_samples Anchors of shape (N, 4).
            gt_boxes (Tensor): Ground-truth bboxes of shape (K, 4).

        Returns:
            reg_targets (Tensor): Regression targets of shape (N, 4).
            labels (Tensor): Labels of shape (N,).
        """
        num_anchors = anchors.size(0)
        num_gts = gt_boxes.size(0)

        # Initiate containers
        reg_targets = anchors.new_zeros((num_anchors, 4), dtype=torch.float32)
        labels = anchors.new_full((num_anchors,), -1, dtype=torch.long)

        if num_anchors == 0:
            return labels, reg_targets
        elif num_gts == 0:
            labels[:] = 0  # All anchors are background
            return labels, reg_targets

        # 1. Assign step
        # Compute IoU between anchors and ground-truth bboxes.
        ious = bbox_iou(anchors, gt_boxes)

        # For each anchor, which gt best overlaps with it
        max_ious, argmax_ious = ious.max(dim=1)  # (N, )
        # For each gt, which anchor best overlaps with it
        gt_max_ious, gt_argmax_ious = ious.max(dim=0)  # (K, )

        labels[max_ious < self.negative_iou_thr] = 0
        labels[gt_argmax_ious] = 1
        labels[max_ious >= self.positive_iou_thr] = 1

        # 2. Sampling step
        num_fg = int(self.fg_fraction * self.num_samples)
        fg_ids = torch.nonzero(labels == 1).squeeze_(1)

        if fg_ids.numel() > num_fg:
            disable_ids = fg_ids[random_choice(fg_ids, fg_ids.numel() - num_fg)]
            labels[disable_ids] = -1
        else:
            num_fg = fg_ids.numel()

        num_bg = self.num_samples - num_fg
        bg_ids = torch.nonzero(labels == 0).squeeze_(1)

        if bg_ids.numel() > num_bg:
            disable_ids = bg_ids[random_choice(bg_ids, bg_ids.numel() - num_bg)]
            labels[disable_ids] = -1
        else:
            num_bg = bg_ids.numel()

        # Compute regression targets
        valid_ids = labels != -1
        reg_targets[valid_ids] = bbox_transform(anchors[valid_ids], gt_boxes[argmax_ious[valid_ids]])

        return reg_targets, labels

    def _get_inside_anchors(self, anchors: torch.Tensor, height: int, width: int):
        inside_ids = (
            (anchors[:, 0] >= -self.allowed_border)
            & (anchors[:, 1] >= -self.allowed_border)
            & (anchors[:, 2] < width + self.allowed_border)
            & (anchors[:, 3] < height + self.allowed_border)
        )

        return anchors[inside_ids], inside_ids

    def _unmap(self, data: torch.Tensor, count: int, inds: torch.Tensor, fill: float = 0):
        """Unmap a subset of item (data) back to the original set of items (of size count)"""
        if data.dim() == 1:
            ret = data.new_full((count,), fill)
            ret[inds] = data
        else:
            ret = data.new_full((count,) + data.size()[1:], fill)
            ret[inds, :] = data
        return ret
