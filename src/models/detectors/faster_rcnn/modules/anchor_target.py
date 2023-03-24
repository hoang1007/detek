from typing import List

import torch

from structures import ImageInfo
from utils.box_utils import bbox_iou, bbox_transform
from utils.functional import random_choice


class AnchorTargetGenerator:
    def __init__(
        self,
        batch_size: int = 128,
        allowed_border: float = 0,
        clobber_positives: bool = False,
        positive_overlap: float = 0.7,
        negative_overlap: float = 0.3,
        fg_fraction: float = 0.5,
    ):
        """Produce bbox targets and objectness labels for RPN.

        Args:
            batch_size: Number of anchors to sample for training
            allowed_border: If an anchor is too close to the border, ignore it
            clobber_positives: If an anchor has IoU > positive_overlap with any gt_box, ignore it
            positive_overlap: Threshold for an anchor to be a positive
            negative_overlap: Threshold for an anchor to be a negative
            fg_fraction: Fraction of anchors that are labeled as foreground
        """

        self._allowed_border = allowed_border
        self.batch_size = batch_size
        self.clobber_positives = clobber_positives
        self.positive_overlap = positive_overlap
        self.negative_overlap = negative_overlap
        self.fg_fraction = fg_fraction

    def __call__(self, anchors: torch.Tensor, gt_boxes: List[torch.Tensor], img_info: ImageInfo):
        """
        Args:
            anchors: Anchors on the images Shape (N, 4)
            gt_boxes: The ground-truth bounding boxes of objects on the images. List of shape (K, 4)
            im_info: The image info

        Returns:
            bbox_targets: Deltas of assigned ground-truth bounding boxes from anchors. Shape (B, N, 4)
            labels: Objectness labels for anchors. Shape (B, N)
        """
        A = anchors.size(0)

        anchors, keep_ids = self._get_inside_anchors(anchors, img_info.height, img_info.width)

        batched_bbox_targets = []
        batched_labels = []

        for boxes in gt_boxes:
            bbox_targets, labels = self._mklabels(anchors, boxes)

            bbox_targets = self._unmap(bbox_targets, A, keep_ids, fill=0)
            labels = self._unmap(labels, A, keep_ids, fill=-1)

            batched_bbox_targets.append(bbox_targets)
            batched_labels.append(labels)

        batched_bbox_targets = torch.stack(batched_bbox_targets, dim=0)
        batched_labels = torch.stack(batched_labels, dim=0)

        return batched_bbox_targets, batched_labels

    def _get_inside_anchors(self, anchors: torch.Tensor, height: int, width: int):
        inside_ids = (
            (anchors[:, 0] >= -self._allowed_border)
            & (anchors[:, 1] >= -self._allowed_border)
            & (anchors[:, 2] < width + self._allowed_border)
            & (anchors[:, 3] < height + self._allowed_border)
        )

        return anchors[inside_ids], inside_ids

    def _mklabels(self, anchors: torch.Tensor, gt_boxes: torch.Tensor):
        A = anchors.size(0)
        G = gt_boxes.size(0)

        assert A > 0, "Num of anchors must be greater than 0"
        assert G > 0, "Num of ground-truth boxes must be greater than 0"

        # Initiate containers
        bbox_targets = torch.zeros((A, 4), dtype=torch.float32, device=anchors.device)
        labels = torch.empty(A, dtype=torch.long, device=anchors.device).fill_(-1)

        ious = bbox_iou(anchors, gt_boxes)  # shape (A, G)

        # lấy gt_boxes có IoU lớn nhất so với mỗi anchor
        max_ious, argmax_ious = torch.max(ious, dim=1)

        # lấy anchors có IoU lớn nhất so với mỗi gt_box
        gt_max_ious, _ = torch.max(ious, dim=0)
        gt_argmax_ious = torch.where(ious == gt_max_ious)[0]

        if not self.clobber_positives:
            labels[max_ious < self.negative_overlap] = 0

        labels[gt_argmax_ious] = 1
        labels[max_ious >= self.positive_overlap] = 1

        if self.clobber_positives:
            labels[max_ious < self.negative_overlap] = 0

        # Phân chia các nhãn: potivies, negatives, non-labels theo batch_size
        num_fg = round(self.batch_size * self.fg_fraction)
        fg_ids = torch.nonzero(labels == 1).squeeze_(1)

        if fg_ids.size(0) > num_fg:
            # Lược bỏ nếu số nhãn foreground quá nhiều
            disable_ids = fg_ids[random_choice(fg_ids, fg_ids.size(0) - num_fg, replacement=False)]
            labels[disable_ids] = -1
        else:
            # Cập nhật num_fg nếu số nhãn foreground quá ít
            num_fg = fg_ids.size(0)

        assert num_fg == (labels == 1).sum()

        num_bg = self.batch_size - num_fg
        bg_ids = torch.nonzero(labels == 0).squeeze_(1)

        if bg_ids.size(0) > num_bg:
            disable_ids = bg_ids[random_choice(bg_ids, bg_ids.size(0) - num_bg, replacement=False)]
            labels[disable_ids] = -1

        keep_ids = torch.hstack((fg_ids, bg_ids))
        bbox_targets[keep_ids] = bbox_transform(anchors[keep_ids], gt_boxes[argmax_ious[keep_ids]])

        return bbox_targets, labels

    def _unmap(self, data: torch.Tensor, count: int, ids: torch.Tensor, fill: float = 0):
        if len(data.shape) == 1:
            ret = torch.empty((count,)).type_as(data).fill_(fill)
            ret[ids] = data
        else:
            ret = torch.empty((count,) + data.shape[1:]).type_as(data).fill_(fill)
            ret[ids, :] = data

        return ret
