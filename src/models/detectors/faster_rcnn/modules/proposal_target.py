import torch

from utils.box_utils import bbox_transform, bbox_iou
from utils.functional import random_choice


class ProposalTargetGenerator:
    def __init__(
        self,
        num_classes: int,
        use_gt: bool,
        batch_size: int,
        fg_fraction: float,
        fg_thresh: float,
        bg_thresh_high: float,
        bg_thresh_low: float,
        bbox_normalize_means: torch.Tensor,
        bbox_normalize_stds: torch.Tensor,
    ):
        """
        Gán nhãn cho các RoI và tạo ra các bbox_targets phục vụ cho việc huấn luyện Proposal Layer.
        """

        self.num_classes = num_classes
        self.use_gt = use_gt
        self.batch_size = batch_size
        self.fg_fraction = fg_fraction
        self.fg_thesh = fg_thresh
        self.bg_thresh_high = bg_thresh_high
        self.bg_thresh_low = bg_thresh_low
        self.bbox_normalize_means = bbox_normalize_means
        self.bbox_normalize_stds = bbox_normalize_stds

        self.num_images = 1

    def __call__(self, rois: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor):
        if self.use_gt:
            rois = torch.cat((rois, gt_boxes), dim=0)

        rois_per_image = self.batch_size // self.num_images
        fg_rois_per_image = round(self.fg_fraction * rois_per_image)

        bbox_targets_data, roi_labels, sampled_rois = self._sample_rois(
            rois, gt_boxes, gt_labels, fg_rois_per_image, rois_per_image
        )

        # Transform bbox_targets_data from shape (S, 4) to (S, num_classes * 4)
        num_samples = bbox_targets_data.size(0)
        bbox_targets = bbox_targets_data.new_zeros((num_samples, self.num_classes, 4))
        bbox_targets[
            torch.arange(num_samples, device=bbox_targets.device), roi_labels, :
        ] = bbox_targets_data
        bbox_targets = bbox_targets.contiguous().view(num_samples, -1)

        return bbox_targets, sampled_rois, roi_labels

    def _sample_rois(self, rois, gt_boxes, gt_labels, fg_rois_per_img, rois_per_img):
        ious = bbox_iou(rois, gt_boxes)

        # Find the ground-truth box with the highest IoU for each RoI
        max_ious, gt_assignment = ious.max(dim=1)
        roi_labels = gt_labels[gt_assignment]

        fg_ids = torch.where(max_ious >= self.fg_thesh)[0]
        bg_ids = torch.where(
            (max_ious < self.bg_thresh_high) & (max_ious >= self.bg_thresh_low)
        )[0]

        if fg_ids.numel() > 0 and bg_ids.numel() > 0:
            fg_rois_per_img = min(fg_rois_per_img, fg_ids.numel())
            bg_rois_per_img = rois_per_img - fg_rois_per_img
        elif fg_ids.numel() > 0:
            fg_rois_per_img = rois_per_img
            bg_rois_per_img = 0
        elif self.use_gt:
            raise ValueError("Num foreground labels cannot equals to 0")
        else:
            bg_rois_per_img = rois_per_img
            fg_rois_per_img = 0

        assert fg_rois_per_img + bg_rois_per_img == rois_per_img

        fg_ids = fg_ids[random_choice(fg_ids, fg_rois_per_img, auto_replace=True)]
        bg_ids = bg_ids[random_choice(bg_ids, bg_rois_per_img, auto_replace=True)]

        keep_ids = torch.hstack((fg_ids, bg_ids))
        assert keep_ids.size(0) == rois_per_img, "Keep indices not match size"

        roi_labels = roi_labels[keep_ids].contiguous()
        roi_labels[fg_rois_per_img:] = 0  # assign background labels

        sampled_rois = rois[keep_ids].contiguous()

        bbox_targets = self._compute_targets(
            sampled_rois, gt_boxes[gt_assignment[keep_ids]]
        )

        return bbox_targets, roi_labels, sampled_rois

    def _compute_targets(self, rois, gt_boxes):
        targets = bbox_transform(rois, gt_boxes)

        targets = (targets - targets.new(self.bbox_normalize_means)) / targets.new(
            self.bbox_normalize_stds
        )

        return targets
