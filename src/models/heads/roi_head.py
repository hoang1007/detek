from typing import Dict, List, Optional

import torch
from torch import nn
from torchvision.ops import batched_nms, roi_align, roi_pool

from src.models.base import BaseModel
from src.models.generators import RoITargetGenerator
from src.structures import DetResult, ImageInfo
from src.utils.box_utils import bbox_inv_transform


class RoIHead(BaseModel):
    def __init__(
        self,
        roi_size: int,
        in_channels: int,
        num_classes: int,
        use_roi_align: bool = True,
        sampling_ratio: int = 0,
        feature_extractor: Optional[nn.Module] = None,
        roi_target_generator: Optional[RoITargetGenerator] = None,
        bbox_deltas_normalize_means: Optional[List[float]] = None,
        bbox_deltas_normalize_stds: Optional[List[float]] = None,
        test_nms_cfg: Dict[str, float] = dict(nms_thr=0.7, score_thr=0.01),
    ):
        """
        Args:
            roi_size (int): Output size after RoIs pooling.
            in_channels (int): Number of channels in the input feature map.
            num_classes (int): Number of classes.
            use_roi_align (bool): Whether to use roi_align or roi_pool.
            sampling_ratio (int): Number of samples to take for each
                region of interest. 0 to take samples densely for current models.
            feature_extractor (nn.Module): Feature extractor.
            roi_target_generator (RoITargetGenerator): Generator for ROI targets.
            bbox_normalize_means (list[float]): Mean values used for bounding box regression.
            bbox_normalize_stds (list[float]): Std values used for bounding box regression.
        """
        super().__init__()

        self.roi_size = roi_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_roi_align = use_roi_align
        self.sampling_ratio = sampling_ratio
        self.roi_target_generator = roi_target_generator
        self.test_nms_cfg = test_nms_cfg

        if bbox_deltas_normalize_means is not None:
            self.register_buffer(
                "bbox_deltas_normalize_means",
                torch.tensor(bbox_deltas_normalize_means, dtype=torch.float32),
            )
        if bbox_deltas_normalize_stds is not None:
            self.register_buffer(
                "bbox_deltas_normalize_stds",
                torch.tensor(bbox_deltas_normalize_stds, dtype=torch.float32),
            )

        self.feature_extractor = (
            feature_extractor if feature_extractor is not None else nn.Identity()
        )
        hidden_channels, scale = self._get_feat_extractor_dim()
        self.roi_feat_size = int(self.roi_size * scale)

        self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        self.fc_cls = nn.Linear(hidden_channels, num_classes)
        self.fc_bbox = nn.Linear(hidden_channels, 4 * num_classes)

    def _get_feat_extractor_dim(self):
        dump = torch.zeros(1, self.in_channels, 64, 64)
        dump_feats = self.feature_extractor(dump)
        out_channels = dump_feats.size(1)
        scale = dump_feats.size(2) / dump.size(2)

        return out_channels, scale

    def _bbox_deltas_normalize(self, bbox_detas: torch.Tensor):
        if hasattr(self, "bbox_deltas_normalize_means") and hasattr(
            self, "bbox_deltas_normalize_stds"
        ):
            bbox_detas = (
                bbox_detas - self.bbox_deltas_normalize_means
            ) / self.bbox_deltas_normalize_stds
        return bbox_detas

    def _bbox_deltas_denormalize(self, bbox_deltas: torch.Tensor):
        if hasattr(self, "bbox_deltas_normalize_means") and hasattr(
            self, "bbox_deltas_normalize_stds"
        ):
            bbox_deltas = (
                bbox_deltas * self.bbox_deltas_normalize_stds + self.bbox_deltas_normalize_means
            )
        return bbox_deltas

    def forward(self, x: torch.Tensor, rois: List[torch.Tensor], im_info: ImageInfo):
        """
        Args:
            x (Tensor): Feature map of shape (B, C, H, W).
            rois (List[Tensor]): List of rois for each image in the batch.
            im_info (ImageInfo): Information about the image.
        """
        assert len(rois) == x.size(0), "Number of rois must match batch size"
        spatial_scale = x.size(3) / im_info.width

        if self.use_roi_align:
            roi_features = roi_align(
                x,
                rois,
                output_size=(self.roi_size, self.roi_size),
                spatial_scale=spatial_scale,
                sampling_ratio=self.sampling_ratio,
            )  # type: ignore
        else:
            roi_features = roi_pool(
                x,
                rois,
                output_size=(self.roi_size, self.roi_size),
                spatial_scale=spatial_scale,
            )  # type: ignore

        roi_features = self.feature_extractor(roi_features)
        roi_features = self.avg_pool(roi_features).flatten(start_dim=1)

        bbox_reg = self.fc_bbox(roi_features)
        cls_logits = self.fc_cls(roi_features)

        split_size = [r.size(0) for r in rois]
        bbox_reg = torch.split(bbox_reg, split_size, dim=0)
        cls_logits = torch.split(cls_logits, split_size, dim=0)

        return bbox_reg, cls_logits

    def forward_train(
        self,
        x: torch.Tensor,
        proposals: List[torch.Tensor],
        batch_gt_bboxes: List[torch.Tensor],
        batch_gt_labels: List[torch.Tensor],
        im_info: ImageInfo,
    ):
        """
        Args:
            x (Tensor): Feature map of shape (B, C, H, W).
            rois (List[Tensor]): List of proposals for each image in the batch.
            batch_gt_bboxes (List[Tensor]): List of ground truth bboxes for each image in the batch.
            batch_gt_labels (List[Tensor]): List of ground truth labels for each image in the batch.
            im_info (ImageInfo): Information about the image.
        """
        assert (
            self.roi_target_generator is not None
        ), "RoiTargetGenerator is required for training!"
        batch_bbox_targets, batch_labels, batch_roi_samples = self.roi_target_generator(
            proposals, batch_gt_bboxes, batch_gt_labels
        )

        bbox_reg, cls_logits = self.forward(x, batch_roi_samples, im_info)

        bbox_targets = torch.cat(batch_bbox_targets, dim=0)
        bbox_targets = self._bbox_deltas_normalize(bbox_targets)
        labels = torch.cat(batch_labels, dim=0)
        bbox_reg = torch.cat(bbox_reg, dim=0)
        cls_logits = torch.cat(cls_logits, dim=0)

        sample_mask = labels >= 0
        assert sample_mask.sum() == bbox_targets.size(0), "Check roi sampler"
        objectness_mask = labels > 0

        # Expand bbox targets from (N, 4) to (N, 4 * num_classes)
        expanded_bbox_targets = bbox_targets.new_zeros(bbox_targets.size(0), self.num_classes, 4)
        expanded_bbox_targets[range(bbox_targets.size(0)), labels] = bbox_targets
        expanded_bbox_targets = expanded_bbox_targets.view(-1, 4 * self.num_classes)

        roi_reg_loss = nn.functional.smooth_l1_loss(
            bbox_reg[objectness_mask],
            expanded_bbox_targets[objectness_mask],
            reduction="sum",
        )
        roi_reg_loss = 10 * roi_reg_loss / sample_mask.sum()

        roi_cls_loss = nn.functional.cross_entropy(cls_logits, labels, ignore_index=-1)

        return dict(roi_reg_loss=roi_reg_loss, roi_cls_loss=roi_cls_loss)

    def forward_test(
        self,
        x: torch.Tensor,
        batch_proposals: List[torch.Tensor],
        im_info: ImageInfo,
    ):
        """
        Args:
            x (Tensor): Feature map of shape (B, C, H, W).
            rois (List[Tensor]): List of proposals for each image in the batch.
            im_info (ImageInfo): Information about the image.
        """
        batch_bbox_deltas, batch_cls_logits = self.forward(x, batch_proposals, im_info)

        results: List[DetResult] = []
        for bbox_deltas, cls_logits, proposals in zip(
            batch_bbox_deltas, batch_cls_logits, batch_proposals
        ):
            conf_scores, labels = nn.functional.softmax(cls_logits, dim=1).max(dim=1)
            # Filter background predictions
            keep = labels > 0
            conf_scores = conf_scores[keep]
            labels = labels[keep]
            proposals = proposals[keep]
            bbox_deltas = bbox_deltas[keep]

            bbox_deltas = bbox_deltas.view(-1, self.num_classes, 4)
            bbox_deltas = self._bbox_deltas_denormalize(
                bbox_deltas[range(bbox_deltas.size(0)), labels]
            )
            pred_bboxes = bbox_inv_transform(proposals, bbox_deltas)

            # Filter out predictions with low confidence
            keep = conf_scores > self.test_nms_cfg.get("score_thr", 0)
            pred_bboxes = pred_bboxes[keep]
            conf_scores = conf_scores[keep]
            labels = labels[keep]

            # Apply NMS
            keep = batched_nms(pred_bboxes, conf_scores, labels, self.test_nms_cfg["nms_thr"])
            pred_bboxes = pred_bboxes[keep]
            conf_scores = conf_scores[keep]
            labels = labels[keep]

            results.append(DetResult(pred_bboxes, conf_scores, labels))

        return results


def roi_feature_extractor(arch: str = "resnet50", pretrained: bool = True):
    """
    Args:
        arch (str): Architecture name. Default: resnet50.
        pretrained (bool): Whether to use pretrained weights. Default: True.
    """

    if arch == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        return resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None).layer4
    elif arch == "resnet101":
        from torchvision.models import ResNet101_Weights, resnet101

        return resnet101(weights=ResNet101_Weights.DEFAULT if pretrained else None).layer4
    else:
        raise ValueError(f"Invalid architecture: {arch}")
