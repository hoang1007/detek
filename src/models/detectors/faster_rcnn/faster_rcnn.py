from typing import Tuple
import torch
from torch import nn
from torchvision import ops

from ..base_detector import BaseDetector
from .modules import RPNLayer, RCNN
from structures import ImageInfo
from utils.box_utils import bbox_transform_inv, clip_boxes


class FasterRCNN(BaseDetector):
    def __init__(self, backbone: nn.Module, rpn: RPNLayer, rcnn: RCNN):
        super().__init__()

        self.set_val_mode()

        self.backbone = backbone
        self.rpn = rpn
        self.rcnn = rcnn

    def backbone_forward(self, images: torch.Tensor):
        feature_map = self.backbone(images)
        h, w = images.shape[-2:]
        scale = round(feature_map.size(-1) / w)

        metadata = ImageInfo(w, h, scale)

        return feature_map, metadata

    def forward(self, images):
        assert images.size(0) == 1, "Only support batch size = 1"
        feature_map, metadata = self.backbone_forward(images)

        rpn_bbox_pred, rpn_cls_scores, rpn_rois, anchors = self.rpn(
            feature_map, metadata
        )
        roi_bbox_pred, roi_cls_scores = self.rcnn(feature_map, rpn_rois)

        return {
            "rpn_bbox_pred": rpn_bbox_pred,
            "rpn_cls_scores": rpn_cls_scores,
            "roi_bbox_pred": roi_bbox_pred,
            "roi_cls_scores": roi_cls_scores,
            "rpn_rois": rpn_rois,
            "anchors": anchors,
        }

    def forward_train(
        self, images: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor
    ):
        assert images.size(0) == 1, "Only support batch size = 1"
        feature_map, metadata = self.backbone_forward(images)

        rpn_losses, rois = self.rpn.forward_train(feature_map, gt_boxes, metadata)
        roi_losses = self.rcnn.forward_train(feature_map, rois, gt_boxes, gt_labels)

        return {**rpn_losses, **roi_losses}

    def forward_test(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_classes = self.rcnn.num_classes
        delta_means = (0, 0, 0, 0)
        delta_stds = (0.1, 0.1, 0.2, 0.2)

        delta_means = (
            torch.tensor(delta_means)
            .view(1, -1)
            .to(images.device)
            .repeat(1, num_classes)
        )
        delta_stds = (
            torch.tensor(delta_stds)
            .view(1, -1)
            .to(images.device)
            .repeat(1, num_classes)
        )

        outputs = self(images)

        roi_bbox_pred = outputs["roi_bbox_pred"] * delta_stds + delta_means
        roi_bbox_pred = roi_bbox_pred.view(-1, num_classes, 4)

        rois = outputs["rpn_rois"].view(-1, 1, 4).expand_as(roi_bbox_pred)

        pred_boxes = bbox_transform_inv(
            rois.reshape(-1, 4), roi_bbox_pred.reshape(-1, 4)
        )

        image_height, image_width = images.shape[-2:]
        pred_boxes = clip_boxes(pred_boxes, image_height, image_width)

        pred_boxes = pred_boxes.view(-1, num_classes * 4)

        box_probs = torch.softmax(outputs["roi_cls_scores"], dim=-1)

        pred_boxes, pred_labels, box_scores = self._suppress(
            pred_boxes, box_probs, num_classes
        )

        return pred_boxes, pred_labels, box_scores

    def set_val_mode(self, mode: str = "EVAL"):
        """
        mode: `EVAL` or `VISUALIZE`
        """
        if mode in ("EVAL", "VISUALIZE"):
            self._val_mode = mode
        else:
            raise ValueError("Invalid val mode")

    def _get_preset(self):
        if self._val_mode == "EVAL":
            return {"nms_thresh": 0.3, "score_thresh": 0.05}
        elif self._val_mode == "VISUALIZE":
            return {"nms_thresh": 0.3, "score_thresh": 0.5}
        else:
            raise ValueError("Invalid val mode")

    def _suppress(
        self, pred_boxes: torch.Tensor, box_scores: torch.Tensor, num_classes: int
    ):
        preset = self._get_preset()
        boxlist = []
        labellist = []
        scorelist = []

        pred_boxes = pred_boxes.reshape(-1, num_classes, 4)

        for clazz in range(1, num_classes):  # ignore background
            boxes_ = pred_boxes[:, clazz]
            scores_ = box_scores[:, clazz]

            mask = scores_ > preset["score_thresh"]
            boxes_ = boxes_[mask]
            scores_ = scores_[mask]

            keep = ops.nms(boxes_, scores_, preset["nms_thresh"])

            boxes_ = boxes_[keep]
            scores_ = scores_[keep]
            labels_ = clazz * pred_boxes.new_ones(boxes_.size(0), dtype=torch.long)

            boxlist.append(boxes_)
            labellist.append(labels_)
            scorelist.append(scores_)

        return (
            torch.cat(boxlist, dim=0),
            torch.cat(labellist, dim=0),
            torch.cat(scorelist, dim=0),
        )
