from typing import Dict, List, Optional

import torch
from torch import nn
from torchvision.ops import nms

from src.models.base import BaseModel
from src.models.generators import AnchorGenerator, RPNTargetGenerator
from src.structures import ImageInfo
from src.utils.box_utils import bbox_inv_transform
from src.utils.functional import init_weight


class RPNHead(BaseModel):
    def __init__(
        self,
        in_channels: int,
        feat_channels: int,
        anchor_generator: AnchorGenerator,
        rpn_target_generator: Optional[RPNTargetGenerator] = None,
        train_cfg: Optional[Dict] = dict(
            min_box_size=0,
            nms=dict(
                num_pre_proposals=12000,
                num_post_proposals=2000,
                iou_thr=0.7,
            ),
        ),
        test_cfg: Optional[Dict] = dict(
            min_box_size=0,
            nms=dict(
                num_pre_proposals=6000,
                num_post_proposals=1000,
                iou_thr=0.7,
            ),
        ),
    ):
        super().__init__()

        self.anchor_generator = anchor_generator
        self.rpn_target_generator = rpn_target_generator
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.rpn_cls = nn.Conv2d(
            feat_channels, 2 * anchor_generator.num_base_anchors, kernel_size=1
        )
        self.rpn_reg = nn.Conv2d(
            feat_channels, 4 * anchor_generator.num_base_anchors, kernel_size=1
        )

    def init_weights(self):
        for m in self.conv.modules():
            init_weight(m)
        init_weight(self.rpn_cls)
        init_weight(self.rpn_reg)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): Feature map of shape (B, C, H, W).

        Returns:
            rpn_cls (Tensor): Objectness logits for each anchor of shape (B, H * W * num_base_anchors, 2).
            rpn_reg (Tensor): Bounding box regression for each anchor of shape (B, H * W * num_base_anchors, 4).
        """
        batch_size = x.size(0)
        x = self.conv(x)

        # (B, 2 * num_base_anchors, H, W) -> (B, H * W * num_base_anchors, 2)
        rpn_cls = torch.permute(self.rpn_cls(x), (0, 2, 3, 1)).contiguous().view(batch_size, -1, 2)
        # (B, 4 * num_base_anchors, H, W) -> (B, H * W * num_base_anchors, 4)
        rpn_reg = torch.permute(self.rpn_reg(x), (0, 2, 3, 1)).contiguous().view(batch_size, -1, 4)

        return rpn_cls, rpn_reg

    def forward_train(self, x: torch.Tensor, gt_bboxes: List[torch.Tensor], im_info: ImageInfo):
        """
        Args:
            x (Tensor): Feature map of shape (B, C, H, W).
            gt_bboxes (List[Tensor]): Ground truth bounding boxes of shape (N, 4).
            im_info (ImageInfo): Image information.
        """
        rpn_cls, rpn_reg = self(x)

        assert (
            self.rpn_target_generator is not None
        ), "RPN target generator is required for training!"
        anchors = self.anchor_generator(im_info.height, im_info.width)
        objectness = torch.softmax(rpn_cls.detach().clone(), dim=-1)[:, :, 1]
        proposals = self.get_proposals(
            rpn_reg.detach().clone(), objectness, anchors, im_info, self.train_cfg
        )
        rpn_reg_targets, rpn_cls_targets = self.rpn_target_generator(anchors, gt_bboxes, im_info)

        rpn_cls = rpn_cls.view(-1, 2)
        rpn_cls_targets = rpn_cls_targets.view(-1)
        rpn_reg = rpn_reg.view(-1, 4)
        rpn_reg_targets = rpn_reg_targets.view(-1, 4)

        sample_mask = rpn_cls_targets >= 0
        objectness_mask = rpn_cls_targets > 0
        rpn_reg_loss = 10 * nn.functional.l1_loss(
            rpn_reg[objectness_mask], rpn_reg_targets[objectness_mask]
        )

        rpn_cls_loss = nn.functional.cross_entropy(rpn_cls[sample_mask], rpn_cls_targets[sample_mask])

        return dict(rpn_cls_loss=rpn_cls_loss, rpn_reg_loss=rpn_reg_loss), proposals

    def forward_test(self, x: torch.Tensor, im_info: ImageInfo):
        rpn_cls, rpn_reg = self(x)
        objectness = torch.softmax(rpn_cls, dim=-1)[:, :, 1]
        anchors = self.anchor_generator(im_info.height, im_info.width)
        proposals = self.get_proposals(rpn_reg, objectness, anchors, im_info, self.test_cfg)

        return proposals, rpn_cls, rpn_reg

    def get_proposals(
        self,
        bbox_pred: torch.Tensor,
        objectness: torch.Tensor,
        anchors: torch.Tensor,
        im_info: ImageInfo,
        proposals_cfg: Dict[str, float],
    ):
        if "nms" in proposals_cfg:
            nms_cfg = proposals_cfg["nms"]
        else:
            nms_cfg = None

        batch_size = bbox_pred.size(0)

        bbox_pred = bbox_pred.view(-1, 4)
        anchors = anchors.repeat(batch_size, 1)

        batch_proposals = bbox_inv_transform(anchors, bbox_pred)
        batch_proposals[:, 0::2].clamp_(min=0, max=im_info.width)
        batch_proposals[:, 1::2].clamp_(min=0, max=im_info.height)
        batch_proposals = batch_proposals.view(batch_size, -1, 4)

        sampled_proposals: List[torch.Tensor] = []
        batch_objectness = objectness.view(batch_size, -1)
        for proposals, objectness in zip(batch_proposals, batch_objectness):
            # Filter out proposals which are too small
            keep_ids = torch.logical_and(
                (proposals[:, 2] - proposals[:, 0]) >= proposals_cfg.get("min_box_size", 0),
                (proposals[:, 3] - proposals[:, 1]) >= proposals_cfg.get("min_box_size", 0),
            )
            proposals = proposals[keep_ids]
            objectness = objectness[keep_ids]

            order = torch.argsort(objectness, descending=True)

            if nms_cfg is not None:
                num_pre_proposals = nms_cfg.get("num_pre_proposals", -1)
                if num_pre_proposals > 0:
                    order = order[:num_pre_proposals]

            objectness = objectness[order]
            proposals = proposals[order]

            if nms_cfg is not None:
                nms_keep_ids = nms(proposals, objectness, nms_cfg.get("iou_thr", 0.5))
                num_post_proposals = nms_cfg.get("num_post_proposals", -1)
                if num_post_proposals > 0:
                    nms_keep_ids = nms_keep_ids[:num_post_proposals]
                proposals = proposals[nms_keep_ids]
                objectness = objectness[nms_keep_ids]

            sampled_proposals.append(proposals)

        return sampled_proposals
