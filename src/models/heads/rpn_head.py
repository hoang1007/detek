from typing import Dict, List, Optional

import torch
from torch import nn
from torchvision.ops import nms

from src.models.base import BaseModel
from src.models.generators import AnchorGenerator, RPNTargetGenerator
from src.structures import ImageInfo
from src.utils.box_utils import bbox_inv_transform


class RPNHead(BaseModel):
    def __init__(
        self,
        in_channels: int,
        feat_channels: int,
        anchor_generator: AnchorGenerator,
        rpn_target_generator: Optional[RPNTargetGenerator] = None,
        train_nms_cfg: Dict[str, float] = dict(
            nms_pre=12000,
            nms_post=2000,
            nms_thr=0.7,
            min_bbox_size=0,
        ),
        test_nms_cfg: Dict[str, float] = dict(
            nms_pre=6000,
            nms_post=1000,
            nms_thr=0.7,
            min_bbox_size=0,
        ),
    ):
        super().__init__()

        self.anchor_generator = anchor_generator
        self.rpn_target_generator = rpn_target_generator
        self.train_nms_cfg = train_nms_cfg
        self.test_nms_cfg = test_nms_cfg

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
        objectness = torch.softmax(rpn_cls.clone().detach(), dim=-1)[:, :, 1]
        proposals = self.get_proposals(
            rpn_reg.clone().detach(), objectness, anchors, im_info, self.train_nms_cfg
        )
        rpn_reg_targets, rpn_cls_targets = self.rpn_target_generator(anchors, gt_bboxes, im_info)

        rpn_cls = rpn_cls.view(-1, 2)
        rpn_cls_targets = rpn_cls_targets.view(-1)
        rpn_reg = rpn_reg.view(-1, 4)
        rpn_reg_targets = rpn_reg_targets.view(-1, 4)

        sample_mask = rpn_cls_targets >= 0
        objectness_mask = rpn_cls_targets > 0
        rpn_reg_loss = nn.functional.smooth_l1_loss(
            rpn_reg[objectness_mask], rpn_reg_targets[objectness_mask], reduction="sum"
        )
        rpn_reg_loss = 10 * rpn_reg_loss / sample_mask.sum()

        rpn_cls_loss = nn.functional.cross_entropy(rpn_cls, rpn_cls_targets, ignore_index=-1)

        return dict(rpn_cls_loss=rpn_cls_loss, rpn_reg_loss=rpn_reg_loss), proposals

    def forward_test(self, x: torch.Tensor, im_info: ImageInfo):
        rpn_cls, rpn_reg = self(x)
        objectness = torch.softmax(rpn_cls, dim=-1)[:, :, 1]
        anchors = self.anchor_generator(im_info.height, im_info.width)
        proposals = self.get_proposals(rpn_reg, objectness, anchors, im_info, self.test_nms_cfg)

        return proposals, rpn_cls, rpn_reg

    def get_proposals(
        self,
        bbox_pred: torch.Tensor,
        objectness: torch.Tensor,
        anchors: torch.Tensor,
        im_info: ImageInfo,
        nms_cfg: Dict[str, float],
    ):
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
            keep_ids = (proposals[:, 2] - proposals[:, 0]) >= nms_cfg["min_bbox_size"]
            keep_ids &= (proposals[:, 3] - proposals[:, 1]) >= nms_cfg["min_bbox_size"]
            proposals = proposals[keep_ids]
            objectness = objectness[keep_ids]

            objectness, order = objectness.sort(descending=True)

            if nms_cfg["nms_pre"] > 0:
                order = order[: nms_cfg["nms_pre"]]
                objectness = objectness[order]
            proposals = proposals[order]

            nms_keep_ids = nms(proposals, objectness, nms_cfg["nms_thr"])

            if nms_cfg["nms_post"] > 0:
                nms_keep_ids = nms_keep_ids[: nms_cfg["nms_post"]]
            proposals = proposals[nms_keep_ids]
            sampled_proposals.append(proposals)

        return sampled_proposals
