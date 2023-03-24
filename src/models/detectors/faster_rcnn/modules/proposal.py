import torch
from torch import nn
from torchvision import ops

from structures.image_info import ImageInfo
from utils.box_utils import bbox_transform_inv, clip_boxes


class ProposalLayer(nn.Module):
    def __init__(
        self,
        train_pre_nms_topN: int,
        train_post_nms_topN: int,
        test_pre_nms_topN: int,
        test_post_nms_topN: int,
        nms_thresh: float,
        min_box_size: int,
    ):
        super().__init__()

        self._pre_nms_topN = {"train": train_pre_nms_topN, "test": test_pre_nms_topN}
        self._post_nms_topN = {"train": train_post_nms_topN, "test": test_post_nms_topN}
        self.nms_thresh = nms_thresh
        self.min_box_size = min_box_size

    def _get_nms_topN(self):
        key = "train" if self.training else "test"

        return self._pre_nms_topN[key], self._post_nms_topN[key]

    def forward(
        self,
        rpn_bbox_pred: torch.Tensor,
        objectness: torch.Tensor,
        anchors: torch.Tensor,
        metadata: ImageInfo,
    ):
        """
        Args:
            rpn_bbox_pred: Deltas of proposals from anchors. Shape (B, N, 4)
            objectness: Objectness scores of proposals. Shape (B, N)
            anchors: Anchors. Shape (N, 4)
            metadata: ImageInfo

        Returns:
            proposals: List of proposals per batch. List of shape (M, 4)
        """
        pre_nms_topN, post_nms_topN = self._get_nms_topN()

        batch_size = rpn_bbox_pred.size(0)

        rpn_bbox_pred = rpn_bbox_pred.view(-1, 4)  # Shape (B * N, 4)
        anchors = anchors.repeat(batch_size, 1)  # Shape (B * N, 4)

        proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
        proposals = clip_boxes(proposals, metadata.height, metadata.width)
        proposals = proposals.view(batch_size, -1, 4)

        batch_proposals = []

        for i in range(batch_size):
            nb_proposals = proposals[i]
            nb_objectness = objectness[i]

            keep = self._filter_boxes(nb_proposals, self.min_box_size)
            nb_proposals = nb_proposals[keep]
            nb_objectness = nb_objectness[keep]

            nb_objectness, order = torch.sort(nb_objectness, descending=True)

            if pre_nms_topN > 0:
                order = order[:pre_nms_topN]
                nb_objectness = nb_objectness[:pre_nms_topN]

            nb_proposals = nb_proposals[order]

            nms_keep_ids = ops.nms(nb_proposals, nb_objectness, self.nms_thresh)

            if post_nms_topN > 0:
                nms_keep_ids = nms_keep_ids[:post_nms_topN]

            nb_proposals = nb_proposals[nms_keep_ids]
            batch_proposals.append(nb_proposals)

        return batch_proposals

    def _filter_boxes(self, boxes: torch.Tensor, min_size: int):
        """
        Remove all boxes with any size smaller than min_size
        Return:
            indices of filltered boxes
        """
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]

        keep = (ws >= min_size) & (hs >= min_size)

        return keep
