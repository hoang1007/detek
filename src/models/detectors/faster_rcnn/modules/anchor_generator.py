from typing import Union, List
import torch
from torch import nn

from utils.box_utils import cxcywh2xyxy


class AnchorGenerator(nn.Module):
    def __init__(
        self,
        feat_stride: int,
        scales: Union[torch.Tensor, List[float]],
        aspect_ratios: Union[torch.Tensor, List[float]] = [0.5, 1, 2],
    ):
        """
        Generate anchors on image

        Args:
            feat_stride: Stride of
            scales: Tỉ lệ kích thước so với base anchor
            aspect_ratios: Tỉ lệ height / width của anchors

        Forward Args:
            feature_map (Tensor): Feature map của ảnh sau khi đưa qua backbone. Shape (1, C, H, W)
        """
        super().__init__()

        if not isinstance(scales, torch.Tensor):
            scales = torch.tensor(scales)
        if not isinstance(aspect_ratios, torch.Tensor):
            aspect_ratios = torch.tensor(aspect_ratios)

        self.feat_stride = feat_stride
        self.scales = scales
        self.aspect_ratios = aspect_ratios

        self.register_buffer("base_anchors", self.mkbase_anchors())

    @property
    def num_base_anchors(self):
        return len(self.scales) * len(self.aspect_ratios)

    def mkbase_anchors(self):
        A = self.num_base_anchors

        base_area = self.feat_stride ** 2

        hs = torch.sqrt(base_area * self.aspect_ratios)  # shape (num_aspect_ratios,)
        ws = base_area / hs  # shape (num_aspect_ratios,)

        scales = self.scales.reshape(1, -1)
        hs = (hs.reshape(-1, 1) * scales).flatten()
        ws = (ws.reshape(-1, 1) * scales).flatten()

        x_ctrs = torch.ones(A) * (self.feat_stride - 1) / 2
        y_ctrs = x_ctrs.clone()

        base_anchors = torch.stack((x_ctrs, y_ctrs, ws, hs), dim=1)
        base_anchors = cxcywh2xyxy(base_anchors, inplace=True)

        return base_anchors.round()

    def forward(self, feature_map: torch.Tensor):
        batch_size, _, feat_height, feat_width = feature_map.shape

        shiftx = (
            torch.arange(0, feat_width, device=feature_map.device) * self.feat_stride
        )
        shifty = (
            torch.arange(0, feat_height, device=feature_map.device) * self.feat_stride
        )

        shiftx, shifty = torch.meshgrid(shiftx, shifty, indexing="ij")

        shifts = torch.vstack(
            (shiftx.ravel(), shifty.ravel(), shiftx.ravel(), shifty.ravel())
        ).transpose(0, 1)

        # shifts.shape == (H * W, 4)
        # base_anchors.shape == (A, 4)
        # => anchors.shape == (H * W * A, 4)
        assert isinstance(self.base_anchors, torch.Tensor)
        anchors = self.base_anchors.view(1, -1, 4) + shifts.view(-1, 1, 4)
        anchors = anchors.reshape(-1, 4)

        return anchors
