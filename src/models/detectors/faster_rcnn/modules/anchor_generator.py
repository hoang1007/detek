from typing import List, Union

import torch
from torch import nn

from utils.box_utils import cxcywh2xyxy


class AnchorGenerator(nn.Module):
    def __init__(
        self,
        stride: int,
        scales: Union[torch.Tensor, List[float]],
        aspect_ratios: Union[torch.Tensor, List[float]] = [0.5, 1, 2],
    ):
        """Generate anchors on image.

        Args:
            stride: Stride of anchors
            scales: Scales of anchors. `anchor = scale * base_anchor`
            aspect_ratios: Aspect ratios of anchors. `w = aspect_ratio * h`
        """
        super().__init__()

        if not isinstance(scales, torch.Tensor):
            scales = torch.tensor(scales)
        else:
            assert scales.ndim == 1, "scales must be 1D tensor"
        if not isinstance(aspect_ratios, torch.Tensor):
            aspect_ratios = torch.tensor(aspect_ratios)
        else:
            assert aspect_ratios.ndim == 1, "aspect_ratios must be 1D tensor"

        self.stride = stride
        self.scales = scales
        self.aspect_ratios = aspect_ratios

        self.register_buffer("base_anchors", self.mkbase_anchors())

    @property
    def num_base_anchors(self):
        return len(self.scales) * len(self.aspect_ratios)

    def mkbase_anchors(self):
        A = self.num_base_anchors

        base_area = self.stride**2

        # base_area = w * h
        # aspect_ratio = h / w
        # h = sqrt(base_area * aspect_ratio)
        hs = torch.sqrt(base_area * self.aspect_ratios)  # shape (num_aspect_ratios,)
        ws = base_area / hs  # shape (num_aspect_ratios,)

        scales = self.scales.reshape(1, -1)
        hs = (
            hs.reshape(-1, 1) * scales
        ).flatten()  # shape (num_aspect_ratios, num_scales) flattened
        ws = (
            ws.reshape(-1, 1) * scales
        ).flatten()  # shape (num_aspect_ratios, num_scales) flattened

        x_ctrs = torch.ones(A) * self.stride * 0.5
        y_ctrs = x_ctrs.clone()

        base_anchors = torch.stack((x_ctrs, y_ctrs, ws, hs), dim=1)
        base_anchors = cxcywh2xyxy(base_anchors, inplace=True)

        return base_anchors.round()

    def forward(self, width: int, height: int):
        assert hasattr(self, "base_anchors")
        shiftx = torch.arange(end=width, step=self.stride, device=self.base_anchors.device)  # type: ignore
        shifty = torch.arange(end=height, step=self.stride, device=self.base_anchors.device)  # type: ignore

        shiftx, shifty = torch.meshgrid(shiftx, shifty, indexing="ij")

        # shape (num_anchors, 4)
        shifts = torch.vstack((shiftx.ravel(), shifty.ravel(), shiftx.ravel(), shifty.ravel())).t()

        # shifts.shape == (H * W, 4)
        # base_anchors.shape == (A, 4)
        # => anchors.shape == (H * W * A, 4)
        assert isinstance(self.base_anchors, torch.Tensor)
        # Shift the anchors from base anchors location to each pixel location with stride
        anchors = self.base_anchors.view(1, -1, 4) + shifts.view(-1, 1, 4)
        anchors = anchors.reshape(-1, 4)

        return anchors
