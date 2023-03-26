from typing import List
import torch
from torch import nn


class AnchorGenerator(nn.Module):
    def __init__(self, stride: int, scales: List[float] = [8, 16, 32], ratios: List[float] = [0.5, 1, 2]):
        """
        Generate anchors on the image.

        Args:
            stride (int): Stride of the feature map.
            scales (List[float]): Scales of the anchors with base size (stride).
            ratios (List[float]): Ratios between height and width of anchors.
        """
        super().__init__()
        self.stride = stride
        self.scales = scales
        self.ratios = ratios

        self.register_buffer('base_anchors', self.generate_base_anchors(self.stride))

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0) # type: ignore
    
    def generate_base_anchors(self, base_size: int):
        scales = torch.tensor(self.scales, dtype=torch.float32)
        ratios = torch.tensor(self.ratios, dtype=torch.float32)

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios

        ws = base_size * (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = base_size * (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-0.5 * ws, -0.5 * hs, 0.5 * ws, 0.5 * hs], dim=-1)
        return base_anchors
    
    def forward(self, img_height: int, img_width: int):
        """Generate anchors for the image size.

        Args:
            img_height (int): Height of the image.
            img_width (int): Width of the image.

        Returns:
            anchors: Anchors in `(x_tl, y_tl, x_br, y_br)` format. Shape (N, 4)
        """
        base_anchors = self.base_anchors
        assert isinstance(base_anchors, torch.Tensor)

        shifts_x = torch.arange(0, img_width, step=self.stride, device=base_anchors.device, dtype=torch.float32)
        shifts_y = torch.arange(0, img_height, self.stride, device=base_anchors.device, dtype=torch.float32)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        anchors = base_anchors[None, :, :] + shifts[:, None, :]
        anchors = anchors.view(-1, 4)
        return anchors
