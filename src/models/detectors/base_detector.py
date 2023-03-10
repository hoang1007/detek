from typing import Dict, Tuple

import torch
from torch import nn


class BaseDetector(nn.Module):
    def forward_train(
        self, images: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass during training.

        Args:
            images: Tensor of shape (N, C, H, W) containing images.
            gt_boxes: Tensor of shape (N, M, 4) containing ground truth boxes.
            gt_labels: Tensor of shape (N, M) containing ground truth labels.

        Returns:
            loss_dict: Dictionary containing loss values.
        """
        raise NotImplementedError

    def forward_test(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass during testing.

        Args:
            images: Tensor of shape (N, C, H, W) containing images.

        Returns:
            pred_boxes: Tensor of shape (N, M, 4) containing predicted boxes.
            pred_labels: Tensor of shape (N, M) containing predicted labels.
            pred_scores: Tensor of shape (N, M) containing predicted scores.
        """
        raise NotImplementedError
