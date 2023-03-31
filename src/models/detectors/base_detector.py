from typing import Dict, List, Optional

import torch
from torch import nn

from src.models.base import BaseModel
from src.structures import DetResult


class BaseDetector(BaseModel):
    def __init__(
        self,
        img_normalize_means: List[float],
        img_normalize_stds: List[float],
        CLASSES: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.CLASSES = CLASSES
        self.register_buffer(
            "img_normalize_means", torch.tensor(img_normalize_means).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_normalize_stds", torch.tensor(img_normalize_stds).view(1, 3, 1, 1)
        )

    @property
    def num_classes(self):
        return len(self.CLASSES)

    def img_normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images.

        Args:
            images: Tensor of shape (N, C, H, W) containing images.
        Returns:
            images: Tensor of shape (N, C, H, W) containing normalized images.
        """
        return (images - self.img_normalize_means) / self.img_normalize_stds

    def img_denormalize(self, images: torch.Tensor) -> torch.Tensor:
        """Denormalize images.

        Args:
            images: Tensor of shape (N, C, H, W) containing images.
        Returns:
            images: Tensor of shape (N, C, H, W) containing denormalized images.
        """
        return images * self.img_normalize_stds + self.img_normalize_means

    def forward_train(
        self,
        images: torch.Tensor,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass during training.

        Args:
            images: Tensor of shape (N, C, H, W) containing images.
            gt_boxes: List tensor of shape (M, 4) containing ground truth boxes for each image.
            gt_labels: List tensor of shape (M, ) containing ground truth labels for each image.
        Returns:
            loss_dict: Dictionary containing loss values.
        """
        raise NotImplementedError

    def forward_test(self, images: torch.Tensor) -> List[DetResult]:
        """Forward pass during testing.

        Args:
            images: Tensor of shape (N, C, H, W) containing images.
        Returns:
            pred_boxes: Tensor of shape (N, M, 4) containing predicted boxes.
            pred_labels: Tensor of shape (N, M) containing predicted labels.
            pred_scores: Tensor of shape (N, M) containing predicted scores.
        """
        raise NotImplementedError
