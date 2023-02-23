from typing import List

import torch


class DataSample:
    def __init__(self, image: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor):
        self.image = image
        self.bboxes = bboxes
        self.labels = labels


class BatchDataSample:
    def __init__(self, samples: List[DataSample]):
        self.images = torch.stack([sample.image for sample in samples], dim=0)
        self.bboxes = torch.stack([sample.bboxes for sample in samples], dim=0)
        self.labels = torch.stack([sample.labels for sample in samples], dim=0)

        self.batch_size = len(samples)

    def to(self, device: torch.device):
        self.images = self.images.to(device)
        self.bboxes = self.bboxes.to(device)
        self.labels = self.labels.to(device)
