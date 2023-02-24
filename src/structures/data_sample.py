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
        self.bboxes = [sample.bboxes for sample in samples]
        self.labels = [sample.labels for sample in samples]

        self.batch_size = len(samples)

    def to(self, device: torch.device):
        self.images = self.images.to(device)

        for i in range(self.batch_size):
            self.bboxes[i] = self.bboxes[i].to(device)
            self.labels[i] = self.labels[i].to(device)
        return self

    def __len__(self):
        return self.batch_size
