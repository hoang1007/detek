from typing import List

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from structures import BatchDataSample, DataSample

from .datasets import BaseDataset


class DataModule(LightningDataModule):
    def __init__(
        self,
        train_data: BaseDataset,
        val_data: BaseDataset,
        batch_size: int,
        num_workers: int,
        train_val_ratio: float = 0.6,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        assert 0 < train_val_ratio < 1
        self.train_val_ratio = train_val_ratio

    def setup(self, stage=None):
        pass

    @staticmethod
    def collate_fn(samples: List[DataSample]) -> BatchDataSample:
        return BatchDataSample(samples)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
