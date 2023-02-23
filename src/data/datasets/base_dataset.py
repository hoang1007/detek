from typing import List

from torch.utils.data import Dataset

from structures import DataSample


class BaseDataset(Dataset):
    def __init__(self, CLASSES: List[str]):
        self.CLASSES = CLASSES

    @property
    def num_classes(self):
        return len(self.CLASSES)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index: int) -> DataSample:
        raise NotImplementedError
