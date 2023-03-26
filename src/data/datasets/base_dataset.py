from typing import List

from torch.utils.data import Dataset

from src.structures import DataSample


class BaseDataset(Dataset):
    def __init__(self, CLASSES: List[str]):
        self.CLASSES = CLASSES
        self._class2idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}

    @property
    def num_classes(self):
        return len(self.CLASSES)

    def get_class_idx(self, class_name: str) -> int:
        return self._class2idx[class_name]

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index: int) -> DataSample:
        raise NotImplementedError
