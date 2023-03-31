from pathlib import Path

import pytest
import torch

from src.data import DataModule
from src.data.datasets import VOCDataset
from src.structures import BatchDataSample


@pytest.mark.parametrize("batch_size", [1, 16])
def test_mnist_datamodule(batch_size):
    data_dir = "data/"

    train_data = VOCDataset(data_dir, split="train")
    val_data = VOCDataset(data_dir, split="val")
    dm = DataModule(train_data, val_data, batch_size=batch_size)

    dm.prepare_data()
    dm.setup()
    assert dm.train_data and dm.val_data
    assert dm.train_dataloader() and dm.val_dataloader()

    batch = next(iter(dm.train_dataloader()))
    assert isinstance(batch, BatchDataSample)
