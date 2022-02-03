from typing import Union, List

import hydra.utils
from omegaconf import DictConfig

from torch.utils.data import DataLoader
import pytorch_lightning as pl


class DMLMPLDataModule(pl.LightningDataModule):
    def __init__(self, conf: DictConfig):
        super().__init__()
        self.conf = conf
        self.inventories = hydra.utils.instantiate(self.conf.data.inventories)
        self.train_dataset = hydra.utils.instantiate(
            self.conf.data.train_dataset, inventories=self.inventories
        )
        self.validation_dataset = hydra.utils.instantiate(
            self.conf.data.validation_dataset, inventories=self.inventories
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.conf.data.train_batch_size,
            collate_fn=lambda x: self.train_dataset.collate_function(x),
            shuffle=True,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.conf.data.validation_batch_size,
            collate_fn=lambda x: self.validation_dataset.collate_function(x),
            shuffle=True,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError
