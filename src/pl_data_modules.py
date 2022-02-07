from typing import Union, List

import hydra.utils
from omegaconf import DictConfig

from torch.utils.data import DataLoader
import pytorch_lightning as pl


class DMLMPLDataModule(pl.LightningDataModule):
    def __init__(self, conf: DictConfig):
        super().__init__()
        self.conf = conf
        self.inventories = (
            hydra.utils.instantiate(self.conf.data.inventories)
            if "inventories" in self.conf.data
            else None
        )
        self.train_dataset = hydra.utils.instantiate(
            self.conf.data.train_dataset, inventories=self.inventories
        )
        self.validation_dataset = [
            hydra.utils.instantiate(val_data_conf, inventories=self.inventories)
            for val_data_conf in self.conf.data.validation_dataset
        ]

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.conf.data.train_batch_size,
            collate_fn=lambda x: self.train_dataset.collate_function(x),
            shuffle=True,
            num_workers=self.conf.data.num_workers,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        if type(self.validation_dataset) == list:
            return [
                DataLoader(
                    dataset=val_dataset,
                    batch_size=self.conf.data.validation_batch_size,
                    collate_fn=lambda x: val_dataset.collate_function(x),
                    shuffle=True,
                    num_workers=self.conf.data.num_workers,
                )
                for val_dataset in self.validation_dataset
            ]
        else:
            raise NotImplementedError

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError
