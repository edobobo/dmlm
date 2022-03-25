from typing import Union, List

import hydra.utils
from omegaconf import DictConfig

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.distributed_sampler import DistributedBatchSampler
from src.sampler import MaxTokensBatchSampler


class DMLMPLDataModule(pl.LightningDataModule):
    def __init__(self, conf: DictConfig):
        super().__init__()
        self.conf = conf
        self.inventories = (
            hydra.utils.instantiate(self.conf.data.inventories)
            if "inventories" in self.conf.data
            else None
        )
        self.train_dataset, self.validation_dataset = None, None

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        if self.train_dataset is None:
            self.train_dataset = hydra.utils.instantiate(
                self.conf.data.train_dataset, inventories=self.inventories
            )
        else:
            self.train_dataset.init_final_dataset()

        lengths = self.train_dataset["length"]
        sampler = MaxTokensBatchSampler(
            lengths, self.conf.data.train_max_tokens, self.conf.data.max_batch_size
        )

        if self.conf.train.pl_trainer.gpus > 1:
            sampler = DistributedBatchSampler(sampler, dump_batches=False)

        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.conf.data.num_workers,
            collate_fn=self.train_dataset.collate_function,
            pin_memory=self.conf.data.get("pin_memory", True),
            prefetch_factor=4,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        if self.validation_dataset is None:
            self.validation_dataset = [
                hydra.utils.instantiate(val_data_conf, inventories=self.inventories)
                for val_data_conf in self.conf.data.validation_dataset
            ]
        else:
            for val_dataset in self.validation_dataset:
                val_dataset.init_final_dataset()

        validation_dataloaders = [
            DataLoader(
                dataset=vd,
                batch_size=self.conf.data.validation_batch_size,
                collate_fn=vd.collate_function,
                shuffle=False,
                num_workers=self.conf.data.num_workers,
            )
            for vd in self.validation_dataset
        ]
        return validation_dataloaders

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError
