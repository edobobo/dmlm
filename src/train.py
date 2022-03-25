import os

import omegaconf
import hydra

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from src.pl_data_modules import DMLMPLDataModule


def train(conf: omegaconf.DictConfig) -> None:

    # environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # reproducibility
    pl.seed_everything(conf.train.seed)

    # data module declaration
    pl_data_module = DMLMPLDataModule(conf)

    # main module declaration
    pl_module = hydra.utils.instantiate(conf.model)

    # callbacks declaration
    callbacks_store = []

    if conf.train.get("early_stopping_callback", None) is not None:
        early_stopping_callback: EarlyStopping = hydra.utils.instantiate(
            conf.train.early_stopping_callback
        )
        callbacks_store.append(early_stopping_callback)

    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint_callback: ModelCheckpoint = hydra.utils.instantiate(
            conf.train.model_checkpoint_callback
        )
        callbacks_store.append(model_checkpoint_callback)

    # logger
    logger = hydra.utils.instantiate(conf.logger)

    # trainer
    trainer: Trainer = hydra.utils.instantiate(
        conf.train.pl_trainer,
        callbacks=callbacks_store,
        logger=logger,
        plugins=DDPPlugin(find_unused_parameters=False),
    )

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    def fix(conf):
        """
        fix paths
        """
        if type(conf) == list or type(conf) == omegaconf.listconfig.ListConfig:
            for i in range(len(conf)):
                conf[i] = fix(conf[i])
            return conf
        elif type(conf) == dict or type(conf) == omegaconf.dictconfig.DictConfig:
            for k, v in conf.items():
                conf[k] = fix(v)
            return conf
        elif type(conf) == str:
            if "/" in conf and os.path.exists(
                hydra.utils.to_absolute_path(conf[: conf.rindex("/")])
            ):
                return hydra.utils.to_absolute_path(conf)
            else:
                return conf
        elif type(conf) in [float, int, bool]:
            return conf
        else:
            raise ValueError(f"Unexpected type {type(conf)}: {conf}")

    fix(conf)
    train(conf)


if __name__ == "__main__":
    main()
