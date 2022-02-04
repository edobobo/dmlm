from typing import Any, Optional

import hydra
from transformers import BertForMaskedLM, AutoConfig

import pytorch_lightning as pl
import torch

from src.dmlm_dataset import DMLMDataset


class BERTDMLM(pl.LightningModule):
    def __init__(
        self,
        transformer_model: str,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        optim_conf: dict,
        additional_special_tokens: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        bert_config = AutoConfig.from_pretrained(
            transformer_model,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
        )
        self.bert_model = BertForMaskedLM(bert_config)

        if additional_special_tokens is not None and additional_special_tokens > 0:
            self.bert_model.resize_token_embeddings(
                bert_config.vocab_size + additional_special_tokens
            )

        self.optim_conf = optim_conf

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> dict:
        model_output = self.bert_model(input_ids, attention_mask, labels=labels)
        output_dict = {"loss": model_output.loss}
        return output_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(**batch)
        self.log("loss", forward_output["loss"])
        return forward_output["loss"]

    def validation_step(
        self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> None:
        forward_output = self.forward(**batch)
        self.log("val_loss", forward_output["loss"])

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        raise NotImplementedError

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.optim_conf, params=self.parameters())

    def set_train_dataset(self, train_dataset: DMLMDataset):
        self.train_dataset = train_dataset

    def on_validation_epoch_end(self) -> None:
        self.train_dataset.init_final_dataset()
