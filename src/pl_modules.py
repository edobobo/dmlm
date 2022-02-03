from typing import Any

import hydra
from transformers import BertForMaskedLM, AutoConfig

import pytorch_lightning as pl
import torch


class BERTDMLM(pl.LightningModule):
    def __init__(
        self,
        transformer_model: str,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        optim_conf: dict,
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

        self.optim_conf = optim_conf

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> dict:
        model_output = self.bert_model(input_ids, attention_mask, labels)
        output_dict = {"loss": model_output.loss}
        return output_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(**batch)
        self.log("loss", forward_output["loss"])
        return forward_output["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        forward_output = self.forward(**batch)
        self.log("val_loss", forward_output["loss"])

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        raise NotImplementedError

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.optim_conf, params=self.parameters())
