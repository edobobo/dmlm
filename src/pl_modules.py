from typing import Any, Optional

import hydra
from transformers import AutoConfig, AutoModelForMaskedLM

import pytorch_lightning as pl
import torch


class TransformerDMLM(pl.LightningModule):
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

        config = AutoConfig.from_pretrained(
            transformer_model,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
        )

        self.model = AutoModelForMaskedLM.from_config(config)

        if additional_special_tokens is not None and additional_special_tokens > 0:
            self.model.resize_token_embeddings(
                config.vocab_size + additional_special_tokens
            )

        self.optim_conf = optim_conf

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> dict:
        model_output = self.model(input_ids, attention_mask, labels=labels)
        output_dict = {"loss": model_output.loss}
        return output_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(**batch)
        self.log("loss", forward_output["loss"], rank_zero_only=True)
        self.log(
            "train_batch_size", int(batch["input_ids"].shape[0]), rank_zero_only=True
        )
        return forward_output["loss"]

    def validation_step(
        self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> None:
        forward_output = self.forward(**batch)
        self.log("val_loss", forward_output["loss"], rank_zero_only=True)

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        raise NotImplementedError

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.optim_conf, params=self.parameters())
