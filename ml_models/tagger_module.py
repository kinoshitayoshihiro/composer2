from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch

from utilities.ml_articulation import ArticulationTagger


class TaggerModule(pl.LightningModule):
    """Lightning wrapper for :class:`ArticulationTagger`."""

    def __init__(self, num_labels: int, **embed_kw: int) -> None:
        super().__init__()
        self.model = ArticulationTagger(num_labels, **embed_kw)

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        return self.model.forward_batch(batch)

    def training_step(self, batch: dict[str, torch.Tensor], _: int) -> torch.Tensor:
        loss = self.model.forward_batch(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], _: int) -> None:
        loss = self.model.forward_batch(batch)
        self.log("val_loss", loss)

    def configure_optimizers(self) -> Any:
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
