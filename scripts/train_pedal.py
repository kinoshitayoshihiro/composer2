from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

try:
    import pytorch_lightning as pl
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional
    pl = None  # type: ignore
    torch = None  # type: ignore
    TensorDataset = object  # type: ignore
    DataLoader = object  # type: ignore

from ml_models.pedal_model import PedalModel


class LightningModule(pl.LightningModule if pl is not None else object):
    def __init__(self, cfg: DictConfig) -> None:
        if pl is None or torch is None:
            raise RuntimeError("PyTorch Lightning required")
        super().__init__()
        self.model = PedalModel(class_weight=cfg.get("class_weight"))
        self.lr = cfg.learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.model.loss(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.model.loss(pred, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def load_csv(path: Path) -> TensorDataset:
    df = pd.read_csv(path)
    chroma_cols = [c for c in df.columns if c.startswith("chroma_")]
    x = df[chroma_cols + ["rel_release"]].values.astype("float32")
    y = df["pedal_state"].values.astype("float32")
    x = torch.tensor(x)
    y = torch.tensor(y)
    return TensorDataset(x, y)


def run(cfg: DictConfig) -> int:
    if pl is None or torch is None:
        raise RuntimeError("PyTorch Lightning required")
    train_ds = load_csv(Path(cfg.data.train))
    val_ds = load_csv(Path(cfg.data.val))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    module = LightningModule(cfg)
    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(module, train_loader, val_loader)
    Path("checkpoints").mkdir(exist_ok=True)
    trainer.save_checkpoint("checkpoints/last.ckpt")
    return 0


@hydra.main(
    config_path="../configs", config_name="pedal_model.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> int:
    return run(cfg)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
