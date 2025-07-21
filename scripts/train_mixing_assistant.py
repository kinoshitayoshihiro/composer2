from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import SaveConfigCallback
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities import rank_zero_info
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["MixAssistLearner", "FeatureDataModule", "MixAssistCLI", "run_cli"]


class MixAssistLearner(pl.LightningModule):
    def __init__(self, lr: float = 1e-3, in_features: int = 128):
        super().__init__()
        self.save_hyperparameters()
        self.project = (
            nn.Linear(in_features, 128) if in_features != 128 else nn.Identity()
        )
        self.encoder = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128)
        )

    def adjust_input(self, in_features: int) -> None:
        if (
            isinstance(self.project, nn.Linear)
            and self.project.in_features == in_features
        ):
            return
        self.project = (
            nn.Linear(in_features, 128) if in_features != 128 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        return self.encoder(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, _ = batch
        z = self(x)
        cosine = torch.nn.functional.cosine_similarity(z, x, dim=-1)
        loss = 0.5 * (1 - cosine).mean()
        self.log("ssl_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class FeatureDataModule(pl.LightningDataModule):
    def __init__(self, parquet_path: Path, batch_size: int = 32):
        super().__init__()
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.in_features: int = 0

    def setup(self, stage: str | None = None) -> None:
        df = pd.read_parquet(self.parquet_path)
        feature_cols = [c for c in df.columns if c != "path"]
        self.in_features = len(feature_cols)
        rank_zero_info("Feature dim %s", self.in_features)
        x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.dataset = TensorDataset(x, x)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


class MixAssistCLI(LightningCLI):
    def before_fit(self) -> None:
        self.datamodule.setup("fit")
        self.model.adjust_input(self.datamodule.in_features)
        super().before_fit()


def run_cli() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "mixing.yaml"
    MixAssistCLI(
        MixAssistLearner,
        FeatureDataModule,
        seed_everything_default=42,
        save_config_callback=SaveConfigCallback,
        parser_kwargs={"default_config_files": [str(config_path)]},
    )


if __name__ == "__main__":
    run_cli()
