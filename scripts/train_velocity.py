from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import DictConfig

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--max-epochs", type=int)
parser.add_argument("--out", type=str)
parser.add_argument("--run-dir", type=str)

if "-h" in sys.argv or "--help" in sys.argv:
    parser.print_help()
    print()

parsed_args, remaining_argv = parser.parse_known_args()
sys.argv = sys.argv[:1] + remaining_argv
extra_overrides: list[str] = []
if parsed_args.max_epochs is not None:
    extra_overrides.append(f"trainer.max_epochs={parsed_args.max_epochs}")
if parsed_args.out is not None:
    extra_overrides.append(f"+out={parsed_args.out}")
if parsed_args.run_dir is not None:
    extra_overrides.append(f"hydra.run.dir={parsed_args.run_dir}")
if extra_overrides:
    sys.argv += extra_overrides

import hydra

try:
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, Dataset
    import torch
    import pandas as pd
except Exception:  # pragma: no cover - optional
    pl = None  # type: ignore
    DataLoader = object  # type: ignore
    Dataset = object  # type: ignore
    torch = None  # type: ignore
    pd = None  # type: ignore

from utilities.ml_velocity import MLVelocityModel, velocity_loss


class CsvDataset(Dataset):
    def __init__(self, path: str, input_dim: int) -> None:
        if pd is None:
            raise RuntimeError("pandas required")
        df = pd.read_csv(path)
        self.x = df.iloc[:, :input_dim].values.astype("float32")
        self.y = df["velocity"].values.astype("float32")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.x[idx]),
            torch.tensor(self.y[idx]),
        )


class LightningModule(pl.LightningModule if pl is not None else object):
    def __init__(self, cfg: DictConfig) -> None:
        if pl is None or torch is None:
            raise RuntimeError("PyTorch Lightning required")
        super().__init__()
        self.model = MLVelocityModel(cfg.input_dim)
        self.lr = cfg.learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x.unsqueeze(0))
        loss = velocity_loss(pred.squeeze(0), y)
        mse = torch.mean((pred.squeeze(0) - y) ** 2)
        self.log("train_loss", loss)
        self.log("train_MSE", mse)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x.unsqueeze(0))
        loss = velocity_loss(pred.squeeze(0), y)
        mse = torch.mean((pred.squeeze(0) - y) ** 2)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_MSE", mse, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@hydra.main(config_path="../configs", config_name="velocity_model.yaml")
def main(cfg: DictConfig) -> None:
    if pl is None or torch is None:
        raise RuntimeError("PyTorch Lightning required")
    train_ds = CsvDataset(cfg.data.train, cfg.input_dim)
    val_ds = CsvDataset(cfg.data.val, cfg.input_dim)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    callbacks = []
    if "callbacks" in cfg and "early_stopping" in cfg.callbacks:
        es_cfg = cfg.callbacks.early_stopping
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor=es_cfg.monitor,
                mode=es_cfg.mode,
                patience=es_cfg.patience,
                stopping_threshold=es_cfg.stopping_threshold,
            )
        )
    logger = False
    if (
        "trainer" in cfg
        and "logger" in cfg.trainer
        and cfg.trainer.logger.use_wandb
    ):
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(project="velocity")

    trainer_kwargs = {k: v for k, v in cfg.trainer.items() if k != "logger"}
    trainer = pl.Trainer(**trainer_kwargs, callbacks=callbacks, logger=logger)
    module = LightningModule(cfg)
    trainer.fit(module, train_loader, val_loader)
    Path("checkpoints").mkdir(exist_ok=True)
    trainer.save_checkpoint("checkpoints/last.ckpt")


if __name__ == "__main__":
    main()
