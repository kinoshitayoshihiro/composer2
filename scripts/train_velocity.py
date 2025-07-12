from __future__ import annotations

import hydra
from omegaconf import DictConfig

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
    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val_MSE", mode="min", patience=cfg.patience, stopping_threshold=0.015)
    ]
    logger = False
    if cfg.wandb:
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(project="velocity")
    trainer = pl.Trainer(max_epochs=cfg.epochs, callbacks=callbacks, logger=logger)
    module = LightningModule(cfg)
    trainer.fit(module, train_loader, val_loader)
    Path("checkpoints").mkdir(exist_ok=True)
    trainer.save_checkpoint("checkpoints/last.ckpt")


if __name__ == "__main__":
    main()
