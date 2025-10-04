#!/usr/bin/env python3
"""Train DUV (Duration + Velocity) model using PhraseTransformer."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

try:
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import CSVLogger
except ImportError:
    try:
        import pytorch_lightning as L
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        from pytorch_lightning.loggers import CSVLogger
    except ImportError:
        print("Error: pytorch-lightning not found!")
        sys.exit(1)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.phrase_transformer import PhraseTransformer
from utilities.duv_infer import load_duv_dataframe


class DUVDataset(Dataset):
    """Dataset for DUV training with phrase-level sequences."""

    def __init__(
        self,
        df: pd.DataFrame,
        stats: Dict[str, Any],
        max_len: int = 256,
        use_program_emb: bool = False,
    ):
        self.df = df
        self.stats = stats
        self.max_len = max_len
        self.use_program_emb = use_program_emb

        # Normalize features
        self.feat_cols = stats.get("feat_cols", [])
        self.mean = np.array([stats["mean"].get(c, 0.0) for c in self.feat_cols], dtype=np.float32)
        self.std = np.array([stats["std"].get(c, 1.0) for c in self.feat_cols], dtype=np.float32)

        # Group by bar/track for phrase-level sequences
        self._prepare_phrases()

    def _prepare_phrases(self):
        """Group data into phrases by bar/track."""
        group_cols = [c for c in ("track_id", "bar") if c in self.df.columns]
        if not group_cols:
            # Fallback: group by index ranges
            self.phrases = []
            for i in range(0, len(self.df), self.max_len):
                end = min(i + self.max_len, len(self.df))
                self.phrases.append((i, end))
        else:
            self.phrases = []
            for _, group in self.df.groupby(group_cols, sort=False):
                if len(group) > 0:
                    indices = group.index.tolist()
                    self.phrases.append((indices[0], indices[-1] + 1))

        print(f"Created {len(self.phrases)} phrases")

    def __len__(self) -> int:
        return len(self.phrases)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx, end_idx = self.phrases[idx]
        phrase_df = self.df.iloc[start_idx:end_idx].copy()

        length = min(len(phrase_df), self.max_len)

        # Required features
        feats = {}
        if "position" in phrase_df.columns:
            feats["position"] = torch.tensor(
                phrase_df["position"].values[:length], dtype=torch.long
            )
        else:
            feats["position"] = torch.arange(length, dtype=torch.long)

        if "pitch_class" in phrase_df.columns:
            feats["pitch_class"] = torch.tensor(
                phrase_df["pitch_class"].values[:length], dtype=torch.long
            )
        else:
            feats["pitch_class"] = torch.zeros(length, dtype=torch.long)

        feats["duration"] = torch.tensor(phrase_df["duration"].values[:length], dtype=torch.float32)
        feats["velocity"] = torch.tensor(phrase_df["velocity"].values[:length], dtype=torch.float32)

        # Optional features (zero-fill if missing)
        if "vel_bucket" in phrase_df.columns:
            feats["vel_bucket"] = torch.tensor(
                phrase_df["vel_bucket"].values[:length], dtype=torch.long
            )
        else:
            feats["vel_bucket"] = torch.zeros(length, dtype=torch.long)

        if "dur_bucket" in phrase_df.columns:
            feats["dur_bucket"] = torch.tensor(
                phrase_df["dur_bucket"].values[:length], dtype=torch.long
            )
        else:
            feats["dur_bucket"] = torch.zeros(length, dtype=torch.long)

        if self.use_program_emb and "program" in phrase_df.columns:
            feats["program"] = torch.tensor(phrase_df["program"].values[:length], dtype=torch.long)

        # Pad to max_len
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:length] = True

        for key in feats:
            padded = torch.zeros(self.max_len, dtype=feats[key].dtype)
            padded[:length] = feats[key]
            feats[key] = padded

        # Target values (for loss calculation)
        targets = {
            "velocity": feats["velocity"].clone(),
            "duration": feats["duration"].clone(),
            "mask": mask,
        }

        return {"features": feats, "targets": targets}


class DUVModel(L.LightningModule):
    """Lightning module for DUV training."""

    def __init__(
        self,
        d_model: int = 256,
        ff_dim: int = 2048,
        n_layers: int = 4,
        n_heads: int = 8,
        max_len: int = 256,
        dropout: float = 0.1,
        lr: float = 3e-4,
        w_vel: float = 1.0,
        w_dur: float = 1.0,
        huber_delta: float = 0.1,
        use_program_emb: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = PhraseTransformer(
            d_model=d_model,
            max_len=max_len,
            ff_dim=ff_dim,
            num_layers=n_layers,
            nhead=n_heads,
            dropout=dropout,
            **kwargs,
        )

        # Loss weights
        self.w_vel = w_vel
        self.w_dur = w_dur
        self.huber_delta = huber_delta

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(features)

    def _compute_loss(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        mask = targets["mask"]

        losses = {}

        # Velocity loss (L1/MAE)
        if "vel_reg" in outputs:
            vel_pred = outputs["vel_reg"]
            vel_target = targets["velocity"]
            vel_loss = F.l1_loss(vel_pred * mask, vel_target * mask, reduction="sum")
            vel_loss = vel_loss / mask.sum().clamp(min=1)
            losses["vel_loss"] = vel_loss

        # Duration loss (Huber)
        if "dur_reg" in outputs:
            dur_pred = outputs["dur_reg"]
            dur_target = targets["duration"]
            dur_loss = F.huber_loss(
                dur_pred * mask, dur_target * mask, delta=self.huber_delta, reduction="sum"
            )
            dur_loss = dur_loss / mask.sum().clamp(min=1)
            losses["dur_loss"] = dur_loss

        # Total loss
        total_loss = 0.0
        if "vel_loss" in losses:
            total_loss += self.w_vel * losses["vel_loss"]
        if "dur_loss" in losses:
            total_loss += self.w_dur * losses["dur_loss"]

        losses["total_loss"] = total_loss
        return losses

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(batch["features"])
        losses = self._compute_loss(outputs, batch["targets"])

        # Log losses
        self.log("train_loss", losses["total_loss"], prog_bar=True)
        if "vel_loss" in losses:
            self.log("train_vel_loss", losses["vel_loss"])
        if "dur_loss" in losses:
            self.log("train_dur_loss", losses["dur_loss"])

        return losses["total_loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(batch["features"])
        losses = self._compute_loss(outputs, batch["targets"])

        # Log losses
        self.log("val_loss", losses["total_loss"], prog_bar=True)
        if "vel_loss" in losses:
            self.log("val_vel_loss", losses["vel_loss"])
        if "dur_loss" in losses:
            self.log("val_dur_loss", losses["dur_loss"])

        # Compute MAE metrics
        mask = batch["targets"]["mask"]
        if "vel_reg" in outputs:
            vel_mae = F.l1_loss(
                outputs["vel_reg"] * mask, batch["targets"]["velocity"] * mask, reduction="sum"
            ) / mask.sum().clamp(min=1)
            self.log("val_vel_mae", vel_mae)

        if "dur_reg" in outputs:
            dur_mae = F.l1_loss(
                outputs["dur_reg"] * mask, batch["targets"]["duration"] * mask, reduction="sum"
            ) / mask.sum().clamp(min=1)
            self.log("val_dur_mae", dur_mae)

        return losses["total_loss"]

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)

        # OneCycleLR scheduler
        if hasattr(self.trainer, "estimated_stepping_batches"):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            total_steps = self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader())

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


class DUVDataModule(L.LightningDataModule):
    """Data module for DUV training."""

    def __init__(
        self,
        csv_train: str,
        csv_valid: str,
        stats_json: str,
        batch_size: int = 64,
        max_len: int = 256,
        use_program_emb: bool = False,
        num_workers: int = 4,
    ):
        super().__init__()
        self.csv_train = csv_train
        self.csv_valid = csv_valid
        self.stats_json = stats_json
        self.batch_size = batch_size
        self.max_len = max_len
        self.use_program_emb = use_program_emb
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        # Load stats
        with open(self.stats_json, "r") as f:
            self.stats = json.load(f)

        # Load datasets
        if stage == "fit" or stage is None:
            train_df = pd.read_csv(self.csv_train)
            valid_df = pd.read_csv(self.csv_valid)

            self.train_dataset = DUVDataset(
                train_df, self.stats, self.max_len, self.use_program_emb
            )
            self.val_dataset = DUVDataset(valid_df, self.stats, self.max_len, self.use_program_emb)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


def main():
    parser = argparse.ArgumentParser(description="Train DUV model")

    # Data arguments
    parser.add_argument("--csv-train", type=str, required=True, help="Training CSV file")
    parser.add_argument("--csv-valid", type=str, required=True, help="Validation CSV file")
    parser.add_argument("--stats-json", type=str, required=True, help="Stats JSON file")

    # Model arguments
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--ff-dim", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--max-len", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--use-program-emb", type=int, default=0, help="Use program embedding (0/1)"
    )

    # Training arguments
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--devices", type=str, default="auto", help="Devices (cpu/cuda/mps/auto)")

    # Loss arguments
    parser.add_argument("--w-vel", type=float, default=1.0, help="Velocity loss weight")
    parser.add_argument("--w-dur", type=float, default=1.0, help="Duration loss weight")
    parser.add_argument("--huber-delta", type=float, default=0.1, help="Huber loss delta")

    # Output arguments
    parser.add_argument(
        "--out", type=str, default="checkpoints/duv_model.ckpt", help="Output checkpoint path"
    )
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Data module
    datamodule = DUVDataModule(
        csv_train=args.csv_train,
        csv_valid=args.csv_valid,
        stats_json=args.stats_json,
        batch_size=args.batch,
        max_len=args.max_len,
        use_program_emb=bool(args.use_program_emb),
        num_workers=min(4, os.cpu_count() or 1),
    )

    # Model
    model = DUVModel(
        d_model=args.d_model,
        ff_dim=args.ff_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_len=args.max_len,
        dropout=args.dropout,
        lr=args.lr,
        w_vel=args.w_vel,
        w_dur=args.w_dur,
        huber_delta=args.huber_delta,
        use_program_emb=bool(args.use_program_emb),
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.out).parent,
        filename=Path(args.out).stem + ".best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )

    # Logger
    logger = CSVLogger(args.log_dir, name="duv_training")

    # Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accum,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator="auto",
        devices="auto" if args.devices == "auto" else args.devices,
    )

    # Train
    print("Starting training...")
    trainer.fit(model, datamodule)

    # Save final model
    print(f"Saving final model to {args.out}")
    trainer.save_checkpoint(args.out)

    print("Training complete!")


if __name__ == "__main__":
    main()
