#!/usr/bin/env python3
"""
Improved DUV training script with better learning rate scheduling
and performance optimizations based on external team feedback.
"""

from __future__ import annotations

import argparse
from argparse import BooleanOptionalAction
import json
import os
import math
import random
import warnings
from typing import Any, Dict, Iterator, List, Optional, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)
from torch.utils.data import DataLoader, Dataset, Sampler
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

EmaCallbackClass: Optional[Any] = None
try:
    from lightning.pytorch.callbacks import EMA as EmaCallbackClass
except ImportError:
    try:
        from lightning.pytorch.callbacks.ema import EMA as EmaCallbackClass
    except ImportError:
        EmaCallbackClass = None

from models.phrase_transformer import PhraseTransformer


class DUVDataset(Dataset):
    """Dataset for DUV (Duration, Velocity) prediction training."""

    def __init__(
        self,
        phrase_df: pd.DataFrame,
        stats: Dict[str, Any],
        max_len: int = 256,
        use_program_emb: bool = False,
    ):
        self.phrase_df = phrase_df
        self.stats = stats
        self.max_len = max_len
        self.use_program_emb = use_program_emb
        self.lengths: List[int] = self._compute_lengths()

    def _compute_lengths(self) -> List[int]:
        lengths: List[int] = []
        for phrase_json in self.phrase_df["phrase_json"].tolist():
            try:
                phrase_data = json.loads(phrase_json)
                length = len(phrase_data)
            except (json.JSONDecodeError, TypeError):
                length = 0

            length = max(1, min(length, self.max_len))
            lengths.append(length)

        return lengths

    def __len__(self) -> int:
        return len(self.phrase_df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.phrase_df.iloc[idx]

        # Load phrase notes
        phrase_data = json.loads(row["phrase_json"])
        phrase_df = pd.DataFrame(phrase_data)

        if len(phrase_df) == 0:
            # Empty phrase - create dummy data
            length = 1
            phrase_df = pd.DataFrame(
                {
                    "pitch": [60],
                    "position": [0],
                    "pitch_class": [0],
                    "duration": [0.5],
                    "velocity": [64],
                }
            )
        else:
            length = self.lengths[idx]
            phrase_df = phrase_df.iloc[:length]

        # Feature extraction
        feats = {}
        feats["pitch"] = torch.tensor(phrase_df["pitch"].values[:length] % 12, dtype=torch.long)

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


class LengthBucketBatchSampler(Sampler[List[int]]):
    """Bucketed batch sampler to group sequences by similar lengths."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        bucket_boundaries: Optional[Sequence[int]] = None,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        if bucket_boundaries is None or len(bucket_boundaries) == 0:
            # Default boundaries: quartiles between 0 and max length
            max_len = max(self.lengths) if self.lengths else batch_size
            step = max(1, max_len // 4)
            bucket_boundaries = [step, step * 2, step * 3]

        ordered_bounds = sorted(int(b) for b in bucket_boundaries if b > 0)
        self.bucket_limits: List[float] = [*ordered_bounds, float("inf")]

        self._epoch = 0

    def __len__(self) -> int:
        counts = self._bucket_counts()
        total_batches = 0
        for count in counts.values():
            if self.drop_last:
                total_batches += count // self.batch_size
            else:
                total_batches += math.ceil(count / self.batch_size)
        return total_batches

    def _bucket_index(self, length: int) -> int:
        for idx, limit in enumerate(self.bucket_limits):
            if length <= limit:
                return idx
        return len(self.bucket_limits) - 1

    def _bucket_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for length in self.lengths:
            bucket = self._bucket_index(length)
            counts[bucket] = counts.get(bucket, 0) + 1
        return counts

    def _bucket_indices(self) -> Dict[int, List[int]]:
        buckets: Dict[int, List[int]] = {i: [] for i in range(len(self.bucket_limits))}
        for idx, length in enumerate(self.lengths):
            bucket_idx = self._bucket_index(length)
            buckets[bucket_idx].append(idx)
        return buckets

    def __iter__(self) -> Iterator[List[int]]:
        self._epoch += 1
        rng = random.Random(self.seed + self._epoch)

        buckets = self._bucket_indices()
        batches: List[List[int]] = []

        for bucket_indices in buckets.values():
            if not bucket_indices:
                continue

            if self.shuffle:
                rng.shuffle(bucket_indices)

            for start in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        for batch in batches:
            yield batch


class ImprovedDUVModel(L.LightningModule):
    """Lightning module for DUV training with flexible scheduling.

    Combines velocity and duration objectives with configurable schedulers.
    """

    def __init__(
        self,
        d_model: int = 256,
        ff_dim: int = 2048,
        n_layers: int = 4,
        n_heads: int = 8,
        max_len: int = 256,
        dropout: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        w_vel: float = 1.0,
        w_dur: float = 1.0,
        huber_delta: float = 0.1,
        use_program_emb: bool = False,
        scheduler_type: str = "reduce_on_plateau",
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        min_lr: float = 1e-6,
        warmup_epochs: int = 0,
        warmup_start_factor: float = 0.1,
        **kwargs: Any,
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

        # Scheduler configuration
        self.scheduler_type = scheduler_type
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.warmup_epochs = max(0, warmup_epochs)
        self.warmup_start_factor = max(1e-4, min(1.0, warmup_start_factor))

    def forward(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self.model(features)

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        mask = targets["mask"]

        losses = {}

        # Velocity loss (L1/MAE)
        if "vel_reg" in outputs:
            vel_pred = outputs["vel_reg"]
            vel_target = targets["velocity"]
            vel_loss = F.l1_loss(
                vel_pred * mask,
                vel_target * mask,
                reduction="sum",
            )
            vel_loss = vel_loss / mask.sum().clamp(min=1)
            losses["vel_loss"] = vel_loss

        # Duration loss (Huber)
        if "dur_reg" in outputs:
            dur_pred = outputs["dur_reg"]
            dur_target = targets["duration"]
            dur_loss = F.huber_loss(
                dur_pred * mask,
                dur_target * mask,
                delta=self.huber_delta,
                reduction="sum",
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

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        outputs = self.forward(batch["features"])
        losses = self._compute_loss(outputs, batch["targets"])

        # Log losses
        self.log("train_loss", losses["total_loss"], prog_bar=True)
        if "vel_loss" in losses:
            self.log("train_vel_loss", losses["vel_loss"], prog_bar=False)
        if "dur_loss" in losses:
            self.log("train_dur_loss", losses["dur_loss"], prog_bar=False)

        # Log learning rate
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            self.log(
                "lr",
                (
                    scheduler.get_last_lr()[0]
                    if hasattr(scheduler, "get_last_lr")
                    else self.hparams.lr
                ),
            )

        return losses["total_loss"]

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        outputs = self.forward(batch["features"])
        losses = self._compute_loss(outputs, batch["targets"])

        # Log losses
        self.log("val_loss", losses["total_loss"], prog_bar=True)
        if "vel_loss" in losses:
            self.log("val_vel_loss", losses["vel_loss"], prog_bar=True)
        if "dur_loss" in losses:
            self.log("val_dur_loss", losses["dur_loss"], prog_bar=True)

        # Compute MAE metrics
        mask = batch["targets"]["mask"]
        if "vel_reg" in outputs:
            vel_mae = F.l1_loss(
                outputs["vel_reg"] * mask,
                batch["targets"]["velocity"] * mask,
                reduction="sum",
            ) / mask.sum().clamp(min=1)
            self.log("val_vel_mae", vel_mae, prog_bar=True)

        if "dur_reg" in outputs:
            dur_mae = F.l1_loss(
                outputs["dur_reg"] * mask,
                batch["targets"]["duration"] * mask,
                reduction="sum",
            ) / mask.sum().clamp(min=1)
            self.log("val_dur_mae", dur_mae, prog_bar=True)

        return losses["total_loss"]

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.weight_decay,
            eps=1e-8,
        )

        if self.scheduler_type == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=self.min_lr,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "reduce_on_plateau": True,
                    "strict": True,
                },
            }

        elif self.scheduler_type == "cosine":
            max_epochs = getattr(self.trainer, "max_epochs", 0) or 1
            warmup_epochs = min(self.warmup_epochs, max_epochs - 1)

            schedulers = []
            milestones: List[int] = []

            if warmup_epochs > 0:
                schedulers.append(
                    LinearLR(
                        optimizer,
                        start_factor=self.warmup_start_factor,
                        total_iters=warmup_epochs,
                    )
                )
                milestones.append(warmup_epochs)

            cosine_t_max = max(1, max_epochs - warmup_epochs)
            schedulers.append(
                CosineAnnealingLR(
                    optimizer,
                    T_max=cosine_t_max,
                    eta_min=self.min_lr,
                )
            )

            if len(schedulers) == 1:
                scheduler = schedulers[0]
            else:
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=schedulers,
                    milestones=milestones,
                )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        else:  # No scheduler
            return optimizer


class DUVDataModule(L.LightningDataModule):
    """Data module for DUV training with optimizations."""

    def __init__(
        self,
        csv_train: str,
        csv_valid: str,
        stats_json: str,
        batch_size: int = 64,
        max_len: int = 256,
        use_program_emb: bool = False,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: Optional[int] = 2,
        enable_length_bucketing: bool = False,
        bucket_boundaries: Optional[Sequence[int]] = None,
        bucket_shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.csv_train = csv_train
        self.csv_valid = csv_valid
        self.stats_json = stats_json
        self.batch_size = batch_size
        self.max_len = max_len
        self.use_program_emb = use_program_emb
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor
        self.enable_length_bucketing = enable_length_bucketing
        self.bucket_boundaries = bucket_boundaries
        self.bucket_shuffle = bucket_shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.train_dataset: Optional[DUVDataset] = None
        self.val_dataset: Optional[DUVDataset] = None

    def setup(self, stage: Optional[str] = None):
        # Load stats
        with open(self.stats_json, "r") as f:
            self.stats = json.load(f)

        # Load datasets
        if stage == "fit" or stage is None:
            train_df = pd.read_csv(self.csv_train)
            valid_df = pd.read_csv(self.csv_valid)

            self.train_dataset = DUVDataset(
                train_df,
                self.stats,
                self.max_len,
                self.use_program_emb,
            )
            self.val_dataset = DUVDataset(
                valid_df,
                self.stats,
                self.max_len,
                self.use_program_emb,
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized. Call setup('fit') first.")

        loader_kwargs: Dict[str, Any] = {
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
        }
        if self.prefetch_factor is not None and self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        if self.enable_length_bucketing:
            batch_sampler = LengthBucketBatchSampler(
                lengths=self.train_dataset.lengths,
                batch_size=self.batch_size,
                bucket_boundaries=self.bucket_boundaries,
                drop_last=self.drop_last,
                shuffle=self.bucket_shuffle,
                seed=self.seed,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                **loader_kwargs,
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
            **loader_kwargs,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized. Call setup('fit') first.")

        loader_kwargs: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
        }
        if self.prefetch_factor is not None and self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(
            self.val_dataset,
            **loader_kwargs,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Train improved DUV model with performance optimizations.",
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--csv_train",
        required=True,
        help="Training CSV file.",
    )
    data_group.add_argument(
        "--csv_valid",
        required=True,
        help="Validation CSV file.",
    )
    data_group.add_argument(
        "--stats_json",
        required=True,
        help="Statistics JSON file.",
    )
    data_group.add_argument(
        "--use_program_emb",
        action=BooleanOptionalAction,
        default=False,
        help="Use program embeddings when available.",
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--d_model",
        type=int,
        default=256,
        help="Model dimension.",
    )
    model_group.add_argument(
        "--ff_dim",
        type=int,
        default=2048,
        help="Feed-forward size.",
    )
    model_group.add_argument(
        "--n_layers",
        type=int,
        default=4,
        help="Number of layers.",
    )
    model_group.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="Attention heads.",
    )
    model_group.add_argument(
        "--max_len",
        type=int,
        default=256,
        help="Max sequence length.",
    )
    model_group.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate.",
    )

    train_group = parser.add_argument_group("training")
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size.",
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )
    train_group.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay.",
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs.",
    )
    train_group.add_argument(
        "--w_vel",
        type=float,
        default=1.0,
        help="Velocity loss weight.",
    )
    train_group.add_argument(
        "--w_dur",
        type=float,
        default=1.0,
        help="Duration loss weight.",
    )
    train_group.add_argument(
        "--huber_delta",
        type=float,
        default=0.1,
        help="Delta parameter for Huber loss.",
    )
    train_group.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )

    scheduler_group = parser.add_argument_group("scheduler")
    scheduler_group.add_argument(
        "--scheduler",
        choices=["reduce_on_plateau", "cosine", "none"],
        default="reduce_on_plateau",
        help="Learning-rate scheduler type.",
    )
    scheduler_group.add_argument(
        "--scheduler_patience",
        type=int,
        default=5,
        help="Patience for ReduceLROnPlateau.",
    )
    scheduler_group.add_argument(
        "--scheduler_factor",
        type=float,
        default=0.5,
        help="Factor for ReduceLROnPlateau.",
    )
    scheduler_group.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate.",
    )
    scheduler_group.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Warmup epochs before cosine decay.",
    )
    scheduler_group.add_argument(
        "--warmup_start_factor",
        type=float,
        default=0.1,
        help="Warmup start factor relative to base LR.",
    )

    loader_group = parser.add_argument_group("dataloader")
    loader_group.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    loader_group.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Prefetch factor per worker (set <=0 to disable).",
    )
    loader_group.add_argument(
        "--pin_memory",
        action=BooleanOptionalAction,
        default=True,
        help="Toggle pinned memory for DataLoaders.",
    )
    loader_group.add_argument(
        "--persistent_workers",
        action=BooleanOptionalAction,
        default=True,
        help="Keep DataLoader workers alive between epochs.",
    )
    loader_group.add_argument(
        "--length_bucketing",
        action=BooleanOptionalAction,
        default=False,
        help="Enable length-aware bucketing for training batches.",
    )
    loader_group.add_argument(
        "--bucket_boundaries",
        type=int,
        nargs="+",
        default=None,
        help="Bucket boundaries for length bucketing.",
    )
    loader_group.add_argument(
        "--bucket_shuffle",
        action=BooleanOptionalAction,
        default=True,
        help="Shuffle batches within buckets.",
    )
    loader_group.add_argument(
        "--drop_last",
        action=BooleanOptionalAction,
        default=True,
        help="Drop incomplete batches during training.",
    )

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument(
        "--precision",
        default="16-mixed",
        help="Trainer precision setting.",
    )
    runtime_group.add_argument(
        "--accelerator",
        default="auto",
        help="Accelerator type.",
    )
    runtime_group.add_argument(
        "--devices",
        default="auto",
        help="Devices to use.",
    )
    runtime_group.add_argument(
        "--strategy",
        default="auto",
        help="Distributed strategy.",
    )
    runtime_group.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="Gradient clipping value.",
    )
    runtime_group.add_argument(
        "--val_check_interval",
        type=float,
        default=0.5,
        help="Validation frequency within an epoch.",
    )
    runtime_group.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="Logging frequency in steps.",
    )
    runtime_group.add_argument(
        "--progress_bar",
        action=BooleanOptionalAction,
        default=True,
        help="Enable the Trainer progress bar.",
    )
    runtime_group.add_argument(
        "--deterministic",
        action=BooleanOptionalAction,
        default=False,
        help="Request deterministic trainer behaviour.",
    )
    runtime_group.add_argument(
        "--num_sanity_val_steps",
        type=int,
        default=2,
        help="Sanity validation batches before training.",
    )
    runtime_group.add_argument(
        "--default_root_dir",
        default=None,
        help="Default root directory for outputs.",
    )
    runtime_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    ema_group = parser.add_argument_group("ema")
    ema_group.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help="EMA decay (set to 0 to disable).",
    )
    ema_group.add_argument(
        "--ema_device",
        default="cpu",
        help="Device for EMA weights (cpu, cuda, auto).",
    )

    checkpoint_group = parser.add_argument_group("checkpoint")
    checkpoint_group.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        help="Directory to store checkpoints.",
    )
    checkpoint_group.add_argument(
        "--model_name",
        default="duv_piano_plus_improved",
        help="Model name prefix for checkpoints.",
    )

    args = parser.parse_args()

    if args.seed is not None:
        L.seed_everything(args.seed, workers=True)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.precision in {"16-mixed", "bf16-mixed"}:
        torch.set_float32_matmul_precision("medium")

    bucket_boundaries = args.bucket_boundaries
    if bucket_boundaries:
        bucket_boundaries = sorted({max(1, min(args.max_len, b)) for b in bucket_boundaries})
        bucket_boundaries = [b for b in bucket_boundaries if b < args.max_len]
        if not bucket_boundaries:
            bucket_boundaries = None

    prefetch_factor: Optional[int] = args.prefetch_factor
    if prefetch_factor is not None and prefetch_factor <= 0:
        prefetch_factor = None

    data_module = DUVDataModule(
        csv_train=args.csv_train,
        csv_valid=args.csv_valid,
        stats_json=args.stats_json,
        batch_size=args.batch_size,
        max_len=args.max_len,
        use_program_emb=args.use_program_emb,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=prefetch_factor,
        enable_length_bucketing=args.length_bucketing,
        bucket_boundaries=bucket_boundaries,
        bucket_shuffle=args.bucket_shuffle,
        drop_last=args.drop_last,
        seed=args.seed,
    )

    model = ImprovedDUVModel(
        d_model=args.d_model,
        ff_dim=args.ff_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_len=args.max_len,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        w_vel=args.w_vel,
        w_dur=args.w_dur,
        huber_delta=args.huber_delta,
        use_program_emb=args.use_program_emb,
        scheduler_type=args.scheduler,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f"{args.model_name}_best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if args.ema_decay > 0:
        if EmaCallbackClass is None:
            warnings.warn(
                "EMA callback requested but unavailable; skipping EMA.",
                RuntimeWarning,
            )
        else:
            callbacks.append(  # type: ignore[arg-type]
                EmaCallbackClass(
                    decay=args.ema_decay,
                    ema_device=args.ema_device,
                    pin_memory=args.pin_memory,
                )
            )

    trainer_kwargs: Dict[str, Any] = {
        "max_epochs": args.epochs,
        "callbacks": callbacks,
        "precision": args.precision,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "strategy": args.strategy,
        "log_every_n_steps": args.log_every_n_steps,
        "val_check_interval": args.val_check_interval,
        "gradient_clip_val": args.gradient_clip_val,
        "deterministic": args.deterministic,
        "enable_progress_bar": args.progress_bar,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "num_sanity_val_steps": args.num_sanity_val_steps,
    }
    if args.default_root_dir is not None:
        trainer_kwargs["default_root_dir"] = args.default_root_dir

    trainer = L.Trainer(**trainer_kwargs)

    print(f"Training {args.model_name} for {args.epochs} epochs...")
    print(f"Scheduler: {args.scheduler}")
    print(f"Precision: {args.precision}")
    if args.length_bucketing:
        print(
            "Length bucketing enabled with boundaries:",
            bucket_boundaries if bucket_boundaries else "auto",
        )

    trainer.fit(model=model, datamodule=data_module)

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"{args.model_name}_best.ckpt",
    )
    print(f"Training completed! Best model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
