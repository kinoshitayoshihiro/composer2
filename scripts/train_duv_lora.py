#!/usr/bin/env python3
"""Train DUV (Duration + Velocity) model with LoRA fine-tuning support.

Supports YAML-based configuration for instrument-specific humanization.
Example:
    python scripts/train_duv_lora.py --config config/duv/guitar_Lora.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
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


logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer for linear transformations."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA weights
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation."""
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        return lora_out * self.scaling


def inject_lora_into_model(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> None:
    """Inject LoRA adapters into target modules of the model.

    Args:
        model: Base model to modify
        target_modules: List of module name patterns
            (e.g., ["attn.q_proj", "ff.in"])
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout rate
    """
    lora_count = 0

    for name, module in model.named_modules():
        # Check if this module matches any target pattern
        if not isinstance(module, nn.Linear):
            continue

        matched = False
        for pattern in target_modules:
            if pattern in name:
                matched = True
                break

        if not matched:
            continue

        # Inject LoRA
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]

        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model

        # Create LoRA adapter
        lora_adapter = LoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        # Wrap the original linear layer
        class LoRAWrapper(nn.Module):
            def __init__(self, linear: nn.Linear, lora: LoRALinear):
                super().__init__()
                self.linear = linear
                self.lora = lora

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x) + self.lora(x)

        wrapped = LoRAWrapper(module, lora_adapter)
        setattr(parent, child_name, wrapped)
        lora_count += 1

    logger.info(f"Injected LoRA into {lora_count} modules")


class DUVDataset(Dataset):
    """Dataset for DUV training with phrase-level sequences
    from JSONL manifests."""

    def __init__(
        self,
        manifest_path: Path,
        feature_cols: List[str],
        target_cols: List[str],
        max_len: int = 256,
        min_notes: int = 64,
        max_duration_sec: float = 600,
        min_duration_sec: float = 3,
    ):
        self.manifest_path = manifest_path
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.max_len = max_len

        # Load data from JSONL manifest
        logger.info(f"Loading data from {manifest_path}")

        # Check if it's JSONL or CSV
        if str(manifest_path).endswith(".jsonl"):
            # Load JSONL format (track-level manifest)
            tracks = []
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    tracks.append(json.loads(line))

            # Convert to DataFrame (note-level data)
            # Each track has arrays of note features
            rows = []
            for track in tracks:
                track_id = track.get("id", "unknown")
                # Check if track has note-level data
                if "pitch" not in track or not track["pitch"]:
                    continue

                n_notes = len(track["pitch"])
                for i in range(n_notes):
                    row = {"track_id": track_id}
                    # Extract note features
                    for col in feature_cols + target_cols:
                        if col in track:
                            val = track[col]
                            row[col] = val[i] if isinstance(val, list) and i < len(val) else 0
                        else:
                            row[col] = 0
                    rows.append(row)

            if not rows:
                raise ValueError(f"No note data found in {manifest_path}")

            self.df = pd.DataFrame(rows)
        else:
            # Use original CSV loader (returns tuple)
            self.df, _ = load_duv_dataframe(str(manifest_path))

        if self.df is None or len(self.df) == 0:
            raise ValueError(f"No data loaded from {manifest_path}")

        # Apply filters
        initial_len = len(self.df)

        # Filter by track length
        if "track_id" in self.df.columns or "song_id" in self.df.columns:
            group_col = "track_id" if "track_id" in self.df.columns else "song_id"
            track_lengths = self.df.groupby(group_col).size()
            valid_tracks = track_lengths[
                (track_lengths >= min_notes) & (track_lengths <= max_len * 10)
            ].index
            self.df = self.df[self.df[group_col].isin(valid_tracks)]

        # Filter by duration if available
        if "duration" in self.df.columns:
            self.df = self.df[
                (self.df["duration"] >= min_duration_sec)
                & (self.df["duration"] <= max_duration_sec)
            ]

        filtered_len = len(self.df)
        logger.info(f"Filtered: {initial_len} → {filtered_len} samples")

        # Compute statistics for normalization
        self._compute_stats()

        # Group into phrases
        self._prepare_phrases()

    def _compute_stats(self):
        """Compute mean/std for normalization."""
        self.stats = {}

        for col in self.feature_cols + self.target_cols:
            if col not in self.df.columns:
                logger.warning(f"Column '{col}' not found in data, will use zeros")
                self.stats[col] = {"mean": 0.0, "std": 1.0}
                continue

            values = self.df[col].values
            self.stats[col] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values) + 1e-8),
            }

    def _prepare_phrases(self):
        """Group data into phrases by track/bar."""
        # Check for grouping columns (track_id/song_id and bar)
        track_col = None
        if "track_id" in self.df.columns:
            track_col = "track_id"
        elif "song_id" in self.df.columns:
            track_col = "song_id"

        group_cols = []
        if track_col:
            group_cols.append(track_col)
        if "bar" in self.df.columns:
            group_cols.append("bar")

        if not group_cols:
            # Fallback: fixed-size chunks
            self.phrases = []
            for i in range(0, len(self.df), self.max_len):
                end = min(i + self.max_len, len(self.df))
                if end - i >= 8:  # Minimum phrase length
                    self.phrases.append((i, end))
        else:
            self.phrases = []
            for _, group in self.df.groupby(group_cols, sort=False):
                if len(group) >= 8:  # Minimum phrase length
                    indices = group.index.tolist()
                    # Split long phrases
                    for i in range(0, len(indices), self.max_len):
                        end = min(i + self.max_len, len(indices))
                        if end - i >= 8:
                            self.phrases.append((indices[i], indices[end - 1] + 1))

        logger.info(f"Created {len(self.phrases)} phrases")

    def __len__(self) -> int:
        return len(self.phrases)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx, end_idx = self.phrases[idx]
        phrase_df = self.df.iloc[start_idx:end_idx].copy()

        length = min(len(phrase_df), self.max_len)

        # Features
        features = {}
        for col in self.feature_cols:
            if col not in phrase_df.columns:
                values = np.zeros(length, dtype=np.float32)
            else:
                values = phrase_df[col].values[:length].astype(np.float32)
                # Normalize
                values = (values - self.stats[col]["mean"]) / self.stats[col]["std"]
            features[col] = torch.tensor(values, dtype=torch.float32)

        # Targets
        targets = {}
        for col in self.target_cols:
            if col not in phrase_df.columns:
                values = np.zeros(length, dtype=np.float32)
            else:
                values = phrase_df[col].values[:length].astype(np.float32)
            targets[col] = torch.tensor(values, dtype=torch.float32)

        # Pad to max_len
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:length] = True

        for key in features:
            padded = torch.zeros(self.max_len, dtype=torch.float32)
            padded[:length] = features[key]
            features[key] = padded

        for key in targets:
            padded = torch.zeros(self.max_len, dtype=torch.float32)
            padded[:length] = targets[key]
            targets[key] = padded

        targets["mask"] = mask

        return {"features": features, "targets": targets}


class DUVLoRAModel(L.LightningModule):
    """Lightning module for DUV training with LoRA."""

    def __init__(
        self,
        base_checkpoint: Optional[Path] = None,
        d_model: int = 256,
        ff_dim: int = 2048,
        n_layers: int = 4,
        n_heads: int = 8,
        max_len: int = 256,
        dropout: float = 0.1,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        w_vel: float = 1.0,
        w_dur: float = 1.0,
        huber_delta: float = 0.1,
        lora_config: Optional[Dict[str, Any]] = None,
        freeze_base: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create or load base model
        if base_checkpoint and base_checkpoint.exists():
            logger.info(f"Loading base model from {base_checkpoint}")
            checkpoint = torch.load(base_checkpoint, map_location="cpu")

            # Load model weights
            self.model = PhraseTransformer(
                d_model=d_model,
                max_len=max_len,
                ff_dim=ff_dim,
                num_layers=n_layers,
                nhead=n_heads,
                dropout=dropout,
                **kwargs,
            )

            if "state_dict" in checkpoint:
                # Remove 'model.' prefix if present
                state_dict = {}
                for k, v in checkpoint["state_dict"].items():
                    if k.startswith("model."):
                        state_dict[k[6:]] = v
                    else:
                        state_dict[k] = v
                self.model.load_state_dict(state_dict, strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)

            logger.info("Base model loaded successfully")
        else:
            logger.info("Creating new base model (no checkpoint provided)")
            self.model = PhraseTransformer(
                d_model=d_model,
                max_len=max_len,
                ff_dim=ff_dim,
                num_layers=n_layers,
                nhead=n_heads,
                dropout=dropout,
                **kwargs,
            )

        # Freeze base model if requested
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("Base model frozen")

        # Inject LoRA adapters
        if lora_config and lora_config.get("enable", False):
            inject_lora_into_model(
                self.model,
                target_modules=lora_config.get("target_modules", []),
                rank=lora_config.get("r", 8),
                alpha=lora_config.get("alpha", 16.0),
                dropout=lora_config.get("dropout", 0.0),
            )

            # Count trainable parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(
                f"Trainable: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
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

        # Velocity loss (MAE)
        # Model outputs "vel_reg", targets have "velocity"
        # Check if outputs is a dict before using 'in'
        if isinstance(outputs, dict) and "vel_reg" in outputs and "velocity" in targets:
            vel_pred = outputs["vel_reg"]
            vel_target = targets["velocity"]
            vel_loss = F.l1_loss(vel_pred * mask, vel_target * mask, reduction="sum")
            vel_loss = vel_loss / mask.sum().clamp(min=1)
            losses["vel_loss"] = vel_loss

        # Duration loss (Huber)
        # Model outputs "dur_reg", targets have "duration"
        if isinstance(outputs, dict) and "dur_reg" in outputs and "duration" in targets:
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
        total_loss = None
        if "vel_loss" in losses:
            total_loss = self.w_vel * losses["vel_loss"]
        if "dur_loss" in losses:
            if total_loss is None:
                total_loss = self.w_dur * losses["dur_loss"]
            else:
                total_loss = total_loss + self.w_dur * losses["dur_loss"]

        if total_loss is None:
            # If no losses, create a zero tensor that requires grad
            total_loss = torch.zeros(1, device=mask.device, requires_grad=True)

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
        self.log("val_loss", losses["total_loss"], prog_bar=True, sync_dist=True)
        if "vel_loss" in losses:
            self.log("val_vel_loss", losses["vel_loss"], sync_dist=True)
        if "dur_loss" in losses:
            self.log("val_dur_loss", losses["dur_loss"], sync_dist=True)

        return losses["total_loss"]

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # OneCycleLR scheduler
        if hasattr(self.trainer, "estimated_stepping_batches"):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            max_epochs = self.hparams.get("max_epochs", 10)
            total_steps = max_epochs * 1000  # Rough estimate

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


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train DUV model with LoRA")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML configuration file (e.g., config/duv/guitar_Lora.yaml)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="Devices to use (auto/cpu/cuda/mps)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Extract config sections
    base_checkpoint = config.get("base_checkpoint")
    if base_checkpoint:
        base_checkpoint = Path(base_checkpoint)

    lora_config = config.get("lora", {})
    trainer_config = config.get("trainer", {})
    feature_config = config.get("features", {})
    filters = config.get("filters", {})
    output_dir = Path(config.get("output_dir", "checkpoints/duv_lora"))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare datasets
    feature_cols = feature_config.get("inputs", [])
    target_cols = feature_config.get("targets", ["velocity", "dur_beats"])

    # Check if using CSV or JSONL manifest
    if "train_csv" in config and "valid_csv" in config:
        # Use CSV data directly
        logger.info("Using CSV data format")
        train_csv = Path(config["train_csv"])
        valid_csv = Path(config["valid_csv"])

        logger.info("Creating training dataset...")
        train_dataset = DUVDataset(
            manifest_path=train_csv,
            feature_cols=feature_cols,
            target_cols=target_cols,
            max_len=trainer_config.get("max_len", 256),
            min_notes=filters.get("min_notes", 64),
            max_duration_sec=filters.get("max_duration_sec", 600),
            min_duration_sec=filters.get("min_duration_sec", 3),
        )

        logger.info("Creating validation dataset...")
        valid_dataset = DUVDataset(
            manifest_path=valid_csv,
            feature_cols=feature_cols,
            target_cols=target_cols,
            max_len=trainer_config.get("max_len", 256),
            min_notes=filters.get("min_notes", 64),
            max_duration_sec=filters.get("max_duration_sec", 600),
            min_duration_sec=filters.get("min_duration_sec", 3),
        )

        full_dataset = train_dataset  # For stats saving

    elif "manifest" in config:
        # Use JSONL manifest with train/valid split
        logger.info("Using JSONL manifest format")
        manifest_path = Path(config["manifest"])

        logger.info("Creating training dataset...")
        full_dataset = DUVDataset(
            manifest_path=manifest_path,
            feature_cols=feature_cols,
            target_cols=target_cols,
            max_len=trainer_config.get("max_len", 256),
            min_notes=filters.get("min_notes", 64),
            max_duration_sec=filters.get("max_duration_sec", 600),
            min_duration_sec=filters.get("min_duration_sec", 3),
        )

        # Split train/valid
        train_split = config.get("train_split", 0.9)
        train_size = int(len(full_dataset) * train_split)
        valid_size = len(full_dataset) - train_size

        train_dataset, valid_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, valid_size],
            generator=torch.Generator().manual_seed(trainer_config.get("seed", 42)),
        )
    else:
        raise ValueError("Config must specify either 'train_csv'/'valid_csv' or 'manifest'")

    logger.info(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=trainer_config.get("batch_size", 64),
        shuffle=config.get("shuffle", True),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=trainer_config.get("batch_size", 64),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    model = DUVLoRAModel(
        base_checkpoint=base_checkpoint,
        d_model=256,
        ff_dim=2048,
        n_layers=4,
        n_heads=8,
        max_len=trainer_config.get("max_len", 256),
        dropout=0.1,
        lr=trainer_config.get("lr", 2e-4),
        weight_decay=trainer_config.get("weight_decay", 0.01),
        w_vel=1.0,
        w_dur=1.0,
        huber_delta=0.1,
        lora_config=lora_config,
        freeze_base=config.get("base_frozen", True),
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="duv_lora_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # Early stopping (optional)
    early_stop_patience = trainer_config.get("early_stop_patience", 3)
    callbacks = [checkpoint_callback]

    if early_stop_patience is not None:
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=early_stop_patience,
            mode="min",
        )
        callbacks.append(early_stop_callback)
    else:
        logger.info("Early stopping disabled")

    # Logger
    csv_logger = CSVLogger(output_dir, name="logs")

    # Trainer
    precision = trainer_config.get("precision", "bf16")
    if precision == "bf16" and not torch.cuda.is_available():
        logger.warning("bf16 not available, using fp32")
        precision = "32"

    # Device configuration
    if args.devices.lower() in ("auto", "cpu"):
        accelerator = "cpu"
        devices = 1
    elif args.devices.lower() in ("cuda", "gpu"):
        accelerator = "cuda"
        devices = 1
    elif args.devices.lower() == "mps":
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "auto"
        devices = "auto"

    trainer = L.Trainer(
        max_epochs=trainer_config.get("epochs", 10),
        gradient_clip_val=trainer_config.get("grad_clip_norm", 1.0),
        callbacks=callbacks,
        logger=csv_logger,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        log_every_n_steps=10,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(model, train_loader, valid_loader)

    # Save final model
    final_path = output_dir / "duv_lora_final.ckpt"
    logger.info(f"Saving final model to {final_path}")
    trainer.save_checkpoint(final_path)

    # Save scaler stats
    scaler_out = feature_config.get("scaler_out")
    if scaler_out:
        scaler_path = Path(scaler_out)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "w") as f:
            json.dump(full_dataset.stats, f, indent=2)
        logger.info(f"Saved scaler stats to {scaler_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
