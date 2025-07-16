from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

try:
    import pandas as pd
    import pretty_midi
    import pytorch_lightning as pl
    import torch
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - optional
    pd = None  # type: ignore
    pretty_midi = None  # type: ignore
    pl = None  # type: ignore
    torch = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore


# ----------------------------- CSV Utilities ----------------------------- #

def _scan_midi_files(paths: list[Path]) -> tuple[list[list[float]], list[list[str]]]:
    """Return per-note velocity rows and simple track statistics."""
    rows: list[list[float]] = []
    stats: list[list[str]] = []
    for path in paths:
        try:
            pm = pretty_midi.PrettyMIDI(str(path))  # type: ignore[arg-type]
        except Exception:
            continue
        total = 0
        prev_vel = 64
        for inst in pm.instruments:
            for note in inst.notes:
                rows.append([
                    note.pitch,
                    note.end - note.start,
                    prev_vel,
                    note.velocity,
                ])
                prev_vel = note.velocity
                total += 1
        stats.append([path.name, str(total)])
    return rows, stats


def build_velocity_csv(
    tracks_dir: Path,
    drums_dir: Path,
    csv_out: Path,
    stats_out: Path,
) -> None:
    midi_paths = sorted(tracks_dir.rglob("*.mid")) + sorted(drums_dir.rglob("*.mid"))
    rows, stats = _scan_midi_files(midi_paths)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pitch", "duration", "prev_vel", "velocity"])
        writer.writerows(rows)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    with stats_out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "events"])
        writer.writerows(stats)
    print(f"wrote {csv_out}")
    print(f"wrote {stats_out}")


# ----------------------------- Datasets ---------------------------------- #

class CsvDataset(Dataset):
    def __init__(self, path: Path, input_dim: int) -> None:
        if pd is None:
            raise RuntimeError("pandas required")
        df = pd.read_csv(path)
        self.x = df.iloc[:, :input_dim].values.astype("float32")
        self.y = df["velocity"].values.astype("float32")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


class LightningModule(pl.LightningModule if pl is not None else object):
    def __init__(self, cfg: DictConfig) -> None:
        if pl is None or torch is None:
            raise RuntimeError("PyTorch Lightning required")
        super().__init__()
        from utilities.ml_velocity import MLVelocityModel, velocity_loss

        self.model = MLVelocityModel(cfg.input_dim)
        self.loss_fn = velocity_loss
        self.lr = cfg.learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x.unsqueeze(0))
        loss = self.loss_fn(pred.squeeze(0), y)
        mse = torch.mean((pred.squeeze(0) - y) ** 2)
        self.log("train_loss", loss)
        self.log("train_MSE", mse)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x.unsqueeze(0))
        loss = self.loss_fn(pred.squeeze(0), y)
        mse = torch.mean((pred.squeeze(0) - y) ** 2)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_MSE", mse, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ----------------------------- Hydra Entry -------------------------------- #

dry_run_flag = False


@hydra.main(config_path="../configs", config_name="velocity_model.yaml")
def hydra_main(cfg: DictConfig) -> int:
    if dry_run_flag:
        print(OmegaConf.to_yaml(cfg))
        return 0

    if pl is None or torch is None:
        print("PyTorch Lightning required", file=sys.stderr)
        return 1

    csv_file = cfg.get("csv", {}).get("path") or cfg.data.train
    train_ds = CsvDataset(Path(csv_file), cfg.input_dim)
    val_ds = CsvDataset(Path(csv_file), cfg.input_dim)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    callbacks = []
    if "callbacks" in cfg.trainer and "early_stopping" in cfg.trainer.callbacks:
        es_cfg = cfg.trainer.callbacks.early_stopping
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor=es_cfg.monitor,
                mode=es_cfg.mode,
                patience=es_cfg.patience,
                stopping_threshold=es_cfg.stopping_threshold,
            )
        )

    logger = False
    if "logger" in cfg.trainer and cfg.trainer.logger.use_wandb:
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(project="velocity")

    trainer_kwargs = {
        k: v
        for k, v in cfg.trainer.items()
        if k not in {"logger", "callbacks", "checkpoint_path"}
    }

    device = cfg.get("device")
    if device:
        trainer_kwargs.setdefault("accelerator", device)
        trainer_kwargs.setdefault("devices", 1)

    trainer = pl.Trainer(**trainer_kwargs, callbacks=callbacks, logger=logger)
    module = LightningModule(cfg)
    trainer.fit(module, train_loader, val_loader)

    ckpt = cfg.get("model", {}).get("checkpoint")
    if ckpt:
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(ckpt)
    else:
        Path("checkpoints").mkdir(exist_ok=True)
        trainer.save_checkpoint("checkpoints/last.ckpt")
    return 0


# ----------------------------- CLI Frontend ------------------------------- #

def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(prog="train_velocity.py")
    sub = parser.add_subparsers(dest="command")

    build = sub.add_parser(
        "build-velocity-csv",
        help="Rebuild velocity_per_event.csv and track_stats.csv",
    )
    build.add_argument("--tracks-dir", type=Path, default=Path("data/tracks"))
    build.add_argument("--drums-dir", type=Path, default=Path("data/loops/drums"))
    build.add_argument(
        "--csv-out", type=Path, default=Path("data/csv/velocity_per_event.csv")
    )
    build.add_argument(
        "--stats-out", type=Path, default=Path("data/csv/track_stats.csv")
    )

    parser.add_argument(
        "--csv-path",
        type=Path,
        help="Path to velocity_per_event.csv for training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and exit",
    )

    args, overrides = parser.parse_known_args(argv)
    return args, overrides


def main(argv: list[str] | None = None) -> int:
    global dry_run_flag
    args, overrides = parse_args(argv)

    if args.command == "build-velocity-csv":
        if pretty_midi is None:
            print("pretty_midi required for CSV build", file=sys.stderr)
            return 1
        build_velocity_csv(args.tracks_dir, args.drums_dir, args.csv_out, args.stats_out)
        return 0

    dry_run_flag = args.dry_run

    if args.csv_path is not None:
        overrides.append(f"+csv.path={args.csv_path}")

    sys.argv = [sys.argv[0]] + overrides
    return hydra_main()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
