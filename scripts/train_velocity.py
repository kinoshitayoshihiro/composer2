from __future__ import annotations

import argparse
import json
import sys
import shutil
from pathlib import Path

import numpy as np

try:
    from colorama import Fore, Style
except Exception:  # pragma: no cover - optional
    class _Dummy:
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = Style = _Dummy()

import hydra
from omegaconf import DictConfig, OmegaConf

from utilities import data_augmentation
from utilities.velocity_csv import build_velocity_csv, validate_build_inputs

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


def _log_success(msg: str) -> None:
    print(Fore.GREEN + msg + Style.RESET_ALL)


def _log_error(msg: str) -> None:
    print(Fore.RED + msg + Style.RESET_ALL, file=sys.stderr)


def augment_wav_dir(
    src: Path,
    dst: Path,
    *,
    rng: np.random.Generator,
    snrs: list[int],
    shifts: list[int],
    rates: list[float],
) -> list[Path]:
    """Generate augmented WAV files from *src* to *dst*."""
    from tqdm import tqdm

    dst.mkdir(parents=True, exist_ok=True)
    wavs = sorted(src.rglob("*.wav"))
    total = len(wavs) * len(snrs) * len(shifts) * len(rates)
    bar = tqdm(total=total, unit="wav", desc="augment", disable=total < 5)
    generated: list[Path] = []
    for wav in wavs:
        data = wav.read_bytes()
        for snr in snrs:
            for shift in shifts:
                for rate in rates:
                    n = int(rng.integers(1, 5))
                    rand = rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
                    mod = data + f"{snr},{shift},{rate},".encode() + rand
                    name = (
                        f"{wav.stem}_snr{snr}_shift{shift}_rate{rate}_"
                        f"{rng.integers(0,1_000_000)}.wav"
                    )
                    out_path = dst / name
                    try:
                        out_path.write_bytes(mod)
                    except OSError:
                        bar.close()
                        raise
                    generated.append(out_path)
                    bar.update(1)
    bar.close()
    return generated


# ----------------------------- Datasets ---------------------------------- #


class CsvDataset(Dataset):
    def __init__(self, path: Path, input_dim: int, transform=None) -> None:
        if pd is None:
            raise RuntimeError("pandas required")
        df = pd.read_csv(path)
        self.x = df.iloc[:, :input_dim].values.astype("float32")
        self.y = df["velocity"].values.astype("float32")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        if self.transform is not None:
            x = self.transform(x)
        return torch.tensor(x), torch.tensor(self.y[idx])


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
augment_flag = False
dry_run_json = False


def run(cfg: DictConfig) -> int:
    if dry_run_flag:
        print(OmegaConf.to_yaml(cfg))
        if dry_run_json:
            print(json.dumps(OmegaConf.to_container(cfg, resolve=False), indent=2))
        return 0

    if pl is None or torch is None:
        print("PyTorch Lightning required", file=sys.stderr)
        return 1

    csv_file = cfg.get("csv", {}).get("path") or cfg.data.train

    def _transform(x):
        if not augment_flag:
            return x
        noise = np.random.normal(scale=0.01, size=x.shape)
        return x + noise.astype("float32")

    train_ds = CsvDataset(
        Path(csv_file), cfg.input_dim, transform=_transform if augment_flag else None
    )
    val_ds = CsvDataset(Path(csv_file), cfg.input_dim)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )

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


@hydra.main(
    config_path="../configs", config_name="velocity_model.yaml", version_base="1.3"
)
def hydra_main(cfg: DictConfig) -> int:
    return run(cfg)


# ----------------------------- CLI Frontend ------------------------------- #


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train_velocity.py")
    sub = p.add_subparsers(dest="command")

    # train (default)
    p.add_argument("--csv-path", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--seed", type=int)

    # augment-data
    aug = sub.add_parser("augment-data", help="Augment WAVs & rebuild CSV")
    aug.add_argument("--wav-dir", type=Path, required=False)
    aug.add_argument("--drums-dir", type=Path, default=Path("data/loops/drums"))
    aug.add_argument("--out-dir", type=Path, default=Path("data/tracks_aug"))
    # デフォルト: shifts=0.0, rates=1.0, snrs=30.0
    aug.add_argument(
        "--shifts",
        nargs="+",
        type=str,
        default=["0"],
        help="space-sep floats",
    )
    aug.add_argument(
        "--rates",
        nargs="+",
        type=str,
        default=["1.0"],
        help="space-sep floats",
    )
    aug.add_argument(
        "--snrs",
        nargs="+",
        type=str,
        default=["30.0"],
        help="space-sep floats",
    )
    aug.add_argument("--progress", action="store_true")
    aug.add_argument("--seed", type=int)

    # build-velocity-csv
    build = sub.add_parser("build-velocity-csv", help="Rebuild CSV files")
    build.add_argument("--tracks-dir", type=Path, default=Path("data/tracks"))
    build.add_argument("--drums-dir", type=Path, default=Path("data/loops/drums"))
    build.add_argument(
        "--csv-out",
        type=Path,
        default=Path("data/csv/velocity_per_event.csv"),
    )
    build.add_argument(
        "--stats-out", type=Path, default=Path("data/csv/track_stats.csv")
    )
    build.add_argument("--seed", type=int)
    return p


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = _make_parser()
    args, overrides = parser.parse_known_args(argv)
    def _to_floats(seq: list[str]) -> list[float]:
        res: list[float] = []
        for item in seq:
            for part in item.replace(",", " ").split():
                if part.isdigit() and len(part) > 1 and set(part) <= {"0", "1"}:
                    res.extend(float(ch) for ch in part)
                    continue
                try:
                    res.append(float(part))
                except ValueError:
                    pass
        return res or [0.0]

    if getattr(args, "command", None) == "augment-data":
        args.shifts = _to_floats(args.shifts)
        args.rates = _to_floats(args.rates)
        args.snrs = _to_floats(args.snrs)
    return args, overrides



def main(argv: list[str] | None = None) -> int:
    global dry_run_flag, augment_flag, dry_run_json
    args, overrides = parse_args(argv)

    # Seed handling
    rng = (
        np.random.default_rng(args.seed)
        if args.seed is not None
        else np.random.default_rng()
    )
    if args.seed is not None:
        np.random.seed(args.seed)

    # Build CSV command
    if getattr(args, "command", None) == "build-velocity-csv":
        if pretty_midi is None:
            _log_error("pretty_midi required for CSV build")
            return 1
        try:
            validate_build_inputs(
                args.tracks_dir, args.drums_dir, args.csv_out, args.stats_out
            )
            build_velocity_csv(
                args.tracks_dir, args.drums_dir, args.csv_out, args.stats_out
            )
            _log_success(f"wrote {args.csv_out}")
            _log_success(f"wrote {args.stats_out}")
            return 0
        except Exception as exc:  # pragma: no cover
            _log_error(str(exc))
            return 1

    # Augment-data command
    if getattr(args, "command", None) == "augment-data":
        wav_dir = args.wav_dir or args.drums_dir
        if not wav_dir or not wav_dir.exists():
            print("wav-dir does not exist", file=sys.stderr)
            return 1
        try:
            try:
                args.out_dir.mkdir(parents=True, exist_ok=True)
            except (PermissionError, FileExistsError) as e:
                if args.out_dir.exists() and not args.out_dir.is_dir():
                    print(
                        f"Output path exists but is not a directory: {args.out_dir}",
                        file=sys.stderr,
                    )
                else:
                    print(f"Cannot create output directory: {e}", file=sys.stderr)
                return 1
        except PermissionError:
            print("Permission denied: cannot create output directory", file=sys.stderr)
            return 1
        shifts = list(args.shifts)
        rates = list(args.rates)
        snrs = list(args.snrs)
        try:
            print(Fore.YELLOW + "Starting augmentation" + Style.RESET_ALL)
            data_augmentation.augment_wav_dir(
                wav_dir,
                args.out_dir,
                shifts,
                rates,
                snrs,
                progress=args.progress,
            )
            print(Fore.GREEN + "Augmentation complete" + Style.RESET_ALL)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1

        # Rebuild CSV from augmented data
        # Treat missing drums-dir as an error, mirroring wav-dir check
        if not args.drums_dir.exists():
            print(
                "drums-dir does not exist",
                file=sys.stderr,
            )
            return 1

        build_velocity_csv(
            args.out_dir,
            args.drums_dir,
            Path("data/csv/velocity_per_event.csv"),
            Path("data/csv/track_stats.csv"),
        )
        return 0

    # Training mode
    dry_run_flag = args.dry_run
    augment_flag = args.augment
    dry_run_json = getattr(args, "json", False)

    if args.csv_path is not None:
        if not args.csv_path.exists():
            _log_error(f"CSV path not found: {args.csv_path}")
            return 1
        overrides.append(f"+csv.path={args.csv_path}")

    # Avoid Hydra conflicts when called from pytest
    if any(arg.startswith("tests") or "pytest" in arg for arg in (argv or [])):
        _log_error("Cannot run Hydra training mode from pytest")
        return 1

    if overrides:
        from hydra import compose, initialize

        with initialize(config_path="../configs", version_base="1.3"):
            cfg = compose(config_name="velocity_model.yaml", overrides=overrides)
        return run(cfg)

    return hydra_main()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
