from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
import json

try:
    import pytorch_lightning as pl
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional
    pl = None  # type: ignore
    torch = None  # type: ignore
    TensorDataset = object  # type: ignore
    DataLoader = object  # type: ignore

# Local import guard so the script works without an editable install
try:
    from ml_models.pedal_model import PedalModel
except ModuleNotFoundError:
    import os, sys
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from ml_models.pedal_model import PedalModel
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise


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


def _make_windows(arr: torch.Tensor, win: int, hop: int) -> torch.Tensor:
    # arr: (T, C) -> (N, win, C)
    T = arr.shape[0]
    if T < win:
        return torch.empty(0, win, arr.shape[1], dtype=arr.dtype)
    starts = list(range(0, T - win + 1, hop))
    out = torch.stack([arr[s : s + win] for s in starts], dim=0)
    return out


def load_csv(path: Path, *, window: int = 64, hop: int = 16) -> TensorDataset:
    df = pd.read_csv(path)
    chroma_cols = [c for c in df.columns if c.startswith("chroma_")]
    needed = set(["pedal_state", "frame_id", "track_id"]) | set(chroma_cols) | {"rel_release"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")
    groups_cols = [c for c in ["file", "track_id"] if c in df.columns]
    if not groups_cols:
        groups_cols = ["track_id"]
    xs, ys = [], []
    for _, g in df.groupby(groups_cols):
        g = g.sort_values(["frame_id"]).reset_index(drop=True)
        x_np = g[chroma_cols + ["rel_release"]].values.astype("float32")
        y_np = g["pedal_state"].values.astype("float32")
        x_t = torch.from_numpy(x_np)
        y_t = torch.from_numpy(y_np)
        x_win = _make_windows(x_t, window, hop)
        if x_win.numel() == 0:
            continue
        # Build label windows aligned to inputs: (N, win)
        T = y_t.shape[0]
        starts = list(range(0, T - window + 1, hop))
        y_win = torch.stack([y_t[s : s + window] for s in starts], dim=0)
        xs.append(x_win)
        ys.append(y_win)
    if not xs:
        raise SystemExit("no windows produced; check window/hop and input CSV")
    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    return TensorDataset(x_all, y_all)


def _feature_stats(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    chroma_cols = [c for c in df.columns if c.startswith("chroma_")]
    cols = chroma_cols + ["rel_release"]
    stats = {
        "features": cols,
        "mean": [float(df[c].mean()) for c in cols],
        "std": [float(df[c].std(ddof=0) or 1.0) for c in cols],
    }
    # avoid zeros
    stats["std"] = [s if s > 1e-8 else 1.0 for s in stats["std"]]
    return stats


def run(cfg: DictConfig) -> int:
    if pl is None or torch is None:
        raise RuntimeError("PyTorch Lightning required")
    window = int(getattr(cfg.data, "window", 64))
    hop = int(getattr(cfg.data, "hop", 16))
    train_ds = load_csv(Path(cfg.data.train), window=window, hop=hop)
    val_ds = load_csv(Path(cfg.data.val), window=window, hop=hop)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    module = LightningModule(cfg)
    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(module, train_loader, val_loader)
    out_path = Path(getattr(cfg, "out", "checkpoints/pedal.ckpt"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(out_path))
    # save feature stats beside checkpoint for inference
    try:
        stats = _feature_stats(Path(cfg.data.train))
        stats_path = out_path.with_suffix(out_path.suffix + ".stats.json")
        stats_path.write_text(json.dumps(stats, indent=2))
    except Exception:
        pass
    return 0


@hydra.main(
    config_path="../configs", config_name="pedal_model.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> int:
    return run(cfg)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
