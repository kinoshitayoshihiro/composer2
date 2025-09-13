from __future__ import annotations

"""Train sustain-pedal model (windowed sequence input) with Hydra.

This resolves merge conflicts by unifying CSV loaders and adding robust handling:
- Windowed dataset: X(B, T, F), y(B, T) with `--data.window`/`--data.hop`.
- Feature columns: `chroma_*` + optional `rel_release`.
- Missing columns are sanitized (e.g., synth `frame_id`, `track_id`).
- Saves `<ckpt>.stats.json` alongside checkpoint for inference scripts.

Hydra config example (configs/pedal_model.yaml):
  data:
    train: data/pedal/train.csv
    val:   data/pedal/val.csv
    window: 64
    hop: 16
  batch_size: 64
  learning_rate: 1e-3
  trainer:
    max_epochs: 20
    accelerator: auto
    devices: 1
"""

import argparse  # not used by Hydra, kept for parity
from pathlib import Path
import json
import os, sys, platform, random

try:
    import hydra
except Exception as e:  # pragma: no cover - guidance
    raise RuntimeError("Hydra is required to run train_pedal. Please `pip install hydra-core`.") from e
try:
    import pandas as pd
except Exception as e:  # pragma: no cover - guidance
    raise RuntimeError("pandas is required for train_pedal. Please `pip install pandas`.") from e
import numpy as np
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

# Local import guard so the script works without an editable install
try:
    from ml_models.pedal_model import PedalModel
except ModuleNotFoundError:  # pragma: no cover
    import os, sys
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ml_models.pedal_model import PedalModel


class LightningModule(pl.LightningModule if pl is not None else object):
    def __init__(self, cfg: DictConfig) -> None:
        if pl is None or torch is None:
            raise RuntimeError("PyTorch Lightning required")
        super().__init__()
        self.model = PedalModel(class_weight=cfg.get("class_weight"))
        self.lr = float(cfg.learning_rate)

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


# ------------------------------
# Data loading (CSV → windows)
# ------------------------------

def _to_float32(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float32")


def _to_int(series: pd.Series, dtype: str = "int64") -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(dtype)


def _make_windows(arr: torch.Tensor, win: int, hop: int) -> torch.Tensor:
    # arr: (T, C) -> (N, win, C)
    T = arr.shape[0]
    if T < win:
        return torch.empty(0, win, arr.shape[1], dtype=arr.dtype)
    starts = list(range(0, T - win + 1, hop))
    out = torch.stack([arr[s : s + win] for s in starts], dim=0)
    return out


def load_csv(path: Path, *, window: int = 64, hop: int = 16) -> TensorDataset:
    if torch is None:
        raise RuntimeError("PyTorch required")
    df = pd.read_csv(path, low_memory=False)

    # Track grouping columns
    if "track_id" not in df.columns:
        if "file" in df.columns:
            df["track_id"] = pd.factorize(df["file"])[0].astype("int32")
        else:
            df["track_id"] = 0  # single-group fallback

    # Sanitize types
    chroma_cols = [c for c in df.columns if c.startswith("chroma_")]
    if not chroma_cols:
        raise SystemExit("CSV must contain at least one chroma_* feature column")
    for c in chroma_cols:
        df[c] = _to_float32(df[c])
    if "rel_release" in df.columns:
        df["rel_release"] = _to_float32(df["rel_release"])
    if "frame_id" in df.columns:
        df["frame_id"] = _to_int(df["frame_id"], "int64")
    else:
        # synthesize a sequential frame index per track
        df["frame_id"] = (
            df.groupby("track_id").cumcount().astype("int64")
        )
    if "pedal_state" not in df.columns:
        raise SystemExit("CSV missing required column: pedal_state")
    df["pedal_state"] = _to_int(df["pedal_state"], "uint8")

    # Feature column order
    feat_cols = chroma_cols + (["rel_release"] if "rel_release" in df.columns else [])

    # Group and windowize
    groups_cols = [c for c in ["file", "track_id"] if c in df.columns] or ["track_id"]
    xs, ys = [], []
    for _, g in df.groupby(groups_cols):
        g = g.sort_values(["frame_id"]).reset_index(drop=True)
        x_np = g[feat_cols].values.astype("float32")
        y_np = g["pedal_state"].values.astype("float32")
        x_t = torch.from_numpy(x_np)
        y_t = torch.from_numpy(y_np)
        x_win = _make_windows(x_t, window, hop)
        if x_win.numel() == 0:
            continue
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
    df = pd.read_csv(csv_path, low_memory=False)
    chroma_cols = [c for c in df.columns if c.startswith("chroma_")]
    if not chroma_cols:
        raise SystemExit("cannot compute stats: no chroma_* columns in CSV")
    feat_cols = chroma_cols + (["rel_release"] if "rel_release" in df.columns else [])
    mean = {c: float(pd.to_numeric(df[c], errors="coerce").mean()) for c in feat_cols}
    std = {}
    for c in feat_cols:
        v = float(pd.to_numeric(df[c], errors="coerce").std(ddof=0))
        std[c] = v if v > 1e-8 else 1.0
    return {
        "feat_cols": feat_cols,
        "mean": mean,
        "std": std,
    }


_WORKER_BASE_SEED = 0

def _dataloader_worker_init_fn(worker_id: int) -> None:
    seed = int(_WORKER_BASE_SEED) ^ (worker_id + 1)
    random.seed(seed)
    try:
        np.random.seed(seed & 0xFFFFFFFF)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _resolve_num_workers(requested: int | None) -> int:
    if requested is not None:
        return max(0, int(requested))
    env = os.getenv("COMPOSER2_NUM_WORKERS")
    if env:
        try:
            return max(0, int(env))
        except ValueError:
            pass
    if platform.system() == "Darwin" and sys.version_info >= (3, 13):
        return 0
    return 2


# ------------------------------
# Train entrypoint
# ------------------------------

def run(cfg: DictConfig) -> int:
    if pl is None or torch is None:
        raise RuntimeError("PyTorch Lightning required")

    window = int(getattr(cfg.data, "window", 64))
    hop = int(getattr(cfg.data, "hop", 16))

    train_ds = load_csv(Path(cfg.data.train), window=window, hop=hop)
    val_ds = load_csv(Path(cfg.data.val), window=window, hop=hop)

    global _WORKER_BASE_SEED
    _WORKER_BASE_SEED = int(getattr(cfg, "seed", 0) or 0)
    requested_nw = getattr(cfg, "num_workers", getattr(getattr(cfg, "data", object()), "num_workers", None))
    _nw = _resolve_num_workers(requested_nw)
    pin_memory = torch.cuda.is_available()
    persistent = _nw > 0
    prefetch = 2 if _nw > 0 else None
    print(f"[composer2] num_workers={_nw} (persistent={persistent}, prefetch_factor={prefetch if prefetch is not None else 'n/a'})")

    dl_kwargs = dict(batch_size=int(cfg.batch_size), pin_memory=pin_memory, persistent_workers=persistent)
    if _nw > 0:
        dl_kwargs.update(num_workers=_nw, prefetch_factor=2, worker_init_fn=_dataloader_worker_init_fn)
    else:
        dl_kwargs.update(num_workers=0)
    try:
        train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    except (AttributeError, RuntimeError, BrokenPipeError) as e:
        if _nw == 0:
            raise
        print(f"[composer2] fallback -> num_workers=0 due to '{type(e).__name__}: {e}'")
        dl_kwargs.update(num_workers=0)
        dl_kwargs.pop("prefetch_factor", None)
        dl_kwargs.pop("worker_init_fn", None)
        train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    module = LightningModule(cfg)
    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(module, train_loader, val_loader)

    out_path = Path(getattr(cfg, "out", "checkpoints/pedal.ckpt"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(out_path))

    # Save feature stats beside checkpoint for inference
    try:
        stats = _feature_stats(Path(cfg.data.train))
        stats.update({
            "fps": getattr(cfg.data, "fps", None),
            "window": window,
            "hop": hop,
            "pad_multiple": getattr(cfg.data, "pad_multiple", 1),
        })
        stats_path = out_path.with_suffix(out_path.suffix + ".stats.json")
        stats_path.write_text(json.dumps(stats, indent=2))
    except Exception:
        pass

    return 0


@hydra.main(config_path="../configs", config_name="pedal_model.yaml", version_base="1.3")
def main(cfg: DictConfig) -> int:  # pragma: no cover
    return run(cfg)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
