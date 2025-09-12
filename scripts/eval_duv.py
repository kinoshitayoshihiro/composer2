from __future__ import annotations

"""Lightweight evaluation for Duration/Velocity (DUV) models.

The script intentionally mirrors ``eval_pedal.py`` but keeps the surface
API tiny.  It accepts a feature CSV and a checkpoint that bundles the
velocity and duration weights.  Feature statistics are loaded from
``--stats-json`` or ``<ckpt>.stats.json`` and are used to coerce the CSV
into the expected column order and dtype before normalisation.

Metrics
-------
``velocity``  : mean absolute error (MAE) and optional Pearson/Spearman
``duration``  : root mean squared error (RMSE)
Both metrics also report the number of evaluated samples.  Velocity MAE
is additionally broken down by ``beat_bin`` if the column exists.

The script prints a **single JSON line** with all metrics making it easy
for calling code to consume.
"""

from pathlib import Path
import argparse
import json
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

try:  # Optional; only used when available
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover - optional dependency
    pearsonr = spearmanr = None  # type: ignore

from utilities.ml_velocity import MLVelocityModel
from utilities.ml_duration import DurationTransformer


_worker_seed = 0


def _worker_init_fn(worker_id: int) -> None:
    np.random.seed(_worker_seed + worker_id)
    torch.manual_seed(_worker_seed + worker_id)


def _resolve_workers(v: Optional[int]) -> int:
    if v is not None:
        return max(int(v), 0)
    env = os.getenv("COMPOSER2_NUM_WORKERS")
    return max(int(env), 0) if (env and env.isdigit()) else 0


def _get_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # pragma: no cover - macOS
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _as_float32(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float32")


def _as_int(s: pd.Series, dtype: str = "int64") -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(dtype)


# ---------------------------------------------------------------------------
# Stats loading / normalisation
# ---------------------------------------------------------------------------

def _load_stats(stats_json: Optional[Path], ckpt_path: Path) -> Optional[Tuple[Optional[List[str]], np.ndarray, np.ndarray, dict]]:
    path = stats_json if stats_json and stats_json.is_file() else ckpt_path.with_suffix(ckpt_path.suffix + ".stats.json")
    if not path.is_file():
        return None
    obj = json.loads(path.read_text())
    feat_cols: Optional[List[str]] = obj.get("feat_cols")
    if feat_cols and isinstance(obj.get("mean"), dict) and isinstance(obj.get("std"), dict):
        mean = np.array([obj["mean"][c] for c in feat_cols], dtype=np.float32)
        std = np.array([obj["std"][c] for c in feat_cols], dtype=np.float32)
    else:
        mean = np.array(obj.get("mean", []), dtype=np.float32)
        std = np.array(obj.get("std", []), dtype=np.float32)
        if mean.size == 0 or std.size == 0:
            return None
    std[std < 1e-8] = 1.0
    meta = {k: obj.get(k) for k in ("fps", "window", "hop", "pad_multiple")}
    return feat_cols, mean, std, meta


def _apply_stats(df: pd.DataFrame, feat_cols: Optional[Sequence[str]], mean: np.ndarray, std: np.ndarray, *, strict: bool = False) -> Tuple[np.ndarray, List[str]]:
    cols = list(feat_cols) if feat_cols else [c for c in df.columns if c.startswith("feat_")]
    missing = [c for c in cols if c not in df.columns]
    extra = [c for c in df.columns if c.startswith("feat_") and c not in cols]
    if strict and (missing or extra):
        raise ValueError(f"stats/CSV mismatch: missing={missing}, extra={extra}")
    arr = df.reindex(columns=cols, fill_value=0).to_numpy(dtype="float32", copy=True)
    if mean.size == arr.shape[1]:
        arr = (arr - mean) / np.maximum(std, 1e-8)
    return arr, cols


def load_stats_and_normalize(df: pd.DataFrame, stats: Optional[Tuple[Optional[List[str]], np.ndarray, np.ndarray]], *, strict: bool = False) -> Tuple[np.ndarray, List[str]]:
    if stats is None:
        raise ValueError("feature stats required")
    return _apply_stats(df, stats[0], stats[1], stats[2], strict=strict)


# ---------------------------------------------------------------------------
# Duration utilities
# ---------------------------------------------------------------------------

def _load_duration_model(path: Path) -> DurationTransformer:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        hparams = obj.get("hyper_parameters") or obj.get("hparams", {})
        d_model = int(hparams.get("d_model", 64))
        max_len = int(hparams.get("max_len", 16))
        model = DurationTransformer(d_model=d_model, max_len=max_len)
        model.load_state_dict(obj["state_dict"], strict=False)
        return model.eval()
    model = DurationTransformer()
    model.load_state_dict(obj, strict=False)
    return model.eval()


def _duration_predict(df: pd.DataFrame, model: DurationTransformer) -> Tuple[np.ndarray, np.ndarray]:
    preds: List[float] = []
    targets: List[float] = []
    max_len = int(model.max_len.item()) if hasattr(model, "max_len") else 16
    for _, g in df.groupby("bar", sort=False):
        g = g.sort_values("position")
        L = len(g)
        pad = max_len - L
        pitch_class = ((g["pitch"].to_numpy() % 12).tolist() + [0] * pad)
        dur = g["duration"].to_list() + [0.0] * pad
        vel = g["velocity"].to_list() + [0.0] * pad
        pos = g["position"].to_list() + [0] * pad
        mask = torch.zeros(1, max_len, dtype=torch.bool)
        mask[0, :L] = 1
        feats = {
            "duration": torch.tensor(dur, dtype=torch.float32).unsqueeze(0),
            "velocity": torch.tensor(vel, dtype=torch.float32).unsqueeze(0),
            "pitch_class": torch.tensor(pitch_class, dtype=torch.long).unsqueeze(0),
            "position_in_bar": torch.tensor(pos, dtype=torch.long).unsqueeze(0),
        }
        with torch.no_grad():
            out = model(feats, mask)[0, :L].cpu().numpy()
        preds.extend(out.tolist())
        targets.extend(g["duration"].to_list())
    return np.array(preds, dtype=np.float32), np.array(targets, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    stats = _load_stats(args.stats_json, args.ckpt)
    if stats is None:
        raise SystemExit("missing or invalid stats json")

    df = pd.read_csv(args.csv, low_memory=False)
    if "track_id" not in df.columns and "file" in df.columns:
        df["track_id"] = pd.factorize(df["file"])[0].astype("int32")
    for c in stats[0] or []:
        if c in df.columns:
            df[c] = _as_float32(df[c])
    for col in ["velocity", "duration", "beat_bin"]:
        if col in df.columns:
            df[col] = _as_float32(df[col])
    if "bar" in df.columns:
        df["bar"] = _as_int(df["bar"], "int32")
    if "position" in df.columns:
        df["position"] = _as_int(df["position"], "int32")

    X, _ = load_stats_and_normalize(df, stats, strict=True)
    y_vel = df.get("velocity")
    if y_vel is None:
        raise SystemExit("CSV missing 'velocity' column")
    y_vel = y_vel.to_numpy(dtype="float32", copy=False)

    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=_resolve_workers(args.num_workers),
        worker_init_fn=_worker_init_fn,
    )

    device = _get_device(args.device)
    vel_model = MLVelocityModel.load(str(args.ckpt))
    vel_model = vel_model.to(device).eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            out = vel_model(xb.to(device))
            preds.append(out.cpu().numpy())
    vel_pred = np.concatenate(preds, axis=0).astype("float32")

    # Duration
    dur_pred = None
    dur_rmse = None
    dur_cnt = 0
    if "duration" in df.columns and "bar" in df.columns and "position" in df.columns and "pitch" in df.columns:
        dmodel = _load_duration_model(args.ckpt)
        dmodel = dmodel.to(device)
        pred_dur, tgt_dur = _duration_predict(df, dmodel)
        dur_cnt = int(tgt_dur.size)
        if dur_cnt:
            dur_rmse = float(np.sqrt(np.mean((pred_dur - tgt_dur) ** 2)))
        dur_pred = pred_dur

    mae = float(np.mean(np.abs(vel_pred - y_vel)))
    metrics: dict[str, object] = {
        "velocity_mae": mae,
        "velocity_count": int(y_vel.size),
    }

    if "beat_bin" in df.columns:
        by = {}
        for beat, g in df.groupby("beat_bin"):
            idx = g.index.to_numpy()
            by[str(int(beat))] = float(np.mean(np.abs(vel_pred[idx] - y_vel[idx])))
        metrics["velocity_mae_by_beat"] = by

    if pearsonr is not None and y_vel.size > 1:
        metrics["velocity_pearson"] = float(pearsonr(vel_pred, y_vel)[0])
        metrics["velocity_spearman"] = float(spearmanr(vel_pred, y_vel)[0])

    if dur_rmse is not None:
        metrics["duration_rmse"] = dur_rmse
        metrics["duration_count"] = dur_cnt

    print(json.dumps(metrics, ensure_ascii=False))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    p = argparse.ArgumentParser(prog="eval_duv.py")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--stats-json", type=Path)
    p.add_argument("--num-workers", dest="num_workers", type=int)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
