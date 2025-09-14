from __future__ import annotations

"""
Evaluate a sustain-pedal model on CSV features.

This resolves merge conflicts by unifying two approaches:
- Frame-wise evaluation (no temporal windows)
- Windowed evaluation (sequence windows, default)

It auto-detects device, optionally loads feature stats from --stats-json or
"<ckpt>.stats.json", and prints ROC-AUC / F1 / precision / recall / accuracy.
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception as e:  # pragma: no cover - friendlier error
    raise RuntimeError("numpy is required for eval_pedal. Try: `pip install numpy`") from e
try:
    import pandas as pd
except Exception as e:  # pragma: no cover - friendlier error
    raise RuntimeError("pandas is required for eval_pedal. Try: `pip install pandas`") from e
import torch
from torch.utils.data import DataLoader, TensorDataset, get_worker_info

try:
    # Optional; if unavailable we fall back to simple metrics
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
except Exception:  # pragma: no cover - optional dependency
    roc_auc_score = None  # type: ignore
    precision_recall_fscore_support = None  # type: ignore

from ml_models.pedal_model import PedalModel


def seed_worker(worker_id: int) -> None:
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
    info = get_worker_info()
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _resolve_workers_cli(v):
    if v is not None:
        return max(int(v), 0)
    env = os.getenv("COMPOSER2_NUM_WORKERS")
    if env and env.isdigit():
        return max(int(env), 0)
    import platform, sys
    if platform.system() == "Darwin" and sys.version_info >= (3, 13):
        return 0
    return 2

# ------------------------------
# Utilities
# ------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _as_float32(arr: pd.Series) -> pd.Series:
    return pd.to_numeric(arr, errors="coerce").astype("float32")


def _as_int(arr: pd.Series, dtype: str = "int64") -> pd.Series:
    return pd.to_numeric(arr, errors="coerce").fillna(0).astype(dtype)


def _get_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


# ------------------------------
# Stats loading / normalization
# ------------------------------

def _load_stats(stats_json: Optional[Path], ckpt_path: Optional[Path]):
    """Return (feat_cols, mean, std, meta) or None if not found/invalid.

    Supports two layouts:
      {"feat_cols": [...], "mean": {col: val}, "std": {col: val}}
      or {"mean": [...], "std": [...]} matching column order
    """
    path: Optional[Path] = None
    if stats_json and stats_json.is_file():
        path = stats_json
    elif ckpt_path is not None:
        cand = ckpt_path.with_suffix(ckpt_path.suffix + ".stats.json")
        if cand.is_file():
            path = cand
    if not path:
        return None

    obj = json.loads(path.read_text())
    feat_cols: Optional[List[str]] = obj.get("feat_cols")

    if isinstance(obj.get("mean"), dict) and isinstance(obj.get("std"), dict) and feat_cols:
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


def _apply_stats(df: pd.DataFrame, feat_cols: Optional[Sequence[str]], mean: np.ndarray, std: np.ndarray, *, strict: bool = False) -> tuple[np.ndarray, List[str]]:
    cols = list(feat_cols) if feat_cols else _feature_columns(df, None)
    have = [c for c in df.columns if c.startswith("chroma_") or c == "rel_release"]
    missing = [c for c in cols if c not in have]
    extra = [c for c in have if c not in cols]
    if missing:
        raise ValueError(f"CSV missing feature columns: {missing}")
    if strict and extra:
        raise ValueError(f"CSV has extra feature columns: {extra}")
    arr = df[cols].to_numpy(dtype="float32", copy=True)
    if arr.shape[1] != mean.size:
        raise ValueError(f"stats dimension mismatch: X={arr.shape[1]} mean={mean.size}")
    arr = (arr - mean) / np.maximum(std, 1e-8)
    return arr, cols


def load_stats_and_normalize(
    df: pd.DataFrame,
    stats: Optional[Tuple[Optional[List[str]], np.ndarray, np.ndarray]],
    *,
    strict: bool = False,
) -> tuple[np.ndarray, List[str]]:
    """Normalize features using precomputed stats.

    Returns `(X, cols)` where ``X`` is ``float32``. If ``stats`` is ``None``
    the raw feature columns are returned without normalization.
    """
    if stats is not None:
        return _apply_stats(df, stats[0], stats[1], stats[2], strict=strict)
    cols = _feature_columns(df, None)
    if not cols:
        raise SystemExit("No feature columns (expected chroma_* and optional rel_release)")
    arr = df[cols].values.astype("float32")
    return arr, cols


# ------------------------------
# CSV loading (frames & windows)
# ------------------------------

def _infer_groups_cols(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    if "file" in df.columns:
        cols.append("file")
    if "track_id" in df.columns:
        cols.append("track_id")
    if not cols:
        # Create a single dummy group if nothing to group by
        df["_gid"] = 0
        cols = ["_gid"]
    return cols


def _feature_columns(df: pd.DataFrame, prefer: Optional[Sequence[str]] = None) -> List[str]:
    chroma_cols = [c for c in df.columns if c.startswith("chroma_")]
    cols = list(chroma_cols)
    if "rel_release" in df.columns:
        cols.append("rel_release")
    if prefer and all(c in df.columns for c in prefer):
        return list(prefer)
    return cols


def load_csv_frames(path: Path, stats: Optional[Tuple[Optional[List[str]], np.ndarray, np.ndarray]], strict: bool = False):
    """Load CSV and return (X[N,F], y[N]).

    More forgiving about required columns; only needs a binary 'pedal_state'.
    """
    df = pd.read_csv(path, low_memory=False)

    # Coerce useful columns
    if "track_id" not in df.columns and "file" in df.columns:
        df["track_id"] = pd.factorize(df["file"])[0].astype("int32")
    for c in [c for c in df.columns if c.startswith("chroma_")]:
        df[c] = _as_float32(df[c])
    if "rel_release" in df.columns:
        df["rel_release"] = _as_float32(df["rel_release"])
    if "frame_id" in df.columns:
        df["frame_id"] = _as_int(df["frame_id"], "int64")
    if "pedal_state" not in df.columns:
        raise SystemExit("CSV missing required column: pedal_state")
    y = _as_int(df["pedal_state"], "uint8").values.astype("float32")

    if stats is not None:
        X_np, feat_cols = _apply_stats(df, stats[0], stats[1], stats[2], strict=strict)
    else:
        feat_cols = _feature_columns(df, None)
        if not feat_cols:
            raise SystemExit("No feature columns (expected chroma_* and optional rel_release)")
        X_np = df[feat_cols].values.astype("float32")

    x_t = torch.from_numpy(X_np)
    y_t = torch.from_numpy(y)
    return x_t, y_t, feat_cols


def _make_windows(arr: torch.Tensor, win: int, hop: int) -> torch.Tensor:
    T = arr.shape[0]
    if T < win:
        return torch.empty(0, win, arr.shape[1], dtype=arr.dtype)
    starts = list(range(0, T - win + 1, hop))
    return torch.stack([arr[s : s + win] for s in starts], dim=0)


def load_csv_to_windows(path: Path, window: int, hop: int,
                        stats: Optional[Tuple[Optional[List[str]], np.ndarray, np.ndarray]], strict: bool = False):
    """Load CSV and return (X[B,T,F], Y[B,T]).

    Requires time order via 'frame_id' when present; otherwise assumes existing order.
    Groups by (file, track_id) when available.
    """
    df = pd.read_csv(path, low_memory=False)

    # Basic sanitization
    if "track_id" not in df.columns and "file" in df.columns:
        df["track_id"] = pd.factorize(df["file"])[0].astype("int32")
    for c in [c for c in df.columns if c.startswith("chroma_")]:
        df[c] = _as_float32(df[c])
    if "rel_release" in df.columns:
        df["rel_release"] = _as_float32(df["rel_release"])
    if "frame_id" in df.columns:
        df["frame_id"] = _as_int(df["frame_id"], "int64")
    if "pedal_state" not in df.columns:
        raise SystemExit("CSV missing required column: pedal_state")

    if stats is not None:
        _, feat_cols = _apply_stats(df, stats[0], stats[1], stats[2], strict=strict)
    else:
        feat_cols = _feature_columns(df, None)
        if not feat_cols:
            raise SystemExit("No feature columns (expected chroma_* and optional rel_release)")

    groups_cols = _infer_groups_cols(df)

    xs, ys = [], []
    # Sort within groups by frame_id if present
    sort_cols = [c for c in ["frame_id"] if c in df.columns] or None
    for _, g in df.groupby(groups_cols):
        if sort_cols:
            g = g.sort_values(sort_cols).reset_index(drop=True)
        if stats is not None:
            x_np, _ = _apply_stats(g, feat_cols, stats[1], stats[2], strict=strict)
        else:
            x_np = g[feat_cols].values.astype("float32")
        y_np = _as_int(g["pedal_state"], "uint8").values.astype("float32")
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
    return x_all, y_all, feat_cols


# ------------------------------
# Model loading
# ------------------------------

def load_pedal_model(ckpt_path: Path, device: torch.device) -> PedalModel:
    state = torch.load(ckpt_path, map_location="cpu")
    model = PedalModel()
    if isinstance(state, dict) and "state_dict" in state:
        # Accept Lightning-style checkpoints with optional "model." prefix
        sd = state["state_dict"]
        # Strip a leading "model." if present
        new_sd = { (k.removeprefix("model.") if hasattr(k, "removeprefix") else k.split("model.",1)[-1] if k.startswith("model.") else k): v
                   for k, v in sd.items() }
        model.load_state_dict(new_sd, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    return model.to(device).eval()


# ------------------------------
# Metrics
# ------------------------------

def _compute_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    assert y_true.shape == prob.shape
    mask = np.isfinite(prob)
    y = y_true[mask].astype(int)
    p = prob[mask]

    # Accuracy at 0.5
    y_pred = (p >= 0.5).astype(int)
    acc = float((y_pred == y).mean()) if y.size else 0.0

    # Precision/Recall/F1
    if precision_recall_fscore_support is not None:
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, y_pred, average="binary", zero_division=0
        )
        prec = float(prec); rec = float(rec); f1 = float(f1)
    else:
        tp = int(((y == 1) & (y_pred == 1)).sum())
        fp = int(((y == 0) & (y_pred == 1)).sum())
        fn = int(((y == 1) & (y_pred == 0)).sum())
        prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    # ROC-AUC
    if roc_auc_score is not None and y.size and np.unique(y).size > 1:
        auc = float(roc_auc_score(y, p))
    else:
        auc = None

    return {
        "roc_auc": auc,
        "f1@0.5": f1,
        "precision": prec,
        "recall": rec,
        "accuracy": acc,
        "n": int(y.size),
    }


# ------------------------------
# Main
# ------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate pedal model on CSV")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--window", type=int, default=64, help="window size (set to 1 for frame-wise)")
    ap.add_argument("--hop", type=int, default=16, help="hop size between windows")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    ap.add_argument("--batch", type=int, default=64, help="mini-batch size for evaluation")
    ap.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (optional, or set COMPOSER2_NUM_WORKERS)")
    ap.add_argument("--seed", type=int, default=0, help="Optional global seed for deterministic runs")
    ap.add_argument("--fail-under-roc", type=float, default=None, help="exit code 2 if ROC-AUC below this")
    ap.add_argument("--stats-json", type=Path, help="feature stats JSON; defaults to <ckpt>.stats.json if present")
    ap.add_argument("--strict-stats", action="store_true", help="enforce stats/CSV column match")
    args = ap.parse_args(argv)

    device = _get_device(args.device)

    stats = _load_stats(args.stats_json, args.ckpt)
    if stats is None:
        print("[composer2] WARNING: feature stats not found; proceeding without normalization")
    else:
        meta = stats[3]
        if isinstance(meta, dict):
            if args.window == 64 and meta.get("window") is not None:
                args.window = int(meta["window"])
            if args.hop == 16 and meta.get("hop") is not None:
                args.hop = int(meta["hop"])

    # Load features
    if args.window and args.window > 1:
        X, Y, feat_cols = load_csv_to_windows(args.csv, args.window, args.hop, stats, args.strict_stats)
        mode = "windows"
    else:
        X, Y, feat_cols = load_csv_frames(args.csv, stats, args.strict_stats)
        mode = "frames"

    model = load_pedal_model(args.ckpt, device)

    _nw = _resolve_workers_cli(args.num_workers)
    _pw = _nw > 0
    _pf = 2 if _nw > 0 else None
    print(f"[composer2] num_workers={_nw} (persistent={_pw}, prefetch_factor={_pf or 'n/a'})")
    if args.seed is not None:
        seed = int(args.seed)
    else:
        seed = torch.initial_seed()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed) if getattr(torch, "Generator", None) else None

    bs = max(1, int(args.batch))
    ds = TensorDataset(X, Y)
    pin = device.type == "cuda"
    dl_kwargs = dict(batch_size=bs, shuffle=False, drop_last=False,
                    num_workers=_nw, persistent_workers=_pw,
                    worker_init_fn=seed_worker,
                    generator=gen,
                    pin_memory=pin)
    if _pf is not None and _nw > 0:
        dl_kwargs["prefetch_factor"] = _pf
    try:
        loader = DataLoader(ds, **dl_kwargs)
    except Exception as e:
        print(f"[composer2] DataLoader failed (num_workers={_nw}): {e} -> falling back to 0")
        fb_kwargs = dict(batch_size=bs, shuffle=False, drop_last=False,
                         num_workers=0, persistent_workers=False,
                         worker_init_fn=seed_worker,
                         generator=gen,
                         pin_memory=pin)
        loader = DataLoader(ds, **fb_kwargs)
    probs: List[np.ndarray] = []
    y_true_list: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            if logits.ndim == 3:
                logits = logits.squeeze(-1)
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            pb = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            probs.append(pb)
            y_true_list.append(yb.numpy().reshape(-1))

    prob = np.concatenate(probs, axis=0) if probs else np.array([], dtype=np.float32)
    y_true = np.concatenate(y_true_list, axis=0) if y_true_list else np.array([], dtype=np.float32)

    metrics = _compute_metrics(y_true, prob)
    metrics["mode"] = mode
    print(metrics)
    exit_code = 0
    if args.fail_under_roc is not None:
        auc = metrics.get("roc_auc")
        if auc is not None and auc < float(args.fail_under_roc):
            exit_code = 2
    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
