from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import json

from ml_models.pedal_model import PedalModel


def _make_windows(arr: torch.Tensor, win: int, hop: int) -> torch.Tensor:
    T = arr.shape[0]
    if T < win:
        return torch.empty(0, win, arr.shape[1], dtype=arr.dtype)
    starts = list(range(0, T - win + 1, hop))
    return torch.stack([arr[s : s + win] for s in starts], dim=0)


def load_csv_to_windows(path: Path, window: int, hop: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        T = y_t.shape[0]
        starts = list(range(0, T - window + 1, hop))
        y_win = torch.stack([y_t[s : s + window] for s in starts], dim=0)
        xs.append(x_win)
        ys.append(y_win)
    if not xs:
        raise SystemExit("no windows produced; check window/hop and input CSV")
    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    return x_all, y_all


def load_pedal_model(ckpt_path: Path, device: torch.device) -> PedalModel:
    state = torch.load(ckpt_path, map_location="cpu")
    model = PedalModel()
    if isinstance(state, dict) and "state_dict" in state:
        sd = {k.removeprefix("model."): v for k, v in state["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def _load_stats(stats_json: Path | None, ckpt_path: Path | None):
    path = None
    if stats_json and stats_json.is_file():
        path = stats_json
    elif ckpt_path is not None:
        cand = ckpt_path.with_suffix(ckpt_path.suffix + ".stats.json")
        if cand.is_file():
            path = cand
    if not path:
        return None
    obj = json.loads(path.read_text())
    mean = np.array(obj.get("mean", []), dtype=np.float32)
    std = np.array(obj.get("std", []), dtype=np.float32)
    if mean.size == 0 or std.size == 0:
        return None
    std[std < 1e-8] = 1.0
    return mean, std


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate pedal model on CSV (ROC-AUC / F1)")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--hop", type=int, default=16)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    ap.add_argument("--batch", type=int, default=64, help="mini-batch size for evaluation")
    ap.add_argument("--stats-json", type=Path, help="feature stats JSON (mean/std); defaults to <ckpt>.stats.json if present")
    args = ap.parse_args(argv)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    X, Y = load_csv_to_windows(args.csv, args.window, args.hop)
    st = _load_stats(args.stats_json, args.ckpt)
    if st is not None:
        mean, std = st
        if X.shape[-1] == mean.size:
            X = (X - torch.tensor(mean).view(1, 1, -1)) / torch.tensor(std).view(1, 1, -1)
    model = load_pedal_model(args.ckpt, device)
    probs: list[np.ndarray] = []
    bs = max(1, int(args.batch))
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            xb = X[i : i + bs].to(device)
            logits = model(xb)  # (B, T)
            pb = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            probs.append(pb)
    prob = np.concatenate(probs, axis=0)
    y_true = Y.numpy().reshape(-1)
    # Remove NaNs/constant issues if any
    mask = np.isfinite(prob)
    y_true = y_true[mask]
    prob = prob[mask]
    auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else float("nan")
    # Choose 0.5 threshold for quick F1
    y_pred = (prob >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    print({"roc_auc": (None if np.isnan(auc) else float(auc)), "f1@0.5": float(f1), "precision": float(p), "recall": float(r), "n": int(y_true.size)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
