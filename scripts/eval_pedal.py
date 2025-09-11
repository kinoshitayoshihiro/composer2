from __future__ import annotations

import argparse
from pathlib import Path
import json, os

try:
    import pandas as pd
except Exception as e:  # pragma: no cover - guidance
    raise RuntimeError("pandas is required for eval_pedal. Please `pip install pandas`.") from e
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml_models.pedal_model import PedalModel


def _make_xy(df: pd.DataFrame, stats_json: str | None) -> tuple[torch.Tensor, torch.Tensor]:
    # feature columns
    chroma_cols = [c for c in df.columns if c.startswith("chroma_")]
    x_cols = chroma_cols + (["rel_release"] if "rel_release" in df.columns else [])
    x = df[x_cols].values.astype("float32")
    # optional stats normalization
    if stats_json and os.path.exists(stats_json):
        s = json.load(open(stats_json, "r"))
        cols = s.get("feat_cols", x_cols)
        mean = np.array([s["mean"][c] for c in cols], dtype="float32")
        std = np.array([s["std"][c] for c in cols], dtype="float32")
        std[std == 0] = 1.0
        x = df[cols].values.astype("float32")
        x = (x - mean) / std
    y = df["pedal_state"].values.astype("float32")
    return torch.tensor(x), torch.tensor(y)


def load_csv(path: Path, stats_json: str | None) -> TensorDataset:
    df = pd.read_csv(path, low_memory=False)
    if "track_id" not in df.columns:
        df["track_id"] = pd.factorize(df["file"])[0].astype("int32")
    chroma_cols = [c for c in df.columns if c.startswith("chroma_")]
    for c in chroma_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    if "rel_release" in df.columns:
        df["rel_release"] = pd.to_numeric(df["rel_release"], errors="coerce").astype(
            "float32"
        )
    if "frame_id" in df.columns:
        df["frame_id"] = (
            pd.to_numeric(df["frame_id"], errors="coerce").fillna(0).astype("int64")
        )
    df["pedal_state"] = pd.to_numeric(df["pedal_state"], errors="coerce").fillna(0).astype(
        "uint8"
    )
    required = ["file", "frame_id", "pedal_state"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    x, y = _make_xy(df, stats_json)
    return TensorDataset(x, y)


def evaluate(ds: TensorDataset, ckpt: Path, device: str, batch_size: int, stats_json: str | None) -> dict:
    dev = torch.device(device)
    model = PedalModel()
    state = torch.load(ckpt, map_location=dev)
    sd = state.get("state_dict", state)
    model.load_state_dict(sd, strict=False)
    model.to(dev)
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size)
    correct = total = 0
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev)
            y = y.to(dev)
            out = model(x).sigmoid()
            pred = (out >= 0.5).float()
            correct += (pred == y).sum().item()
            total += y.numel()
            y_true.append(y.cpu().numpy().ravel())
            y_prob.append(out.cpu().numpy().ravel())
    import numpy as _np
    y_true = _np.concatenate(y_true) if y_true else _np.array([])
    y_prob = _np.concatenate(y_prob) if y_prob else _np.array([])
    acc = correct / total if total else 0.0
    roc_auc = None
    try:  # optional metrics
        from sklearn.metrics import roc_auc_score, f1_score

        if y_true.size and _np.unique(y_true).size > 1:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        f1 = float(f1_score(y_true, (y_prob >= 0.5).astype(int))) if y_true.size else 0.0
    except Exception:
        f1 = None
    print(f"accuracy={acc:.4f}, roc_auc={roc_auc}, f1@0.5={f1}")
    return {"accuracy": acc, "roc_auc": roc_auc, "f1@0.5": f1}


def main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats-json", default=None)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--hop", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    ds = load_csv(Path(args.csv), args.stats_json)
    evaluate(ds, Path(args.ckpt), args.device, args.batch, args.stats_json)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()

