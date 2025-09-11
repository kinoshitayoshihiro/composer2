from __future__ import annotations

import argparse, os, json
from pathlib import Path

try:
    import pandas as pd
except Exception as e:  # pragma: no cover - guidance
    raise RuntimeError("pandas is required for predict_pedal. Please `pip install pandas`.") from e
import numpy as np
import torch
import pretty_midi

from ml_models.pedal_model import PedalModel
try:
    from utilities.pedal_frames import HOP_LENGTH, SR  # optional
except Exception:  # pragma: no cover - optional
    HOP_LENGTH, SR = None, None


def read_csv(path: Path) -> pd.DataFrame:
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
    return df


def _make_x(df: pd.DataFrame, stats_json: str | None) -> np.ndarray:
    chroma_cols = [c for c in df.columns if c.startswith("chroma_")]
    x_cols = chroma_cols + (["rel_release"] if "rel_release" in df.columns else [])
    x = df[x_cols].values.astype("float32")
    if stats_json and os.path.exists(stats_json):
        s = json.load(open(stats_json, "r"))
        cols = s.get("feat_cols", x_cols)
        mean = np.array([s["mean"][c] for c in cols], dtype="float32")
        std = np.array([s["std"][c] for c in cols], dtype="float32")
        std[std == 0] = 1.0
        x = df[cols].values.astype("float32")
        x = (x - mean) / std
    return x


def predict(df: pd.DataFrame, ckpt: Path, device: str, stats_json: str | None, batch: int) -> np.ndarray:
    dev = torch.device(device)
    x_np = _make_x(df, stats_json)
    x = torch.tensor(x_np)
    model = PedalModel()
    state = torch.load(ckpt, map_location=dev)
    sd = state.get("state_dict", state)
    model.load_state_dict(sd, strict=False)
    model.to(dev)
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), max(1, batch)):
            xb = x[i : i + batch].to(dev)
            ob = model(xb).sigmoid().cpu().numpy().ravel()
            outs.append(ob)
    return np.concatenate(outs) if outs else np.zeros((0,), dtype="float32")


def write_midi(prob: np.ndarray, path: Path, fps: float | None, hop: int, sr: int) -> None:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    if fps and fps > 0:
        step = 1.0 / float(fps)
    elif HOP_LENGTH and SR:
        step = float(HOP_LENGTH) / float(SR)
    elif hop and sr:
        step = float(hop) / float(sr)
    else:
        step = 0.01
    prev = 0
    for i, p in enumerate(prob):
        state = int(p >= 0.5)
        if state != prev:
            inst.control_changes.append(
                pretty_midi.ControlChange(64, 127 if state else 0, i * step)
            )
            prev = state
    pm.instruments.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))


def main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--csv", dest="inp", help="alias of --in")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-mid", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--out", dest="out_mid", help="alias of --out-mid")
    ap.add_argument("--stats-json", default=None)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--hop", type=int, default=16)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.stats_json is None and args.ckpt:
        cand = args.ckpt + ".stats.json"
        if os.path.exists(cand):
            args.stats_json = cand

    if args.out_mid is None and args.out_dir is None:
        args.out_mid = args.out_mid or "outputs/pedal_pred.mid"

    df = read_csv(Path(args.inp))
    prob = predict(df, Path(args.ckpt), args.device, args.stats_json, args.batch)
    if args.out_mid:
        fps = None
        if args.stats_json and os.path.exists(args.stats_json):
            try:
                fps = float(json.load(open(args.stats_json))["fps"])
            except Exception:
                fps = None
        write_midi(prob, Path(args.out_mid), fps=fps, hop=args.hop, sr=args.sr)
    print("done")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()

