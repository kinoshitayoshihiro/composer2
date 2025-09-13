from __future__ import annotations

"""Predict velocity and duration from feature CSV and apply to MIDI.

The command mirrors ``predict_pedal.py`` in spirit but is intentionally
minimal.  Input is a note‑wise CSV with feature columns matching the
statistics saved alongside the checkpoint.  The resulting MIDI will have
predicted velocities and durations applied.
"""

from pathlib import Path
import argparse
import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import pretty_midi as pm

from utilities.ml_velocity import MLVelocityModel
from utilities.ml_duration import DurationTransformer

from .eval_duv import (  # reuse helpers
    _as_float32,
    _as_int,
    _get_device,
    _load_duration_model,
    _load_stats,
    _resolve_workers,
    _worker_init_fn,
    load_stats_and_normalize,
    _duration_predict,
)


def _median_smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    k = 3 if k < 3 else 3  # only support median-3 for simplicity
    pad = k // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(out.size):
        out[i] = np.median(x_pad[i : i + k])
    return out


def _parse_quant(step_str: Optional[str], meta: dict) -> float:
    if step_str:
        if "_" in step_str:
            step_str = step_str.split("_", 1)[1]
        num, denom = step_str.split("/")
        return float(num) / float(denom)
    fps = float(meta.get("fps") or 0)
    hop = float(meta.get("hop") or 0)
    return hop / fps if fps > 0 and hop > 0 else 0.0


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
    for col in ["velocity", "duration"]:
        if col in df.columns:
            df[col] = _as_float32(df[col])
    if "bar" in df.columns:
        df["bar"] = _as_int(df["bar"], "int32")
    if "position" in df.columns:
        df["position"] = _as_int(df["position"], "int32")

    X, _ = load_stats_and_normalize(df, stats, strict=True)
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
    vel_pred = np.clip(_median_smooth(vel_pred, args.vel_smooth), 1, 127)

    dur_pred = None
    if "duration" in df.columns and "bar" in df.columns and "position" in df.columns and "pitch" in df.columns:
        dmodel = _load_duration_model(args.ckpt).to(device)
        dur_pred, _ = _duration_predict(df, dmodel)
        grid = _parse_quant(args.dur_quant, stats[3])
        if grid > 0:
            dur_pred = np.maximum(grid, np.round(dur_pred / grid) * grid)

    pm_obj = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    start_col = "start" if "start" in df.columns else "onset"
    for i, row in df.reset_index(drop=True).iterrows():
        start = float(row.get(start_col, 0.0))
        pitch = int(row.get("pitch", 60))
        dur = float(dur_pred[i]) if dur_pred is not None and i < len(dur_pred) else float(row.get("duration", 0.5))
        vel = int(vel_pred[i])
        end = start + max(dur, 0.0)
        inst.notes.append(pm.Note(velocity=vel, pitch=pitch, start=start, end=end))
    inst.notes.sort(key=lambda n: (n.start, n.pitch))
    pm_obj.instruments.append(inst)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pm_obj.write(str(args.out))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    p = argparse.ArgumentParser(prog="predict_duv.py")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Output MIDI path")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--stats-json", type=Path)
    p.add_argument("--num-workers", dest="num_workers", type=int)
    p.add_argument("--vel-smooth", type=int, default=1, dest="vel_smooth")
    p.add_argument("--dur-quant", type=str, dest="dur_quant")
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
