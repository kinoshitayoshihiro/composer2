from __future__ import annotations

"""Predict velocity and duration from feature CSV and apply to MIDI.

The command mirrors ``predict_pedal.py`` in spirit but is intentionally
minimal.  Input is a note‑wise CSV with feature columns matching the
statistics saved alongside the checkpoint.  The resulting MIDI will have
predicted velocities and durations applied.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi as pm
import torch
from torch.utils.data import DataLoader, TensorDataset

from utilities.ml_velocity import MLVelocityModel

from .eval_duv import (  # reuse helpers
    _as_float32,
    _as_int,
    _duration_predict,
    _duv_sequence_predict,
    _get_device,
    _load_duration_model,
    _load_stats,
    _parse_quant,
    _resolve_workers,
    _worker_init_fn,
    load_stats_and_normalize,
)


def _median_smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    if k % 2 == 0:
        raise ValueError("median window must be odd")
    pad = k // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(out.size):
        out[i] = np.median(x_pad[i : i + k])
    return out


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

    device = _get_device(args.device)
    vel_model = MLVelocityModel.load(str(args.ckpt)).to(device).eval()
    duv_preds = _duv_sequence_predict(df, vel_model, device)

    vel_pred: np.ndarray | None
    vel_mask: np.ndarray | None = None
    if duv_preds is not None and duv_preds["velocity_mask"].any():
        vel_mask = duv_preds["velocity_mask"]
        base = df.get("velocity")
        if base is not None:
            vel_pred = base.to_numpy(dtype="float32", copy=False).copy()
        else:
            vel_pred = np.zeros(len(df), dtype=np.float32)
        vel_pred[vel_mask] = duv_preds["velocity"].astype("float32", copy=False)[vel_mask]
    else:
        if getattr(vel_model, "requires_duv_feats", False):
            required = {"pitch", "velocity", "duration", "position"}
            missing = sorted(required - set(df.columns))
            detail = f"; missing columns: {', '.join(missing)}" if missing else ""
            raise RuntimeError(
                "DUV checkpoint requires phrase-level features (pitch, velocity, duration, position) "
                "for inference and cannot fall back to dense feature tensors"
                f"{detail}."
            )
        preds: list[np.ndarray] = []
        X, _ = load_stats_and_normalize(df, stats, strict=True)
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=_resolve_workers(args.num_workers),
            worker_init_fn=_worker_init_fn,
        )
        with torch.no_grad():
            for (xb,) in loader:
                out = vel_model(xb.to(device))
                preds.append(out.cpu().numpy())
        vel_pred = np.concatenate(preds, axis=0).astype("float32")

    if args.vel_smooth > 1:
        smoothed = _median_smooth(vel_pred.copy(), args.vel_smooth)
        if args.smooth_pred_only and vel_mask is not None and vel_mask.any():
            vel_pred[vel_mask] = smoothed[vel_mask]
        else:
            vel_pred = smoothed
    vel_pred = np.clip(vel_pred, 1, 127)

    dur_pred: np.ndarray | None = None
    if duv_preds is not None and duv_preds["duration_mask"].any():
        base = df.get("duration")
        if base is not None:
            dur_pred = base.to_numpy(dtype="float32", copy=False).copy()
        else:
            dur_pred = np.zeros(len(df), dtype=np.float32)
        mask = duv_preds["duration_mask"]
        dur_pred[mask] = duv_preds["duration"].astype("float32", copy=False)[mask]
    elif "duration" in df.columns and "bar" in df.columns and "position" in df.columns and "pitch" in df.columns:
        dmodel = _load_duration_model(args.ckpt).to(device)
        dur_pred, _ = _duration_predict(df, dmodel)
    grid = _parse_quant(args.dur_quant, stats[3])
    if grid > 0 and dur_pred is not None:
        dur_pred = np.maximum(grid, np.round(dur_pred / grid) * grid)
    elif grid <= 0:
        print({"dur_quant": "skipped", "grid": float(grid)}, file=sys.stderr)

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


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    p = argparse.ArgumentParser(prog="predict_duv.py")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Checkpoint (.ckpt state_dict or .ts TorchScript)",
    )
    p.add_argument("--out", type=Path, required=True, help="Output MIDI path")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--stats-json", type=Path)
    p.add_argument("--num-workers", dest="num_workers", type=int)
    p.add_argument(
        "--vel-smooth",
        type=int,
        default=1,
        dest="vel_smooth",
        choices=(1, 3, 5),
        help="Velocity median smoothing window; 1 disables, 3/5 apply a median filter",
    )
    p.add_argument(
        "--smooth-pred-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When smoothing velocities, only adjust bins predicted by the model",
    )
    p.add_argument("--dur-quant", type=str, dest="dur_quant")
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
