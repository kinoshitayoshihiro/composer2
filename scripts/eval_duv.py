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

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def _ensure_int(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


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
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # pragma: no cover - macOS
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
# Duration quantisation helpers
# ---------------------------------------------------------------------------

def _parse_quant(step_str: Optional[str], meta: dict) -> float:
    if step_str:
        if "_" in step_str:
            step_str = step_str.split("_", 1)[1]
        try:
            num, denom = step_str.split("/")
            return float(num) / float(denom)
        except Exception:
            return 0.0
    fps = float(meta.get("fps") or 0)
    hop = float(meta.get("hop") or 0)
    return hop / fps if fps > 0 and hop > 0 else 0.0


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


def _tensor_slice(tensor: Optional[torch.Tensor], length: int) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.ndim == 0:
        return tensor.unsqueeze(0)
    if tensor.ndim == 1:
        return tensor[:length]
    return tensor.reshape(tensor.shape[0], -1)[0, :length]


def _prepare_feature_tensor(series: pd.Series, pad: int, *, dtype: str, numeric: bool = True) -> torch.Tensor:
    if numeric:
        vals = pd.to_numeric(series, errors="coerce").fillna(0)
    else:
        vals = series.fillna(0)
    data = vals.to_list() + ([0] * pad if dtype.startswith("int") else [0.0] * pad)
    if dtype == "int64":
        return torch.tensor(data, dtype=torch.long)
    return torch.tensor(data, dtype=torch.float32)


def _duv_sequence_predict(
    df: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
) -> Optional[Dict[str, np.ndarray]]:
    if not getattr(model, "requires_duv_feats", False):
        return None
    if not {"pitch", "velocity", "duration", "position"}.issubset(df.columns):
        return None

    core = getattr(model, "core", model)
    max_len = _ensure_int(getattr(core, "max_len", getattr(model, "max_len", 16)), 16)
    has_vel = bool(getattr(model, "has_vel_head", getattr(core, "head_vel_reg", None)))
    has_dur = bool(getattr(model, "has_dur_head", getattr(core, "head_dur_reg", None)))
    if not has_vel and not has_dur:
        return None

    n = len(df)
    vel_pred = np.zeros(n, dtype=np.float32)
    dur_pred = np.zeros(n, dtype=np.float32)
    vel_mask = np.zeros(n, dtype=bool)
    dur_mask = np.zeros(n, dtype=bool)

    group_cols = ["track_id", "bar"] if "track_id" in df.columns else ["bar"]

    for _, group in df.groupby(group_cols, sort=False):
        if group.empty:
            continue
        g = group.sort_values("position")
        idx = g.index.to_numpy()
        length = len(g)
        if length > max_len:
            g = g.iloc[:max_len]
            idx = g.index.to_numpy()
            length = len(g)
        pad = max_len - length

        feats: Dict[str, torch.Tensor] = {}
        pitch_cls = (g["pitch"].to_numpy(dtype="int64") % 12).tolist() + [0] * pad
        feats["pitch_class"] = torch.tensor(pitch_cls, dtype=torch.long, device=device).unsqueeze(0)
        vel_vals = g["velocity"].to_numpy(dtype="float32").tolist() + [0.0] * pad
        feats["velocity"] = torch.tensor(vel_vals, dtype=torch.float32, device=device).unsqueeze(0)
        dur_vals = g["duration"].to_numpy(dtype="float32").tolist() + [0.0] * pad
        feats["duration"] = torch.tensor(dur_vals, dtype=torch.float32, device=device).unsqueeze(0)
        pos_vals = g["position"].to_numpy(dtype="int64").tolist() + [0] * pad
        feats["position"] = torch.tensor(pos_vals, dtype=torch.long, device=device).unsqueeze(0)

        if bool(getattr(core, "use_bar_beat", False)) and {"bar_phase", "beat_phase"}.issubset(g.columns):
            bar_phase = _prepare_feature_tensor(g["bar_phase"], pad, dtype="float32").to(device)
            beat_phase = _prepare_feature_tensor(g["beat_phase"], pad, dtype="float32").to(device)
            feats["bar_phase"] = bar_phase.unsqueeze(0)
            feats["beat_phase"] = beat_phase.unsqueeze(0)
        if getattr(core, "section_emb", None) is not None and "section" in g.columns:
            section = _prepare_feature_tensor(g["section"], pad, dtype="int64").to(device)
            feats["section"] = section.unsqueeze(0)
        if getattr(core, "mood_emb", None) is not None and "mood" in g.columns:
            mood = _prepare_feature_tensor(g["mood"], pad, dtype="int64").to(device)
            feats["mood"] = mood.unsqueeze(0)
        if getattr(core, "vel_bucket_emb", None) is not None and "vel_bucket" in g.columns:
            vel_bucket = _prepare_feature_tensor(g["vel_bucket"], pad, dtype="int64").to(device)
            feats["vel_bucket"] = vel_bucket.unsqueeze(0)
        if getattr(core, "dur_bucket_emb", None) is not None and "dur_bucket" in g.columns:
            dur_bucket = _prepare_feature_tensor(g["dur_bucket"], pad, dtype="int64").to(device)
            feats["dur_bucket"] = dur_bucket.unsqueeze(0)

        mask = torch.zeros(1, max_len, dtype=torch.bool, device=device)
        mask[0, :length] = True

        with torch.no_grad():
            raw = model(feats, mask=mask)
        if isinstance(raw, tuple):
            vel_out = raw[0] if len(raw) > 0 else None
            dur_out = raw[1] if len(raw) > 1 else None
        elif isinstance(raw, dict):
            vel_out = raw.get("vel_reg")
            dur_out = raw.get("dur_reg")
        else:
            vel_out = raw
            dur_out = None

        vel_slice = _tensor_slice(vel_out, length)
        if has_vel and vel_slice is not None:
            vel_vals = (
                vel_slice.clamp(0.0, 1.0).mul(127.0).round().clamp(0, 127).to(torch.float32).cpu().numpy()
            )
            vel_pred[idx[: length]] = vel_vals
            vel_mask[idx[: length]] = True

        dur_slice = _tensor_slice(dur_out, length)
        if has_dur and dur_slice is not None:
            dur_vals = torch.expm1(dur_slice).clamp(min=0.0).to(torch.float32).cpu().numpy()
            dur_pred[idx[: length]] = dur_vals
            dur_mask[idx[: length]] = True

    return {
        "velocity": vel_pred,
        "velocity_mask": vel_mask,
        "duration": dur_pred,
        "duration_mask": dur_mask,
    }


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
    df = df.reset_index(drop=True)

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
    loader_type = getattr(vel_model, "_duv_loader", "ts" if str(args.ckpt).endswith((".ts", ".torchscript")) else "ckpt")
    core = getattr(vel_model, "core", vel_model)
    d_model = _ensure_int(getattr(vel_model, "d_model", getattr(core, "d_model", 0)), 0)
    max_len = _ensure_int(getattr(vel_model, "max_len", getattr(core, "max_len", 0)), 0)
    heads = getattr(
        vel_model,
        "heads",
        {
            "vel_reg": bool(getattr(vel_model, "has_vel_head", getattr(core, "head_vel_reg", None))),
            "dur_reg": bool(getattr(vel_model, "has_dur_head", getattr(core, "head_dur_reg", None))),
        },
    )
    print(
        {
            "ckpt": str(args.ckpt),
            "loader": loader_type,
            "d_model": d_model or None,
            "max_len": max_len or None,
            "heads": {k: bool(v) for k, v in heads.items()},
        }
    )

    vel_model = vel_model.to(device).eval()
    duv_preds = _duv_sequence_predict(df, vel_model, device)

    vel_pred: Optional[np.ndarray] = None
    vel_mask: Optional[np.ndarray] = None
    if duv_preds is not None and duv_preds["velocity_mask"].any():
        vel_pred = duv_preds["velocity"].astype("float32", copy=False)
        vel_mask = duv_preds["velocity_mask"]
    else:
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for (xb,) in loader:
                out = vel_model(xb.to(device))
                preds.append(out.cpu().numpy())
        vel_pred = np.concatenate(preds, axis=0).astype("float32")
        vel_mask = np.ones_like(vel_pred, dtype=bool)

    # Duration
    dur_pred: Optional[np.ndarray] = None
    dur_target_seq: Optional[np.ndarray] = None
    dur_mask: Optional[np.ndarray] = None
    if duv_preds is not None and "duration" in df.columns and duv_preds["duration_mask"].any():
        dur_pred = duv_preds["duration"].astype("float32", copy=False)
        dur_target_seq = df["duration"].to_numpy(dtype="float32", copy=False)
        dur_mask = duv_preds["duration_mask"]
    else:
        if "duration" in df.columns and "bar" in df.columns and "position" in df.columns and "pitch" in df.columns:
            dmodel = _load_duration_model(args.ckpt)
            dmodel = dmodel.to(device)
            pred_dur, tgt_dur = _duration_predict(df, dmodel)
            if pred_dur.size and tgt_dur.size:
                dur_pred = pred_dur.astype("float32", copy=False)
                dur_target_seq = tgt_dur.astype("float32", copy=False)

    grid = _parse_quant(getattr(args, "dur_quant", None), stats[3])
    if dur_pred is not None:
        if grid > 0:
            dur_pred = np.maximum(grid, np.round(dur_pred / grid) * grid)
        else:
            print({"dur_quant": "skipped", "grid": float(grid)}, file=sys.stderr)

    metrics: dict[str, object] = {}

    if vel_pred is not None and vel_mask is not None and vel_mask.any():
        vel_targets = y_vel[vel_mask]
        vel_values = vel_pred[vel_mask]
        diff = vel_values - vel_targets
        metrics["velocity_mae"] = float(np.mean(np.abs(diff)))
        metrics["velocity_rmse"] = float(np.sqrt(np.mean(diff**2)))
        metrics["velocity_count"] = int(vel_targets.size)
        if "beat_bin" in df.columns:
            beat_vals = df.loc[vel_mask, "beat_bin"].to_numpy()
            if beat_vals.size:
                by = {}
                for beat in np.unique(beat_vals):
                    sel = beat_vals == beat
                    by[str(int(beat))] = float(
                        np.mean(np.abs(vel_values[sel] - vel_targets[sel]))
                    )
                metrics["velocity_mae_by_beat"] = by
        if pearsonr is not None and vel_targets.size > 1:
            metrics["velocity_pearson"] = float(pearsonr(vel_values, vel_targets)[0])
            metrics["velocity_spearman"] = float(spearmanr(vel_values, vel_targets)[0])

    if dur_pred is not None:
        if dur_mask is not None and dur_target_seq is not None:
            tgt = dur_target_seq[dur_mask]
            pred_vals = dur_pred[dur_mask]
        else:
            tgt = dur_target_seq
            pred_vals = dur_pred
        if tgt is not None and pred_vals is not None and tgt.size and pred_vals.size:
            diff = pred_vals - tgt
            metrics["duration_mae"] = float(np.mean(np.abs(diff)))
            metrics["duration_rmse"] = float(np.sqrt(np.mean(diff**2)))
            metrics["duration_count"] = int(tgt.size)

    out_text = json.dumps(metrics, ensure_ascii=False)
    print(out_text)
    if getattr(args, "out_json", None):
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text + "\n", encoding="utf-8")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    p = argparse.ArgumentParser(prog="eval_duv.py")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Checkpoint (.ckpt state_dict or .ts TorchScript)",
    )
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--stats-json", type=Path)
    p.add_argument("--out-json", type=Path, help="Optional path to write metrics JSON")
    p.add_argument("--dur-quant", type=str, dest="dur_quant")
    p.add_argument("--num-workers", dest="num_workers", type=int)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
