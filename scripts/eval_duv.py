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
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

try:  # Optional; only used when available
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover - optional dependency
    pearsonr = spearmanr = None  # type: ignore

from utilities.csv_io import coerce_columns
from utilities.duv_infer import (
    CSV_FLOAT32_COLUMNS,
    CSV_INT32_COLUMNS,
    OPTIONAL_FLOAT32_COLUMNS,
    OPTIONAL_INT32_COLUMNS,
    REQUIRED_COLUMNS,
    duv_sequence_predict,
    load_duv_dataframe,
    duv_verbose,
)
from utilities.ml_duration import DurationTransformer
from utilities.ml_velocity import MLVelocityModel


_duv_sequence_predict = duv_sequence_predict


def _mask_any(mask: object) -> bool:
    if mask is None:
        return False
    if isinstance(mask, torch.Tensor):
        return bool(torch.any(mask).item())
    if isinstance(mask, np.ndarray):
        return bool(np.any(mask))
    any_method = getattr(mask, "any", None)
    if callable(any_method):
        result = any_method()
        if isinstance(result, torch.Tensor):
            return bool(torch.any(result).item())
        if isinstance(result, np.ndarray):
            return bool(np.any(result))
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        item = getattr(result, "item", None)
        if callable(item):
            try:
                return bool(item())
            except Exception:
                pass
        return bool(result)
    return bool(mask)


def _ensure_int(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


_worker_seed = 0


def _worker_init_fn(worker_id: int) -> None:
    np.random.seed(_worker_seed + worker_id)
    torch.manual_seed(_worker_seed + worker_id)


def _resolve_workers(v: int | None) -> int:
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


def _first_group_length(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    group_cols = [c for c in ("track_id", "bar") if c in df.columns]
    if group_cols:
        for _, group in df.groupby(group_cols, sort=False):
            return int(len(group))
    return int(len(df))


# ---------------------------------------------------------------------------
# Stats loading / normalisation
# ---------------------------------------------------------------------------

def _load_stats(stats_json: Path | None, ckpt_path: Path) -> tuple[list[str] | None, np.ndarray, np.ndarray, dict] | None:
    path = stats_json if stats_json and stats_json.is_file() else ckpt_path.with_suffix(ckpt_path.suffix + ".stats.json")
    if not path.is_file():
        return None
    obj = json.loads(path.read_text())
    feat_cols: list[str] | None = obj.get("feat_cols")
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


def _apply_stats(df: pd.DataFrame, feat_cols: Sequence[str] | None, mean: np.ndarray, std: np.ndarray, *, strict: bool = False) -> tuple[np.ndarray, list[str]]:
    cols = list(feat_cols) if feat_cols else [c for c in df.columns if c.startswith("feat_")]
    missing = [c for c in cols if c not in df.columns]
    extra = [c for c in df.columns if c.startswith("feat_") and c not in cols]
    if strict and (missing or extra):
        raise ValueError(f"stats/CSV mismatch: missing={missing}, extra={extra}")
    arr = df.reindex(columns=cols, fill_value=0).to_numpy(dtype="float32", copy=True)
    if mean.size == arr.shape[1]:
        arr = (arr - mean) / np.maximum(std, 1e-8)
    return arr, cols


def load_stats_and_normalize(df: pd.DataFrame, stats: tuple[list[str] | None, np.ndarray, np.ndarray] | None, *, strict: bool = False) -> tuple[np.ndarray, list[str]]:
    if stats is None:
        raise ValueError("feature stats required")
    return _apply_stats(df, stats[0], stats[1], stats[2], strict=strict)


# ---------------------------------------------------------------------------
# Duration quantisation helpers
# ---------------------------------------------------------------------------

def _parse_quant(step_str: str | None, meta: dict) -> float:
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


def _duration_predict(df: pd.DataFrame, model: DurationTransformer) -> tuple[np.ndarray, np.ndarray]:
    preds: list[float] = []
    targets: list[float] = []
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


def _tensor_slice(tensor: torch.Tensor | None, length: int) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim == 0:
        return tensor.unsqueeze(0)
    if tensor.ndim == 1:
        return tensor[:length]
    return tensor.reshape(tensor.shape[0], -1)[0, :length]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    stats = _load_stats(args.stats_json, args.ckpt)
    if stats is None:
        raise SystemExit("missing or invalid stats json")

    # limitの正規化とCSV読込（filter→reset_index→head(limit) を helper 側で実施）
    limit = max(int(getattr(args, "limit", 0) or 0), 0)
    df, program_hist = load_duv_dataframe(
        args.csv,
        feature_columns=list(stats[0] or []),
        filter_expr=getattr(args, "filter_program", None),
        limit=limit,
        collect_program_hist=getattr(args, "verbose", False),
    )

    # 進捗ログ
    print({"rows": int(len(df)), "limit": limit or None}, file=sys.stderr)
    if getattr(args, "verbose", False) and program_hist is not None and not program_hist.empty:
        top = program_hist.head(10)
        print({"program_top": top.to_dict()}, file=sys.stderr)

    # track_id が無ければ file から作る
    if "track_id" not in df.columns and "file" in df.columns:
        df["track_id"] = pd.factorize(df["file"])[0].astype("int32")

    # dtypeを揃える：統一キャスト（存在する列のみ適用）
    # stats[0] は dense 特徴量列（float想定）
    float_cols = (
        set(stats[0] or [])
        | (set(CSV_FLOAT32_COLUMNS) - {"velocity"})
        | set(OPTIONAL_FLOAT32_COLUMNS)
    )
    if "beat_bin" in df.columns:
        float_cols.add("beat_bin")

    # int列は基本スキーマ＋必須列（velocity/durationはfloatなので除外）＋オプションint
    int_cols = set(CSV_INT32_COLUMNS) | (
        {c for c in REQUIRED_COLUMNS if c not in {"velocity", "duration"}}
    ) | set(OPTIONAL_INT32_COLUMNS)
    int_cols.discard("pitch")
    int_cols.discard("program")

    df = coerce_columns(df, float32=float_cols, int32=int_cols)

    # …この後の処理へ …

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
    extras = {
        "use_bar_beat": bool(getattr(core, "use_bar_beat", False)),
        "section_emb": getattr(core, "section_emb", None) is not None,
        "mood_emb": getattr(core, "mood_emb", None) is not None,
        "vel_bucket_emb": getattr(core, "vel_bucket_emb", None) is not None,
        "dur_bucket_emb": getattr(core, "dur_bucket_emb", None) is not None,
    }
    print(
        {
            "ckpt": str(args.ckpt),
            "loader": loader_type,
            "d_model": d_model or None,
            "max_len": max_len or None,
            "heads": {k: bool(v) for k, v in heads.items()},
            "extras": extras,
        }
    )

    vel_model = vel_model.to(device).eval()
    verbose = duv_verbose(getattr(args, "verbose", False))
    try:
        duv_preds = _duv_sequence_predict(df, vel_model, device, verbose=verbose)
    except Exception as exc:
        raise RuntimeError(f"DUV inference failed (rows={len(df)}): {exc}") from exc

    y_vel = df.get("velocity")
    if y_vel is None:
        raise SystemExit("CSV missing 'velocity' column")
    y_vel = y_vel.to_numpy(dtype="float32", copy=False)

    vel_pred: np.ndarray | None = None
    vel_mask: np.ndarray | None = None
    if duv_preds is not None and _mask_any(duv_preds["velocity_mask"]):
        vel_pred = duv_preds["velocity"].astype("float32", copy=False)
        vel_mask = duv_preds["velocity_mask"]
    else:
        if getattr(vel_model, "requires_duv_feats", False):
            missing = sorted(REQUIRED_COLUMNS - set(df.columns))
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
        vel_mask = np.ones_like(vel_pred, dtype=bool)

    if (
        verbose
        and vel_pred is not None
        and (duv_preds is None or not _mask_any(duv_preds["velocity_mask"]))
    ):
        print(
            {
                "duv_preview": {
                    "velocity_head": [float(v) for v in vel_pred[:8].tolist()],
                }
            },
            file=sys.stderr,
        )

    # Duration
    dur_pred: np.ndarray | None = None
    dur_target_seq: np.ndarray | None = None
    dur_mask: np.ndarray | None = None
    if duv_preds is not None and "duration" in df.columns and _mask_any(duv_preds["duration_mask"]):
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

    if vel_pred is not None and vel_mask is not None and _mask_any(vel_mask):
        vel_targets = y_vel[vel_mask]
        vel_values = vel_pred[vel_mask]
        diff = vel_values - vel_targets
        metrics["velocity_mae"] = float(np.mean(np.abs(diff)))
        metrics["velocity_rmse"] = float(np.sqrt(np.mean(diff**2)))
        metrics["velocity_count"] = int(vel_targets.size)
        constant_pred = bool(vel_values.size) and float(np.ptp(vel_values)) < 1e-6
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
        metrics["velocity_pearson"] = None
        metrics["velocity_spearman"] = None
        target_constant = bool(vel_targets.size) and float(np.ptp(vel_targets)) < 1e-6
        if constant_pred:
            metrics["velocity_stats_note"] = "constant_prediction"
        if (
            pearsonr is not None
            and not constant_pred
            and not target_constant
            and vel_targets.size > 1
        ):
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

    if getattr(args, "verbose", False) and vel_pred is not None:
        seq_len = _first_group_length(df)
        preview = vel_pred[:8].astype("float32", copy=False).tolist()
        print(
            {
                "preview": {
                    "seq_len": seq_len,
                    "d_model_effective": d_model or None,
                    "velocity": [float(v) for v in preview],
                }
            },
            file=sys.stderr,
        )

    out_text = json.dumps(metrics, ensure_ascii=False)
    print(out_text)
    if getattr(args, "out_json", None):
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text + "\n", encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
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
    p.add_argument(
        "--dur-quant",
        type=str,
        dest="dur_quant",
        help="Quantise durations to the given fraction (e.g. 1/16); omit to keep predictions",
    )
    p.add_argument("--num-workers", dest="num_workers", type=int)
    p.add_argument(
        "--filter-program",
        dest="filter_program",
        help=(
            "Optional pandas query applied immediately after loading the CSV (before --limit); "
            "the DataFrame index is reset"
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of rows to retain after filtering (0 loads all rows)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Emit additional DUV diagnostics including optional zero-filled feature usage",
    )
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
