from __future__ import annotations

"""Predict CC64 (sustain pedal) from MIDI **or precomputed feature CSV** using a trained model.

This file resolves merge conflicts by **unifying** both branches:
- "CSV → probs → CC64" pipeline (older script)
- "MIDI → feature extract → windowed model → hysteresis → CC64" pipeline (newer script)

Key features
- Accepts **MIDI file/dir** *or* **feature CSV** (`--feat-csv` / `--csv`).
- Loads feature stats from `--stats-json` or `<ckpt>.stats.json` (supports
  *named* per-column stats with `feat_cols` **or** plain arrays in column order).
- Windowed inference with overlap-averaging back to **per-frame** probabilities.
- Optional Gaussian smoothing, robust **hysteresis** & minimal-duration postprocess.
- Device auto-detection (CUDA → MPS → CPU).

Examples
  Single MIDI file:
    python predict_pedal.py \
      --in data/songs_norm/omokage.mid \
      --ckpt checkpoints/pedal.ckpt \
      --out-mid outputs/pedal_pred.mid

  Directory of MIDIs:
    python predict_pedal.py \
      --in data/songs_norm \
      --ckpt checkpoints/pedal.ckpt \
      --out-dir outputs/pedal_pred

  Using a precomputed **feature CSV** (chroma_* + optional rel_release):
    python predict_pedal.py \
      --feat-csv data/features/track01.csv \
      --ckpt checkpoints/pedal.ckpt \
      --out-mid outputs/track01.pedal.mid
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
try:
    import numpy as np
except Exception as e:  # pragma: no cover - friendlier error
    raise RuntimeError("numpy is required for predict_pedal. Try: `pip install numpy`") from e
try:
    import pandas as pd
except Exception as e:  # pragma: no cover - friendlier error
    raise RuntimeError("pandas is required for predict_pedal. Try: `pip install pandas`") from e
import torch
from torch.utils.data import DataLoader, TensorDataset
import pretty_midi as pm

from ml_models.pedal_model import PedalModel
try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - optional dependency
    roc_auc_score = None  # type: ignore
try:
    from eval_pedal import load_stats_and_normalize
except Exception:  # pragma: no cover - package or script execution
    from scripts.eval_pedal import load_stats_and_normalize  # type: ignore


def _resolve_workers_cli(v):
    if v is not None:
        return max(int(v), 0)
    env = os.getenv("COMPOSER2_NUM_WORKERS")
    return max(int(env), 0) if (env and env.isdigit()) else 0

# Optional dependency: feature extraction from MIDI
try:
    from utilities.pedal_frames import (
        extract_from_midi,
        SR as DEFAULT_SR,
        HOP_LENGTH as DEFAULT_HOP,
    )
except Exception:  # pragma: no cover - optional; enable CSV mode even without this module
    extract_from_midi = None  # type: ignore
    DEFAULT_SR = 22050  # safe defaults
    DEFAULT_HOP = 512


# ------------------------------
# Stats loading / normalization
# ------------------------------

def _load_stats(ckpt_path: Path, stats_json: Optional[Path] = None) -> Optional[tuple[Optional[List[str]], np.ndarray, np.ndarray, dict]]:
    """Return (feat_cols, mean, std, meta) from JSON.

    Supports both schemas:
      A) {"feat_cols": [...], "mean": {col: val}, "std": {col: val}, ...}
      B) {"mean": [...], "std": [...], ...}  # arrays match column order
    """
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
    if isinstance(meta.get("fps"), (int, float)):
        meta["fps"] = float(meta["fps"])
    else:
        meta["fps"] = None
    return feat_cols, mean, std, meta


def _apply_stats(df: pd.DataFrame, feat_cols: Optional[Sequence[str]], mean: np.ndarray, std: np.ndarray, *, strict: bool = False) -> tuple[np.ndarray, List[str]]:
    cols = list(feat_cols) if feat_cols else _feature_columns(df, None)
    have = [c for c in df.columns if c.startswith("chroma_") or c == "rel_release"]
    missing = [c for c in cols if c not in have]
    extra = [c for c in have if c not in cols]
    if strict and (missing or extra):
        raise ValueError(f"stats/CSV mismatch: missing={missing}, extra={extra}")
    if missing:
        sys.stderr.write(f"[composer2] warn: filling missing feat cols with zeros: {missing}\n")
    arr = df.reindex(columns=cols, fill_value=0).to_numpy(dtype="float32", copy=True)
    if arr.shape[1] != mean.size:
        print(f"[composer2] WARNING: stats dimension mismatch: X={arr.shape[1]} mean={mean.size}; skip normalization")
        return arr, cols
    arr = (arr - mean) / np.maximum(std, 1e-8)
    if missing:
        arr[:, [cols.index(c) for c in missing]] = 0.0
    return arr, cols


# ------------------------------
# CSV utilities
# ------------------------------

def _as_float32(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float32")


def _feature_columns(df: pd.DataFrame, prefer: Optional[Sequence[str]] = None) -> List[str]:
    """Pick feature columns in a stable order.
    Uses `prefer` if all present; otherwise chroma_* (+ optional rel_release).
    """
    if prefer and all(c in df.columns for c in prefer):
        return list(prefer)
    chroma_cols = [c for c in df.columns if c.startswith("chroma_")]
    cols = list(chroma_cols)
    if "rel_release" in df.columns:
        cols.append("rel_release")
    return cols


def read_feature_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # relax requirements: only need feature columns for prediction
    # sanitize numerics
    for c in [c for c in df.columns if c.startswith("chroma_")]:
        df[c] = _as_float32(df[c])
    if "rel_release" in df.columns:
        df["rel_release"] = _as_float32(df["rel_release"])
    if "pedal_state" in df.columns:
        df["pedal_state"] = pd.to_numeric(df["pedal_state"], errors="coerce").fillna(0).astype("uint8")
    return df


# ------------------------------
# Model loading
# ------------------------------

def _load_model(ckpt: Path, device: torch.device) -> PedalModel:
    state = torch.load(ckpt, map_location="cpu")
    model = PedalModel()
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
        # Strip optional leading "model." prefix (Lightning convention)
        new_sd = {
            (k.removeprefix("model.") if hasattr(k, "removeprefix") else k.split("model.", 1)[-1] if k.startswith("model.") else k): v
            for k, v in sd.items()
        }
        model.load_state_dict(new_sd, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    return model.to(device).eval()


# ------------------------------
# Inference (windowed + overlap averaging)
# ------------------------------

def _make_windows(arr: torch.Tensor, win: int, hop: int) -> tuple[torch.Tensor, List[int]]:
    T = arr.shape[0]
    if T < win:
        return torch.empty(0, win, arr.shape[1], dtype=arr.dtype), []
    starts = list(range(0, T - win + 1, hop))
    out = torch.stack([arr[s : s + win] for s in starts], dim=0)
    return out, starts


def predict_per_frame(df: pd.DataFrame, *, feat_cols: Optional[List[str]], mean: Optional[np.ndarray], std: Optional[np.ndarray],
                      model: PedalModel, window: int, hop: int, device: torch.device, batch: int = 64,
                      num_workers: int = 0, strict: bool = False) -> np.ndarray:
    stats = (feat_cols, mean, std) if (mean is not None and std is not None) else None
    X, cols = load_stats_and_normalize(df, stats, strict=strict)
    xt = torch.from_numpy(X)
    win, starts = _make_windows(xt, window, hop)
    if len(starts) == 0:
        # too short: return zeros
        return np.zeros(X.shape[0], dtype=np.float32)

    _nw = max(int(num_workers), 0)
    _pw = _nw > 0
    _pf = 2 if _nw > 0 else None
    base_seed = torch.initial_seed()
    def _worker_init_fn(worker_id: int) -> None:
        np.random.seed(base_seed + worker_id)
        torch.manual_seed(base_seed + worker_id)

    bs = max(1, int(batch))
    starts_t = torch.tensor(starts, dtype=torch.int64)
    ds = TensorDataset(win, starts_t)
    dl_kwargs = dict(batch_size=bs, shuffle=False, drop_last=False,
                    num_workers=_nw, persistent_workers=_pw,
                    worker_init_fn=_worker_init_fn)
    if _pf is not None and _nw > 0:
        dl_kwargs["prefetch_factor"] = _pf
    if device.type == "cuda":
        dl_kwargs["pin_memory"] = True
    try:
        loader = DataLoader(ds, **dl_kwargs)
    except Exception:
        print("[composer2] DataLoader failed with workers, falling back to num_workers=0")
        fb_kwargs = dict(batch_size=bs, shuffle=False, drop_last=False,
                         num_workers=0, persistent_workers=False,
                         worker_init_fn=_worker_init_fn)
        if device.type == "cuda":
            fb_kwargs["pin_memory"] = True
        loader = DataLoader(ds, **fb_kwargs)
    prob_sum = np.zeros(X.shape[0], dtype=np.float64)
    logit_sum = np.zeros(X.shape[0], dtype=np.float64)
    cnt = np.zeros(X.shape[0], dtype=np.int32)

    with torch.no_grad():
        for wb, sb in loader:
            wb = wb.to(device)
            logits = model(wb)               # expect (B, win) or (B, win, 1)
            if logits.ndim == 3:
                logits = logits.squeeze(-1)
            pb = torch.sigmoid(logits).cpu().numpy()
            lg = logits.cpu().numpy()
            for k, s in enumerate(sb.numpy().tolist()):
                e = s + window
                prob_sum[s:e] += pb[k]
                logit_sum[s:e] += lg[k]
                cnt[s:e] += 1

    cnt[cnt == 0] = 1
    prob = (prob_sum / cnt).astype(np.float32)

    # salvage if probabilities collapse
    if float(prob.max() - prob.min()) < 1e-3:
        z = (logit_sum / cnt).astype(np.float32)
        z = (z - z.mean()) / (z.std() + 1e-6)
        prob = 1.0 / (1.0 + np.exp(-z))

    return prob


# ------------------------------
# Post-processing (hysteresis + min lengths)
# ------------------------------

def hysteresis_threshold(prob: np.ndarray, *, k_std: float, min_margin: float, off_margin: float, hyst_delta: float, eps_on: float) -> tuple[float, float]:
    m = float(np.median(prob))
    s = float(prob.std())
    thr = max(m + k_std * s, m + min_margin)
    on_thr = float(np.clip(thr, 0.0, 1.0))
    off_thr = float(max(on_thr - hyst_delta, m + off_margin))
    return on_thr, off_thr


def postprocess_on(prob: np.ndarray, *, on_thr: float, off_thr: float, fps: float,
                   min_on_sec: float, min_hold_sec: float, eps_on: float,
                   off_consec_sec: float = 0.0) -> np.ndarray:
    on = np.zeros_like(prob, dtype=np.uint8)
    state = 0
    need_off = int(round(max(0.0, off_consec_sec) * fps))
    off_run = 0
    for i, p in enumerate(prob):
        if state == 0 and (p > on_thr + eps_on):
            state = 1
            off_run = 0
        elif state == 1 and (p < off_thr - eps_on):
            if need_off <= 1:
                state = 0
            else:
                off_run += 1
                if off_run >= need_off:
                    state = 0
                    off_run = 0
        on[i] = state
    # dilate ON segments to minimum length
    min_on_frames = int(round(min_on_sec * fps))
    if min_on_frames > 1:
        k = np.ones(min_on_frames, dtype=int)
        on = (np.convolve(on, k, mode="same") > 0).astype(np.uint8)
    # minimum hold for both states
    if min_hold_sec and min_hold_sec > 0:
        min_len = int(round(min_hold_sec * fps))
        i = 0
        L = len(on)
        while i < L:
            j = i
            while j < L and on[j] == on[i]:
                j += 1
            if (j - i) < min_len:
                on[i:j] = 1 - on[i]
            i = j
    return on


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
    """Compute F1/precision/recall/accuracy/ROC-AUC with single-class safety."""
    assert y_true.shape == prob.shape == pred.shape
    y = y_true.astype(int)
    p = pred.astype(int)
    acc = float((p == y).mean()) if y.size else 0.0
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
    if np.unique(y).size > 1:
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        auc = float(roc_auc_score(y, prob)) if roc_auc_score is not None else None
    else:
        f1 = None
        auc = None
    return {
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "accuracy": acc,
        "roc_auc": auc,
    }


# ------------------------------
# I/O helpers
# ------------------------------

def write_cc64(out_mid: Path, on: np.ndarray, step_sec: float) -> int:
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    midi.instruments.append(inst)
    cc: List[pm.ControlChange] = []
    cur = None
    for i, v in enumerate(on):
        t = float(i * step_sec)
        val = 127 if v else 0
        if cur != val:
            cc.append(pm.ControlChange(number=64, value=val, time=t))
            cur = val
    inst.control_changes = cc
    out_mid.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(out_mid))
    return len(cc)


def iter_midis(root: Path) -> Iterable[Path]:
    if root.is_file() and root.suffix.lower() in {".mid", ".midi"}:
        yield root
        return
    for p in root.rglob("*"):
        if p.suffix.lower() in {".mid", ".midi"}:
            yield p


# ------------------------------
# Main
# ------------------------------

def _auto_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # Inputs
    ap.add_argument("--in", dest="inp", type=Path, help="input MIDI file or directory")
    ap.add_argument("--feat-csv", dest="feat_csv", type=Path, help="precomputed feature CSV (chroma_* [+ rel_release])")
    ap.add_argument("--csv", dest="feat_csv_alias", type=Path, help="alias of --feat-csv")
    ap.add_argument("--ckpt", type=Path, required=True)
    # Outputs
    ap.add_argument("--out-mid", type=Path, help="output MIDI path for single-file mode")
    ap.add_argument("--out", dest="out_mid_alias", type=Path, help="alias of --out-mid")
    ap.add_argument("--out-dir", type=Path, help="output directory for directory mode")
    ap.add_argument("--debug-json", type=Path, help="write debug metrics JSON")
    ap.add_argument("--dump-prob", type=Path, help="save per-frame probabilities to NPZ")
    ap.add_argument("--compute-f1", action="store_true", help="compute F1/precision/recall if labels available")
    # Feature/Model params
    ap.add_argument("--stats-json", type=Path, help="feature stats JSON (defaults to <ckpt>.stats.json)")
    ap.add_argument("--strict-stats", action="store_true", help="enforce stats/CSV column match")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--hop", type=int, default=16)
    # Extraction timing (used for MIDI extraction or CSV fps fallback)
    ap.add_argument("--sr", type=int, default=DEFAULT_SR)
    ap.add_argument("--feat-hop", type=int, default=DEFAULT_HOP, help="feature extraction hop length for librosa")
    # Runtime
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (optional, or set COMPOSER2_NUM_WORKERS)")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    # Thresholding / hysteresis
    ap.add_argument("--on-th", type=float, help="fixed ON threshold; if set, overrides auto")
    ap.add_argument("--off-th", type=float, help="fixed OFF threshold; if set, overrides auto")
    ap.add_argument("--k-std", type=float, default=2.0)
    ap.add_argument("--min-margin", type=float, default=0.002)
    ap.add_argument("--off-margin", type=float, default=0.0005)
    ap.add_argument("--hyst-delta", type=float, default=0.002)
    ap.add_argument("--min-on-sec", type=float, default=0.05)
    ap.add_argument("--min-hold-sec", type=float, default=0.05)
    ap.add_argument("--eps-on", type=float, default=1e-6)
    ap.add_argument("--target-on-ratio", type=float, default=None,
                    help="target ON ratio; sets on_thr via quantile (1-ratio) and off_thr=on_thr-hyst_delta")
    # Optional smoothing / debouncing
    ap.add_argument("--smooth-sigma", type=float, default=0.0, help="Gaussian smoothing sigma (frames) before hysteresis; 0 to disable")
    ap.add_argument("--off-consec-sec", type=float, default=0.0, help="require this many consecutive seconds below OFF threshold before turning OFF")

    args = ap.parse_args(argv)
    if args.feat_csv and args.feat_csv_alias and args.feat_csv != args.feat_csv_alias:
        raise SystemExit("conflict: --feat-csv and --csv point to different files")
    args.feat_csv = args.feat_csv or args.feat_csv_alias
    if args.inp and args.feat_csv:
        raise SystemExit("conflict: cannot use --in with --feat-csv/--csv")
    if args.out_mid and args.out_mid_alias and args.out_mid != args.out_mid_alias:
        raise SystemExit("conflict: --out and --out-mid point to different files")
    args.out_mid = args.out_mid or args.out_mid_alias
    if args.out_mid is None and args.out_dir is None:
        args.out_mid = Path("outputs/pedal_pred.mid")

    # Validate input mode
    if args.feat_csv is None and args.inp is None:
        raise SystemExit("either --feat-csv/--csv or --in (MIDI) must be provided")

    device = _auto_device(args.device)
    _nw = _resolve_workers_cli(args.num_workers)
    _pw = _nw > 0
    _pf = 2 if _nw > 0 else None
    print(f"[composer2] num_workers={_nw} (persistent={_pw}, prefetch_factor={_pf or 'n/a'})")

    # Stats + model
    stats = _load_stats(args.ckpt, args.stats_json)
    if stats is None:
        print("[composer2] WARNING: feature stats not found; proceeding without normalization")
        feat_cols = None
        mean = std = None
        stats_meta = {}
    else:
        feat_cols, mean, std, stats_meta = stats
        if isinstance(stats_meta, dict):
            if args.window == 64 and stats_meta.get("window") is not None:
                args.window = int(stats_meta["window"])
            if args.hop == 16 and stats_meta.get("hop") is not None:
                args.hop = int(stats_meta["hop"])
    model = _load_model(args.ckpt, device)

    # Determine frame rate (fps) for time axis
    # Priority: stats.fps -> (sr / feat_hop)
    stats_fps = stats_meta.get("fps") if isinstance(stats_meta, dict) else None
    fps = stats_fps if stats_fps is not None else (float(args.sr) / float(args.feat_hop))
    step_sec = 1.0 / float(fps)

    # ------------------ CSV mode ------------------
    if args.feat_csv is not None:
        if args.out_mid is None and args.out_dir is None:
            # default single-file output
            args.out_mid = Path("outputs/pedal_pred.mid")
        df = read_feature_csv(args.feat_csv)
        y_true = (
            df["pedal_state"].to_numpy(dtype="uint8", copy=False)
            if "pedal_state" in df.columns
            else None
        )
        prob = predict_per_frame(
            df,
            feat_cols=feat_cols,
            mean=mean,
            std=std,
            model=model,
            window=args.window,
            hop=args.hop,
            device=device,
            batch=args.batch,
            num_workers=_nw,
            strict=args.strict_stats,
        )
        prob_raw = prob.copy()
        # smoothing
        if args.smooth_sigma and args.smooth_sigma > 0:
            sig = float(args.smooth_sigma)
            rad = max(1, int(round(3 * sig)))
            xk = np.arange(-rad, rad + 1, dtype=np.float32)
            k = np.exp(-0.5 * (xk / sig) ** 2)
            k /= k.sum()
            prob = np.convolve(prob, k, mode="same")
        # hysteresis
        if args.on_th is not None and args.off_th is not None:
            on_thr, off_thr = float(args.on_th), float(args.off_th)
        else:
            on_thr, off_thr = hysteresis_threshold(
                prob,
                k_std=args.k_std,
                min_margin=args.min_margin,
                off_margin=args.off_margin,
                hyst_delta=args.hyst_delta,
                eps_on=args.eps_on,
            )
        if args.target_on_ratio is not None:
            r = float(max(0.0, min(1.0, args.target_on_ratio)))
            qthr = float(np.quantile(prob, 1.0 - r))
            on_thr = qthr
            off_thr = on_thr - args.hyst_delta
            if not np.isfinite(on_thr):
                on_thr = float(np.nanmedian(prob))
                off_thr = on_thr - args.hyst_delta
        on = postprocess_on(
            prob,
            on_thr=on_thr,
            off_thr=off_thr,
            fps=fps,
            min_on_sec=args.min_on_sec,
            min_hold_sec=args.min_hold_sec,
            eps_on=args.eps_on,
            off_consec_sec=args.off_consec_sec,
        )
        if args.dump_prob:
            dump = {"prob": prob_raw.astype("float32"), "fps": fps, "window": int(args.window), "hop": int(args.hop)}
            if y_true is not None:
                dump["y_true"] = y_true.astype("uint8")
            args.dump_prob.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.dump_prob, **dump)
        if args.compute_f1 and y_true is not None:
            metrics = compute_metrics(y_true.astype(int), prob_raw, on)
            print(metrics)
        out_mid = args.out_mid or (args.out_dir / (args.feat_csv.stem + ".pedal.mid"))
        n_cc = write_cc64(out_mid, on, step_sec)
        print({
            "mode": "csv",
            "file": str(args.feat_csv),
            "out": str(out_mid),
            "frames": int(len(on)),
            "on_ratio": float(on.mean()),
            "on_thr": float(on_thr),
            "off_thr": float(off_thr),
            "cc": int(n_cc),
        })
        if args.debug_json:
            dbg = {
                "frames": int(len(on)),
                "on_ratio": float(on.mean()),
                "on_thr": float(on_thr),
                "off_thr": float(off_thr),
                "prob_median": float(np.median(prob)),
                "prob_std": float(np.std(prob)),
                "cc": int(n_cc),
                "fps": float(fps),
                "window": int(args.window),
                "hop": int(args.hop),
            }
            args.debug_json.parent.mkdir(parents=True, exist_ok=True)
            args.debug_json.write_text(json.dumps(dbg, indent=2))
        return 0

    # ------------------ MIDI mode ------------------
    if extract_from_midi is None:
        raise SystemExit("utilities.pedal_frames.extract_from_midi not available; cannot run MIDI mode. Install project extras or use --feat-csv.")

    paths = list(iter_midis(args.inp))
    if not paths:
        raise SystemExit(f"no MIDI found under {args.inp}")
    multi = len(paths) > 1 or args.inp.is_dir()
    if multi and not args.out_dir:
        raise SystemExit("--out-dir required for directory mode")
    if args.debug_json and multi:
        args.debug_json.mkdir(parents=True, exist_ok=True)

    for p in paths:
        df = extract_from_midi(p, sr=args.sr, hop_length=args.feat_hop)
        prob = predict_per_frame(
            df,
            feat_cols=feat_cols,
            mean=mean,
            std=std,
            model=model,
            window=args.window,
            hop=args.hop,
            device=device,
            batch=args.batch,
            num_workers=_nw,
            strict=args.strict_stats,
        )
        # optional smoothing before thresholding
        if args.smooth_sigma and args.smooth_sigma > 0:
            sig = float(args.smooth_sigma)
            rad = max(1, int(round(3 * sig)))
            xk = np.arange(-rad, rad + 1, dtype=np.float32)
            k = np.exp(-0.5 * (xk / sig) ** 2)
            k /= k.sum()
            prob = np.convolve(prob, k, mode="same")
        if args.on_th is not None and args.off_th is not None:
            on_thr, off_thr = float(args.on_th), float(args.off_th)
        else:
            on_thr, off_thr = hysteresis_threshold(
                prob,
                k_std=args.k_std,
                min_margin=args.min_margin,
                off_margin=args.off_margin,
                hyst_delta=args.hyst_delta,
                eps_on=args.eps_on,
            )
        if args.target_on_ratio is not None:
            r = float(max(0.0, min(1.0, args.target_on_ratio)))
            qthr = float(np.quantile(prob, 1.0 - r))
            on_thr = qthr
            off_thr = on_thr - args.hyst_delta
            if not np.isfinite(on_thr):
                on_thr = float(np.nanmedian(prob))
                off_thr = on_thr - args.hyst_delta
        on = postprocess_on(
            prob,
            on_thr=on_thr,
            off_thr=off_thr,
            fps=fps,
            min_on_sec=args.min_on_sec,
            min_hold_sec=args.min_hold_sec,
            eps_on=args.eps_on,
            off_consec_sec=args.off_consec_sec,
        )
        out_mid = args.out_mid if (not multi and args.out_mid) else (args.out_dir / (p.stem + ".pedal.mid"))
        n_cc = write_cc64(out_mid, on, step_sec)
        print(f"[{p.name}] wrote {out_mid} cc={n_cc} frames={len(on)} on_ratio={on.mean():.6f} on_thr={on_thr:.6f} off_thr={off_thr:.6f}")
        if args.debug_json:
            dbg_path = args.debug_json / (p.stem + ".json") if multi else args.debug_json
            if not multi:
                dbg_path.parent.mkdir(parents=True, exist_ok=True)
            dbg = {
                "frames": int(len(on)),
                "on_ratio": float(on.mean()),
                "on_thr": float(on_thr),
                "off_thr": float(off_thr),
                "prob_median": float(np.median(prob)),
                "prob_std": float(np.std(prob)),
                "cc": int(n_cc),
                "fps": float(fps),
                "window": int(args.window),
                "hop": int(args.hop),
            }
            dbg_path.write_text(json.dumps(dbg, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
