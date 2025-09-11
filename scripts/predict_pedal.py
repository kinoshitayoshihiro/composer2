from __future__ import annotations

"""Predict CC64 (sustain pedal) from MIDI using a trained pedal model.

Features are extracted with the same pipeline as training (chroma + rel_release),
standardized using stats saved beside the checkpoint (<ckpt>.stats.json), then
fed to the Conv1D+BiLSTM model over sliding windows. Overlapping predictions are
averaged back to per-frame probabilities. Hysteresis + minimum lengths are used
to produce stable on/off, and CC64 events are written as a new MIDI.

Examples
  Single file:
    python -m scripts.predict_pedal \
      --in data/songs_norm/omokage.mid \
      --ckpt checkpoints/pedal.ckpt \
      --out-mid outputs/pedal_pred.mid

  Directory:
    python -m scripts.predict_pedal \
      --in data/songs_norm \
      --ckpt checkpoints/pedal.ckpt \
      --out-dir outputs/pedal_pred
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch

import pretty_midi as pm

from ml_models.pedal_model import PedalModel
from utilities.pedal_frames import extract_from_midi, SR as DEFAULT_SR, HOP_LENGTH as DEFAULT_HOP


def _load_stats(ckpt_path: Path, stats_json: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    path = stats_json if stats_json and stats_json.is_file() else ckpt_path.with_suffix(ckpt_path.suffix + ".stats.json")
    if not path.is_file():
        raise SystemExit(f"feature stats JSON not found: {path}")
    obj = json.loads(path.read_text())
    mean = np.array(obj.get("mean", []), dtype=np.float32)
    std = np.array(obj.get("std", []), dtype=np.float32)
    if mean.size == 0 or std.size == 0:
        raise SystemExit("invalid stats JSON (missing mean/std)")
    std[std < 1e-8] = 1.0
    return mean, std


def _load_model(ckpt: Path, device: torch.device) -> PedalModel:
    state = torch.load(ckpt, map_location="cpu")
    model = PedalModel()
    if isinstance(state, dict) and "state_dict" in state:
        sd = {k.removeprefix("model."): v for k, v in state["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def _make_windows(arr: torch.Tensor, win: int, hop: int) -> tuple[torch.Tensor, list[int]]:
    T = arr.shape[0]
    if T < win:
        return torch.empty(0, win, arr.shape[1], dtype=arr.dtype), []
    starts = list(range(0, T - win + 1, hop))
    out = torch.stack([arr[s : s + win] for s in starts], dim=0)
    return out, starts


def predict_per_frame(df: pd.DataFrame, mean: np.ndarray, std: np.ndarray, model: PedalModel, *, window: int, hop: int, device: torch.device, batch: int = 64) -> tuple[np.ndarray, float]:
    chroma_cols = [c for c in df.columns if c.startswith("chroma_")]
    X = df[chroma_cols + ["rel_release"]].values.astype("float32")
    T, C = X.shape
    if C != mean.size:
        raise SystemExit(f"feature dimension mismatch: got {C}, expected {mean.size}")
    X = (X - mean) / std
    xt = torch.from_numpy(X)
    win, starts = _make_windows(xt, window, hop)
    if len(starts) == 0:
        # too short
        return np.zeros(T, dtype=np.float32), hop / DEFAULT_SR
    bs = max(1, int(batch))
    prob_sum = np.zeros(T, dtype=np.float64)
    logit_sum = np.zeros(T, dtype=np.float64)
    cnt = np.zeros(T, dtype=np.int32)
    with torch.no_grad():
        for i in range(0, win.shape[0], bs):
            wb = win[i : i + bs].to(device)
            logits = model(wb)  # (B, win)
            pb = torch.sigmoid(logits).cpu().numpy()
            lg = logits.cpu().numpy()
            for k, s in enumerate(starts[i : i + bs]):
                e = s + window
                prob_sum[s:e] += pb[k]
                logit_sum[s:e] += lg[k]
                cnt[s:e] += 1
    cnt[cnt == 0] = 1
    prob = (prob_sum / cnt).astype(np.float32)
    step_sec = float(DEFAULT_HOP / DEFAULT_SR)
    # salvage if probabilities collapse
    if float(prob.max() - prob.min()) < 1e-3:
        z = logit_sum / cnt
        z = (z - z.mean()) / (z.std() + 1e-6)
        prob = 1.0 / (1.0 + np.exp(-z)).astype(np.float32)
    return prob, step_sec


def hysteresis_threshold(prob: np.ndarray, *, k_std: float, min_margin: float, off_margin: float, hyst_delta: float, eps_on: float) -> tuple[float, float]:
    m = float(np.median(prob))
    s = float(prob.std())
    thr = max(m + k_std * s, m + min_margin)
    on_thr = float(np.clip(thr, 0.0, 1.0))
    off_thr = float(max(on_thr - hyst_delta, m + off_margin))
    return on_thr, off_thr


def postprocess_on(prob: np.ndarray, *, on_thr: float, off_thr: float, fps: float, min_on_sec: float, min_hold_sec: float, eps_on: float, off_consec_sec: float = 0.0) -> np.ndarray:
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


def write_cc64(out_mid: Path, on: np.ndarray, step_sec: float) -> int:
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    midi.instruments.append(inst)
    cc, cur = [], None
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="input MIDI file or directory")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-mid", type=Path, help="output MIDI path for single-file mode")
    ap.add_argument("--out-dir", type=Path, help="output directory for directory mode")
    ap.add_argument("--stats-json", type=Path, help="feature stats JSON (defaults to <ckpt>.stats.json)")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--hop", type=int, default=16)
    ap.add_argument("--sr", type=int, default=DEFAULT_SR)
    ap.add_argument("--feat-hop", type=int, default=DEFAULT_HOP, help="feature extraction hop length for librosa")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    # thresholding / hysteresis
    ap.add_argument("--on-th", type=float, help="fixed ON threshold; if set, overrides auto")
    ap.add_argument("--off-th", type=float, help="fixed OFF threshold; if set, overrides auto")
    ap.add_argument("--k-std", type=float, default=2.0)
    ap.add_argument("--min-margin", type=float, default=0.002)
    ap.add_argument("--off-margin", type=float, default=0.0005)
    ap.add_argument("--hyst-delta", type=float, default=0.002)
    ap.add_argument("--min-on-sec", type=float, default=0.05)
    ap.add_argument("--min-hold-sec", type=float, default=0.05)
    ap.add_argument("--eps-on", type=float, default=1e-6)
    # optional smoothing / debouncing
    ap.add_argument("--smooth-sigma", type=float, default=0.0, help="Gaussian smoothing sigma (in frames) before hysteresis; 0 to disable")
    ap.add_argument("--off-consec-sec", type=float, default=0.0, help="require this many consecutive seconds below OFF threshold before turning OFF")
    args = ap.parse_args(argv)

    # device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    mean, std = _load_stats(args.ckpt, args.stats_json)
    model = _load_model(args.ckpt, device)

    paths = list(iter_midis(args.inp))
    if not paths:
        raise SystemExit(f"no MIDI found under {args.inp}")
    multi = len(paths) > 1 or args.inp.is_dir()
    if multi and not args.out_dir:
        raise SystemExit("--out-dir required for directory mode")

    for p in paths:
        df = extract_from_midi(p, sr=args.sr, hop_length=args.feat_hop)
        prob, step_sec = predict_per_frame(df, mean, std, model, window=args.window, hop=args.hop, device=device, batch=args.batch)
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
            on_thr, off_thr = hysteresis_threshold(prob, k_std=args.k_std, min_margin=args.min_margin, off_margin=args.off_margin, hyst_delta=args.hyst_delta, eps_on=args.eps_on)
        fps = 1.0 / step_sec
        on = postprocess_on(prob, on_thr=on_thr, off_thr=off_thr, fps=fps, min_on_sec=args.min_on_sec, min_hold_sec=args.min_hold_sec, eps_on=args.eps_on, off_consec_sec=args.off_consec_sec)
        out_mid = args.out_mid if (not multi and args.out_mid) else (args.out_dir / (p.stem + ".pedal.mid"))
        n_cc = write_cc64(out_mid, on, step_sec)
        print(f"[{p.name}] wrote {out_mid} cc={n_cc} frames={len(on)} on_ratio={on.mean():.6f} on_thr={on_thr:.6f} off_thr={off_thr:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
