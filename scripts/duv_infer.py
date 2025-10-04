#!/usr/bin/env python3
"""
DUV Inference — Humanize velocity & duration on existing MIDI without altering pitches or onsets.

Usage (examples)
-----------------
# Basic (overwrite velocities & durations in-place)
python scripts/duv_infer.py IN.mid -m checkpoints/duv_bass.ckpt -s checkpoints/duv_bass.scaler.json -o OUT.mid

# Blend model output with original (50%) and preview as CSV without writing MIDI
python scripts/duv_infer.py IN.mid -m duv.ckpt -s scaler.json --intensity 0.5 --dry-run > preview.csv

# Process only instruments matching /Bass|DANDY/ (case-insensitive), exclude drums
python scripts/duv_infer.py IN.mid -m duv_bass.ckpt -s scaler.json -i "(?i)bass|dandy" -x "(?i)drum|perc" -o OUT.mid

Design Notes
------------
- Onset times and pitches are preserved exactly. Only velocity (0..127) and duration are adjusted.
- Duration adjustments are done in *beats* space to be tempo-robust, then mapped back to seconds.
- "Intensity" (0..1.5) blends original vs model prediction. 0=original, 1=full model, >1 exaggerates within safe clamps.
- Safe clamps keep durations positive and prevent zero/negative-length notes. Velocities are clipped to [1,127].
- If --dry-run is set, no MIDI is written; a CSV preview is printed to stdout.
- Model loader supports (in this order): TorchScript .pt/.ckpt, then pickle/sklearn-like with .predict().

Dependencies
------------
- pretty_midi (required)
- numpy (required)
- torch (optional, only if TorchScript model is used)
- PyYAML/json (for scaler and optional config metadata)

This script is intentionally self-contained and makes minimal assumptions about your model.
If your repository exposes a custom feature extractor, wire it in at FEATURE EXTRACTION POINT below.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import pretty_midi as pm

try:
    import torch  # type: ignore

    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

EPS = 1e-9

# ----------------------------
# Utilities
# ----------------------------


def eprint(*a: Any, **k: Any) -> None:
    print(*a, file=sys.stderr, **k)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Scaler:
    """Simple feature scaler.

    Attributes
    ----------
    mean : np.ndarray
        Feature-wise means.
    std : np.ndarray
        Feature-wise standard deviations (zeros are treated as 1.0 to avoid NaN).
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_json(cls, path: str) -> "Scaler":
        obj = load_json(path)
        mean = np.asarray(obj.get("mean"), dtype=np.float32)
        std = np.asarray(obj.get("std"), dtype=np.float32)
        std = np.where(np.abs(std) < EPS, 1.0, std)
        return cls(mean, std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        # broadcast-safe
        return (X - self.mean) / self.std


class DUVModel:
    """Thin wrapper to unify TorchScript and sklearn-like models.

    Expected behavior: predict(X) -> ndarray of shape (N, 2)
    where columns are [delta_vel, dur_scale] or [vel, dur_beats] depending on your model.

    This implementation uses a *convention*:
      - If model outputs 2 columns:
          y[:,0] = target velocity (0..127) *or* delta-velocity; auto-detected via --mode
          y[:,1] = target duration in beats *or* multiplicative scale; auto-detected via --mode

    You choose with --mode in {"absolute", "delta"}.
    """

    def __init__(self, fn, kind: str):
        self.fn = fn
        self.kind = kind  # "torch" or "pickle"

    @classmethod
    def load(cls, path: str) -> "DUVModel":
        ext = os.path.splitext(path)[1].lower()
        if _HAVE_TORCH and ext in {".pt", ".ckpt", ".pth"}:
            # Try TorchScript first
            try:
                eprint(f"[duv] Loading TorchScript model: {path}")
                model = torch.jit.load(path, map_location="cpu")
                model.eval()

                def _predict(X: np.ndarray) -> np.ndarray:
                    with torch.no_grad():
                        tX = torch.from_numpy(np.asarray(X, dtype=np.float32))
                        y = model(tX)
                        y = y.detach().cpu().numpy()
                    return np.asarray(y, dtype=np.float32)

                return cls(_predict, kind="torch")
            except RuntimeError as e:
                # If TorchScript fails, try Lightning checkpoint format
                if "constants.pkl" in str(e) or "PytorchStreamReader" in str(e):
                    eprint(f"[duv] TorchScript failed, trying Lightning checkpoint format: {path}")
                    ckpt = torch.load(path, map_location="cpu")
                    # Assume this is a Lightning checkpoint with 'model' key
                    from models.phrase_transformer import PhraseTransformer

                    # Extract metadata
                    meta = ckpt.get("meta", {})
                    state_dict = ckpt.get("model", {})

                    # Create model from metadata
                    model = PhraseTransformer(
                        d_model=meta.get("d_model", 512),
                        nhead=meta.get("n_heads", 8),
                        num_layers=meta.get("n_layers", 4),
                        ff_dim=meta.get("ff_dim", 1024),  # Will be inferred
                        dropout=0.1,
                        max_len=meta.get("max_len", 128),
                        use_bar_beat=meta.get("use_bar_beat", True),
                        use_harmony=meta.get("use_harmony", False),
                        duv_mode=meta.get("duv_mode", "both"),
                        vel_bins=meta.get("vel_bins", 16),
                        dur_bins=meta.get("dur_bins", 16),
                    )
                    # Load state dict (may have mismatches for legacy checkpoints)
                    # Skip problematic keys for old checkpoints
                    filtered_state_dict = {
                        k: v for k, v in state_dict.items() if not k.startswith("_proj.")
                    }
                    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
                    eprint(
                        f"[duv] Loaded checkpoint with {len(missing)} missing keys, {len(unexpected)} unexpected keys"
                    )
                    if missing:
                        eprint(f"[duv] Missing keys: {missing[:10]}")
                    model.eval()

                    def _predict(X: np.ndarray) -> np.ndarray:
                        with torch.no_grad():
                            # X shape: (N, feature_dim)
                            # Need to convert to dict format for PhraseTransformer
                            N = X.shape[0]
                            feats = {
                                "pitch": torch.from_numpy(X[:, 0].astype(np.int64)).unsqueeze(
                                    0
                                ),  # (1, N)
                                "velocity": torch.from_numpy(X[:, 1].astype(np.float32)).unsqueeze(
                                    0
                                ),
                                "duration": torch.from_numpy(X[:, 2].astype(np.float32)).unsqueeze(
                                    0
                                ),
                                "pos": torch.from_numpy(X[:, 3].astype(np.int64)).unsqueeze(0),
                                "bar": (
                                    torch.from_numpy(X[:, 4].astype(np.int64)).unsqueeze(0)
                                    if X.shape[1] > 4
                                    else torch.zeros(1, N, dtype=torch.long)
                                ),
                                "velocity_bucket": torch.zeros(1, N, dtype=torch.long),
                                "duration_bucket": torch.zeros(1, N, dtype=torch.long),
                            }

                            out = model(feats)
                            # out is dict with 'vel_reg' and 'dur_reg', shape (1, N)
                            # Stack into (N, 2) array: [velocity, duration]
                            vel = out["vel_reg"].squeeze(0).squeeze(-1).cpu().numpy()  # (N,)
                            dur = out["dur_reg"].squeeze(0).squeeze(-1).cpu().numpy()  # (N,)
                            y = np.stack([vel, dur], axis=1)  # (N, 2)
                        return np.asarray(y, dtype=np.float32)

                    return cls(_predict, kind="torch")
                else:
                    raise
        else:
            eprint(f"[duv] Loading pickle-like model: {path}")
            import pickle  # lazy import

            with open(path, "rb") as f:
                mdl = pickle.load(f)
            if not hasattr(mdl, "predict"):
                raise ValueError("Loaded object has no .predict()")
            return cls(lambda X: np.asarray(mdl.predict(X), dtype=np.float32), kind="pickle")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.fn(X)


# ----------------------------
# Beat mapping helpers
# ----------------------------


def compute_beats(midi: pm.PrettyMIDI) -> Tuple[np.ndarray, np.ndarray]:
    """Return (beat_times, beat_numbers).

    beat_times: increasing seconds of each beat boundary
    beat_numbers: float beat index (0,1,2,...) matching beat_times
    """
    beat_times = midi.get_beats()
    if beat_times.size == 0:
        # Synthesize from first tempo if missing (rare). Assume 120 bpm 4/4 over file length.
        # Estimate length from last note end.
        last_end = 0.0
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for n in inst.notes:
                last_end = max(last_end, n.end)
        bpm = 120.0
        if last_end <= 0.0:
            return np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64)
        beat_dur = 60.0 / bpm
        n_beats = int(max(1, math.ceil(last_end / beat_dur)))
        beat_times = np.arange(0, n_beats + 1, dtype=np.float64) * beat_dur
    beat_nums = np.arange(len(beat_times), dtype=np.float64)
    return beat_times, beat_nums


def time_to_beat(t: float, beat_times: np.ndarray, beat_nums: np.ndarray) -> float:
    idx = np.searchsorted(beat_times, t, side="right") - 1
    idx = max(0, min(idx, len(beat_times) - 2))
    t0, t1 = beat_times[idx], beat_times[idx + 1]
    b0, b1 = beat_nums[idx], beat_nums[idx + 1]
    if t1 <= t0 + EPS:
        return float(b0)
    alpha = (t - t0) / (t1 - t0)
    return float(b0 + alpha * (b1 - b0))


def beat_to_time(b: float, beat_times: np.ndarray, beat_nums: np.ndarray) -> float:
    idx = np.searchsorted(beat_nums, b, side="right") - 1
    idx = max(0, min(idx, len(beat_nums) - 2))
    b0, b1 = beat_nums[idx], beat_nums[idx + 1]
    t0, t1 = beat_times[idx], beat_times[idx + 1]
    if b1 <= b0 + EPS:
        return float(t0)
    alpha = (b - b0) / (b1 - b0)
    return float(t0 + alpha * (t1 - t0))


# ----------------------------
# Feature extraction (generic)
# ----------------------------


def extract_features_for_track(
    notes: List[pm.Note], beat_times: np.ndarray, beat_nums: np.ndarray
) -> np.ndarray:
    """Compute per-note features.

    Features per note (column order):
        0: beat_pos           (note-on in beats)
        1: beat_frac          (fractional position within current beat [0,1))
        2: pitch              (MIDI note number)
        3: dur_beats          (note duration in beats)
        4: prev_ioi_beats     (prev onset gap in beats; 0 for first note)
        5: next_ioi_beats     (next onset gap in beats; 0 for last note)
        6: is_downbeat        (1 if near whole beat boundary)
        7: vel_norm           (original velocity / 127.0)

    NOTE: Replace or extend this at the FEATURE EXTRACTION POINT if your model expects different inputs.
    """
    if not notes:
        return np.zeros((0, 8), dtype=np.float32)

    onsets = np.array([n.start for n in notes], dtype=np.float64)
    beats = np.array([time_to_beat(t, beat_times, beat_nums) for t in onsets], dtype=np.float64)
    beat_pos = beats

    beat_frac = np.mod(beat_pos, 1.0)

    durs_beats = np.array(
        [
            max(
                EPS,
                time_to_beat(n.end, beat_times, beat_nums)
                - time_to_beat(n.start, beat_times, beat_nums),
            )
            for n in notes
        ],
        dtype=np.float64,
    )

    prev_ioi = np.zeros(len(notes), dtype=np.float64)
    next_ioi = np.zeros(len(notes), dtype=np.float64)
    if len(notes) > 1:
        prev_ioi[1:] = beat_pos[1:] - beat_pos[:-1]
        next_ioi[:-1] = beat_pos[1:] - beat_pos[:-1]

    is_downbeat = (beat_frac < 1e-3).astype(np.float64)
    vel_norm = np.array([n.velocity / 127.0 for n in notes], dtype=np.float64)

    X = np.stack(
        [
            beat_pos,
            beat_frac,
            np.array([n.pitch for n in notes], dtype=np.float64),
            durs_beats,
            prev_ioi,
            next_ioi,
            is_downbeat,
            vel_norm,
        ],
        axis=1,
    )
    return X.astype(np.float32)


# ----------------------------
# Core application of predictions
# ----------------------------


def apply_predictions(
    notes: List[pm.Note],
    X: np.ndarray,
    y: np.ndarray,
    *,
    mode: str,
    intensity: float,
    beat_times: np.ndarray,
    beat_nums: np.ndarray,
    clip_vel: Tuple[int, int] = (1, 127),
    min_dur_beats: float = 1.0 / 64.0,
) -> List[Tuple[int, float]]:
    """Return list of (new_velocity, new_dur_beats) for each note without mutating in place.

    mode: "absolute" or "delta"
      - absolute: y[:,0] is target velocity (0..127), y[:,1] is target duration in beats.
      - delta   : y[:,0] is delta velocity (add), y[:,1] is multiplicative duration scale.
    """
    assert y.shape[0] == len(notes), (y.shape, len(notes))
    out_vel: List[int] = []
    out_dur: List[float] = []

    orig_vel = np.array([n.velocity for n in notes], dtype=np.float32)
    # derive original durations in beats
    orig_dur_beats = np.array(
        [
            max(
                EPS,
                time_to_beat(n.end, beat_times, beat_nums)
                - time_to_beat(n.start, beat_times, beat_nums),
            )
            for n in notes
        ],
        dtype=np.float32,
    )

    if mode == "absolute":
        target_vel = y[:, 0]
        target_dur_beats = np.maximum(min_dur_beats, y[:, 1])
        new_vel = (1 - intensity) * orig_vel + intensity * target_vel
        new_dur_beats = (1 - intensity) * orig_dur_beats + intensity * target_dur_beats
    elif mode == "delta":
        delta_vel = y[:, 0]
        scale_dur = np.maximum(0.0, y[:, 1])  # negative scales are floored at 0
        target_vel = orig_vel + delta_vel
        target_dur_beats = np.maximum(min_dur_beats, orig_dur_beats * scale_dur)
        new_vel = (1 - intensity) * orig_vel + intensity * target_vel
        new_dur_beats = (1 - intensity) * orig_dur_beats + intensity * target_dur_beats
    else:
        raise ValueError("mode must be 'absolute' or 'delta'")

    new_vel = np.clip(new_vel, clip_vel[0], clip_vel[1])
    # duration safety: keep at least min_dur_beats
    new_dur_beats = np.maximum(new_dur_beats, min_dur_beats)

    for v, d in zip(new_vel, new_dur_beats):
        out_vel.append(int(round(float(v))))
        out_dur.append(float(d))
    return list(zip(out_vel, out_dur))


# ----------------------------
# High-level processing per instrument
# ----------------------------


def process_instrument(
    inst: pm.Instrument,
    model: DUVModel,
    scaler: Optional[Scaler],
    *,
    mode: str,
    intensity: float,
    beat_times: np.ndarray,
    beat_nums: np.ndarray,
    dry_run_rows: List[str],
) -> None:
    notes = list(sorted(inst.notes, key=lambda n: (n.start, n.pitch)))
    if not notes:
        return
    X = extract_features_for_track(notes, beat_times, beat_nums)
    if scaler is not None and X.size:
        X = scaler.transform(X)
    y = model.predict(X)
    if y.ndim != 2 or y.shape[1] < 2:
        raise ValueError(f"Model returned unexpected shape {y.shape}; expected (N,2)")

    updates = apply_predictions(
        notes,
        X,
        y,
        mode=mode,
        intensity=intensity,
        beat_times=beat_times,
        beat_nums=beat_nums,
    )

    # Apply (without altering onset times or pitches)
    for n, (v, dur_beats) in zip(notes, updates):
        start_b = time_to_beat(n.start, beat_times, beat_nums)
        new_end_b = start_b + dur_beats
        new_end_t = beat_to_time(new_end_b, beat_times, beat_nums)
        # ensure strictly > start and monotonic with minimal epsilon
        if new_end_t <= n.start + EPS:
            new_end_t = n.start + 1e-4
        if dry_run_rows is not None:
            dur_orig_b = time_to_beat(n.end, beat_times, beat_nums) - start_b
            dry_run_rows.append(
                f"{json.dumps(inst.name)}\t{n.start:.6f}\t{n.end:.6f}\t{n.pitch}\t{n.velocity}\t{v}\t{dur_orig_b:.5f}\t{dur_beats:.5f}"
            )
        else:
            n.velocity = v
            n.end = new_end_t


# ----------------------------
# CLI
# ----------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply DUV model to humanize velocity & duration in a MIDI file."
    )
    p.add_argument("input", help="Input MIDI path")
    p.add_argument("-o", "--output", help="Output MIDI path (omitted if --dry-run)")
    p.add_argument(
        "-m",
        "--model",
        required=True,
        help="Model path (.pt/.ckpt TorchScript or pickle with .predict)",
    )
    p.add_argument("-s", "--scaler", help="Scaler JSON with 'mean' and 'std' arrays")
    p.add_argument(
        "--mode",
        choices=["absolute", "delta"],
        default="absolute",
        help="How to interpret model outputs: absolute targets or deltas",
    )
    p.add_argument(
        "--intensity", type=float, default=1.0, help="Blend factor (0..1.5). 1.0 = full model"
    )
    p.add_argument("--include", "-i", help="Regex: only process instruments whose name matches")
    p.add_argument("--exclude", "-x", help="Regex: skip instruments whose name matches")
    p.add_argument(
        "--include-drums", action="store_true", help="Allow processing drum tracks (off by default)"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print TSV preview instead of writing MIDI"
    )
    p.add_argument(
        "--seed", type=int, default=0, help="Random seed (if your model uses stochastic layers)"
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.intensity < 0.0:
        eprint("[duv] intensity < 0 reset to 0.0")
        args.intensity = 0.0

    if _HAVE_TORCH and args.seed is not None:
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))

    midi = pm.PrettyMIDI(args.input)
    beat_times, beat_nums = compute_beats(midi)

    model = DUVModel.load(args.model)
    scaler = Scaler.from_json(args.scaler) if args.scaler else None

    inc_re = re.compile(args.include) if args.include else None
    exc_re = re.compile(args.exclude) if args.exclude else None

    dry_rows: Optional[List[str]] = [] if args.dry_run else None

    for inst in midi.instruments:
        name = inst.name or ""
        if inst.is_drum and not args.include_drums:
            continue
        if inc_re and not inc_re.search(name):
            continue
        if exc_re and exc_re.search(name):
            continue
        process_instrument(
            inst,
            model,
            scaler,
            mode=args.mode,
            intensity=float(args.intensity),
            beat_times=beat_times,
            beat_nums=beat_nums,
            dry_run_rows=dry_rows,
        )

    if args.dry_run:
        # Header (TSV)
        print("instrument\tstart\tend\tpitch\tvel_orig\tvel_new\tdur_orig_beats\tdur_new_beats")
        assert dry_rows is not None
        for row in dry_rows:
            print(row)
        return 0

    out_path = args.output or os.path.splitext(args.input)[0] + ".duv.mid"
    midi.write(out_path)
    eprint(f"[duv] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
