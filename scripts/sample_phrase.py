#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phrase-level inference script.
- Loads checkpoint saved by scripts/train_phrase.py
- Rebuilds model using checkpoint meta (arch/d_model/etc.)
- Generates a phrase (e.g., bass) conditioned on optional seed
- Supports DUV: continuous regression or bucket decoding (auto)
- Saves to CSV and/or MIDI

Usage:
  python -m scripts.sample_phrase \
    --ckpt checkpoints/bass_duv_v1.ckpt \
    --out-midi out/bass_phrase.mid \
    --out-csv  out/bass_phrase.csv \
    --length 64 --temperature 1.0 --topk 0 --topp 0.0 \
    --bpm 110 --instrument-program 33  # 33=Fingered Bass (GM)
"""

from __future__ import annotations

import os
import sys
import json
import math
import argparse
import logging
import random
import csv
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover - optional
    np = None  # type: ignore
try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore
    F = None  # type: ignore

# ---------- project import (same pattern as train_phrase.py) ----------
try:
    from models.phrase_transformer import PhraseTransformer
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parent.parent
    if os.environ.get("ALLOW_LOCAL_IMPORT") == "1":
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
            logging.warning("ALLOW_LOCAL_IMPORT=1, inserted repo root into sys.path")
        try:  # pragma: no cover - import fallback
            from models.phrase_transformer import PhraseTransformer
        except ModuleNotFoundError:
            PhraseTransformer = None  # type: ignore
    else:
        PhraseTransformer = None  # type: ignore

try:
    import pretty_midi
except Exception:  # pragma: no cover - pretty_midi optional
    pretty_midi = None

try:
    from utilities.phrase_data import denorm_duv
except Exception:  # pragma: no cover - simple fallback
    def denorm_duv(vel_reg: torch.Tensor, dur_reg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vel = vel_reg.mul(127.0).round().clamp(0, 127)
        dur = torch.expm1(dur_reg)
        return vel, dur


# instrument pitch range presets
PITCH_PRESETS = {
    "bass": (28, 52),
    "piano": (21, 108),
    "strings": (40, 84),
}


# ----------------------- helpers -----------------------
def load_checkpoint(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        return obj
    raise SystemExit(
        f"Unexpected checkpoint format: {list(obj.keys()) if isinstance(obj, dict) else type(obj)}"
    )


def build_model_from_meta(meta: dict) -> PhraseTransformer:
    d_model = int(meta.get("d_model", 384))
    n_layers = int(meta.get("n_layers", 6))
    n_heads = int(meta.get("n_heads", 8))
    max_len = int(meta.get("max_len", 512))
    duv_mode = meta.get("duv_mode", "reg")
    dur_bins = int(meta.get("dur_bins", 16))
    vel_bins = int(meta.get("vel_bins", 8))
    vocab_pitch = int(meta.get("vocab_pitch", 128))

    model = PhraseTransformer(
        d_model=d_model,
        max_len=max_len,
        duv_mode=duv_mode,
        dur_bins=dur_bins,
        vel_bins=vel_bins,
        nhead=n_heads,
        num_layers=n_layers,
        pitch_vocab_size=vocab_pitch,
    )
    return model


def decode_duv(
    out_dict: dict, vel_mode: str, dur_mode: str, meta: dict, dur_max_beats: float
) -> tuple[float, float]:
    """Return real velocity and duration from model outputs."""
    if vel_mode == "reg" and "vel_reg" in out_dict:
        vel = float(
            denorm_duv(out_dict["vel_reg"], torch.zeros_like(out_dict["vel_reg"]))[0].item()
        )
    elif "vel_cls" in out_dict:
        vel_idx = int(F.softmax(out_dict["vel_cls"], dim=-1).multinomial(1).item())
        vel_bins = int(meta.get("vel_bins", 8))
        vel = (vel_idx + 0.5) * (127.0 / vel_bins)
    else:
        vel = 64.0

    if dur_mode == "reg" and "dur_reg" in out_dict:
        dur = float(
            denorm_duv(torch.zeros_like(out_dict["dur_reg"]), out_dict["dur_reg"])[1].item()
        )
    elif "dur_cls" in out_dict:
        dur_idx = int(F.softmax(out_dict["dur_cls"], dim=-1).multinomial(1).item())
        dur_bins = int(meta.get("dur_bins", 16))
        dur = 4.0 * (dur_idx + 1) / dur_bins
    else:
        dur = 0.25

    vel = float(max(1.0, min(127.0, vel)))
    dur = float(min(dur_max_beats, max(1e-3, dur)))
    return vel, dur


def sample_logits(logits: torch.Tensor, temperature: float, topk: int, topp: float) -> int:
    """Sample an index from *logits*.

    ``temperature`` scales logits by ``1/T``. If both ``topk`` and ``topp``
    are specified, top-k filtering is applied first, then nucleus (top-p)
    filtering. Probabilities are always renormalized after filtering.
    """

    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / float(temperature)
    probs = F.softmax(logits, dim=-1)

    if (not topk or topk <= 0) and (not topp or topp <= 0):
        topk = 1

    if topk and topk > 0:
        k = min(topk, probs.size(-1))
        topk_probs, topk_idx = torch.topk(probs, k)
        keep = torch.zeros_like(probs)
        keep.scatter_(0, topk_idx, topk_probs)
        probs = keep
        probs = probs / probs.sum()

    if topp and topp > 0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        mask = cdf <= topp
        mask[0] = True
        filtered = sorted_probs * mask
        filtered = filtered / filtered.sum()
        idx = sorted_idx[torch.multinomial(filtered, 1)]
        return int(idx.item())

    idx = torch.multinomial(probs, 1)
    return int(idx.item())


def events_to_prettymidi(
    events: list[dict],
    bpm: float,
    *,
    gm_program: int = 33,
    is_drum: bool = False,
    humanize_timing: float = 0.0,
    humanize_vel: float = 0.0,
) -> "pretty_midi.PrettyMIDI|None":
    if pretty_midi is None:
        return None
    try:
        pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    except Exception:  # pragma: no cover - fallback path
        pm = pretty_midi.PrettyMIDI()
        if np is not None:
            pm._tempo_changes = np.array([0.0])
            pm._tempos = np.array([float(bpm)])
    inst = pretty_midi.Instrument(program=int(gm_program), name="phrase", is_drum=is_drum)
    t = 0.0
    sec_per_beat = 60.0 / float(bpm)
    for ev in events:
        pitch = int(ev["pitch"])
        vel = float(ev.get("velocity", 64.0))
        dur_beats = float(ev.get("duration_beats", 0.25))
        dur_beats = max(dur_beats, 1e-3)
        if humanize_vel > 0.0:
            vel = max(1, min(127, vel + random.gauss(0.0, humanize_vel)))
        start = t
        if humanize_timing > 0.0:
            start += random.gauss(0.0, humanize_timing)
        end = start + dur_beats * sec_per_beat
        inst.notes.append(
            pretty_midi.Note(velocity=int(max(1, min(127, vel))), pitch=pitch, start=start, end=end)
        )
        t += dur_beats * sec_per_beat
    pm.instruments.append(inst)
    return pm


# ----------------------- main inference -----------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        epilog="If project modules fail to import, run 'pip install -e .' or set ALLOW_LOCAL_IMPORT=1"
    )
    ap.add_argument(
        "--ckpt",
        required=True,
        type=Path,
        help="checkpoint from train_phrase.py (install package or set ALLOW_LOCAL_IMPORT=1)",
    )
    ap.add_argument("--length", type=int, default=64, help="number of steps to generate")
    ap.add_argument("--temperature", type=float, default=None,
                    help="deprecated; use --temperature-start/--temperature-end")
    ap.add_argument("--temperature-start", type=float, default=1.0)
    ap.add_argument("--temperature-end", type=float, default=1.0)
    ap.add_argument(
        "--topk",
        type=int,
        default=0,
        help="top-k sampling (0 disables; if both top-k and top-p are 0, falls back to top-1)",
    )
    ap.add_argument(
        "--topp",
        type=float,
        default=0.0,
        help="top-p nucleus sampling (0 disables; see --topk)",
    )
    ap.add_argument(
        "--seed-json",
        type=str,
        default="",
        help="JSON list of seed events [{'pitch':..,'velocity':..,'duration_beats':..},..]",
    )
    ap.add_argument("--bpm", type=float, default=110.0)
    ap.add_argument("--instrument-program", type=int, default=33)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seed-csv", type=Path, default=None)
    ap.add_argument("--is-drum", action="store_true")
    ap.add_argument("--humanize-timing", type=float, default=0.0)
    ap.add_argument("--humanize-vel", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--duv-decode", choices=["auto", "reg", "bucket"], default="auto")
    ap.add_argument("--dur-decode", choices=["reg", "bucket"], default=None)
    ap.add_argument("--vel-decode", choices=["reg", "bucket"], default=None)
    ap.add_argument("--bars", type=int, default=None, help="stop after N bars (approx 4 beats each)")
    ap.add_argument(
        "--instrument-name",
        choices=sorted(PITCH_PRESETS),
        help="preset pitch range (overridden by --pitch-min/--pitch-max)",
    )
    ap.add_argument(
        "--pitch-min",
        type=int,
        default=0,
        help="minimum MIDI pitch; overrides preset",
    )
    ap.add_argument(
        "--pitch-max",
        type=int,
        default=127,
        help="maximum MIDI pitch; overrides preset",
    )
    ap.add_argument(
        "--dur-max-beats",
        type=float,
        default=16.0,
        help="clamp decoded duration to this many beats",
    )
    ap.add_argument("--out-midi", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    temp_start = (
        args.temperature if args.temperature is not None else args.temperature_start
    )
    temp_end = (
        args.temperature if args.temperature is not None else args.temperature_end
    )

    if args.instrument_program < 0 or args.instrument_program > 127:
        raise SystemExit("--instrument-program must be in [0,127]")

    if args.instrument_name:
        args.pitch_min, args.pitch_max = PITCH_PRESETS[args.instrument_name]

    if args.pitch_min > args.pitch_max:
        raise SystemExit("--pitch-min must be <= --pitch-max")
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if np is not None:
            np.random.seed(args.seed)

    if torch is None or F is None:
        raise SystemExit("torch is required for sampling")
    if PhraseTransformer is None:
        raise SystemExit(
            "Could not import project modules. Run 'pip install -e .' or set ALLOW_LOCAL_IMPORT=1"
        )

    state = load_checkpoint(args.ckpt)
    meta = state.get("meta", {})
    duv_mode = meta.get("duv_mode", "reg")
    eff_duv = args.duv_decode if args.duv_decode != "auto" else duv_mode
    vel_mode = args.vel_decode or ("reg" if eff_duv == "reg" else "bucket")
    dur_mode = args.dur_decode or ("reg" if eff_duv == "reg" else "bucket")

    model = build_model_from_meta(meta)
    model.load_state_dict(state["model"])
    model.eval().to(args.device)

    seq: list[dict] = []
    if args.seed_json:
        try:
            seq = json.loads(args.seed_json)
        except Exception as e:  # pragma: no cover
            raise SystemExit(f"Invalid --seed-json: {e}")
    if args.seed_csv:
        if not args.seed_csv.is_file():
            raise SystemExit(f"seed CSV not found: {args.seed_csv}")
        with args.seed_csv.open() as f:
            reader = csv.DictReader(f)
            required = {"pitch", "velocity", "duration_beats"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise SystemExit(f"seed CSV missing columns {required}")
            for row in reader:
                seq.append(
                    {
                        "pitch": int(row["pitch"]),
                        "velocity": float(row["velocity"]),
                        "duration_beats": float(row["duration_beats"]),
                    }
                )
    if seq and len(seq) > model.max_len:
        raise SystemExit("seed longer than model max_len")
    if args.seed_csv and not seq:
        raise SystemExit("seed CSV had no rows")

    if not hasattr(model, "encode_seed") or not hasattr(model, "step"):
        print("NOTE: Adapt 'encode_seed' and 'step' calls to your model API.", file=sys.stderr)

    state_enc = model.encode_seed(seq) if hasattr(model, "encode_seed") else None

    out_events: list[dict] = []
    total_beats = 0.0
    with torch.no_grad():
        for step in range(args.length):
            out = model.step(state_enc) if hasattr(model, "step") else {}
            alpha = step / max(args.length - 1, 1)
            temp = temp_start + (temp_end - temp_start) * alpha
            if "pitch_logits" in out:
                pitch = sample_logits(
                    out["pitch_logits"].squeeze(0), temp, args.topk, args.topp
                )
            else:
                pitch = random.randint(36, 51)
            pitch = int(max(args.pitch_min, min(args.pitch_max, pitch)))

            vel, dur = decode_duv(out, vel_mode, dur_mode, meta, args.dur_max_beats)

            ev = {"pitch": pitch, "velocity": vel, "duration_beats": dur}
            out_events.append(ev)
            total_beats += dur

            if hasattr(model, "update_state"):
                state_enc = model.update_state(state_enc, ev)

            if args.bars is not None and total_beats >= 4 * args.bars:
                break

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["pitch", "velocity", "duration_beats"])
            w.writeheader()
            w.writerows(out_events)

    if args.out_midi:
        if pretty_midi is not None:
            args.out_midi.parent.mkdir(parents=True, exist_ok=True)
            pm = events_to_prettymidi(
                out_events,
                args.bpm,
                gm_program=args.instrument_program,
                is_drum=args.is_drum,
                humanize_timing=args.humanize_timing,
                humanize_vel=args.humanize_vel,
            )
            if pm is not None:
                pm.write(str(args.out_midi))
        else:
            logging.warning("pretty_midi not installed; skipping MIDI export")

    print(f"Generated {len(out_events)} events.")
    if args.out_csv:
        print(f"CSV  -> {args.out_csv}")
    if args.out_midi:
        print(f"MIDI -> {args.out_midi}")
    info = {
        "n_events": len(out_events),
        "duv_mode": eff_duv,
        "topp": args.topp,
        "topk": args.topk,
        "temperature_start": temp_start,
        "temperature_end": temp_end,
        "temperature_schedule": "linear",
        "pitch_min": args.pitch_min,
        "pitch_max": args.pitch_max,
        "seed": args.seed,
    }
    cfg_path: Path | None = None
    if args.out_midi is not None:
        cfg_path = args.out_midi.with_suffix(args.out_midi.suffix + ".json")
    elif args.out_csv is not None:
        cfg_path = args.out_csv.with_suffix(args.out_csv.suffix + ".json")
    if cfg_path is not None:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":  # pragma: no cover
    main()

