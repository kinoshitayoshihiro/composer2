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

import argparse
import csv
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

# Optional deps --------------------------------------------------------------
try:  # pragma: no cover - optional
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover - optional
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore

try:  # pragma: no cover - optional
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover
    pretty_midi = None  # type: ignore

logger = logging.getLogger(__name__)

# Project import guard (same pattern as train_phrase.py) ---------------------
try:  # pragma: no cover - regular import
    from models.phrase_transformer import PhraseTransformer  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - local fallback
    repo_root = Path(__file__).resolve().parent.parent
    if os.environ.get("ALLOW_LOCAL_IMPORT") == "1":
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
            logging.warning("ALLOW_LOCAL_IMPORT=1, inserted repo root into sys.path")
        try:
            from models.phrase_transformer import PhraseTransformer  # type: ignore
        except ModuleNotFoundError:  # still failing
            PhraseTransformer = None  # type: ignore
    else:
        PhraseTransformer = None  # type: ignore

# Optional utility for de-normalizing velocity/duration ----------------------
try:  # pragma: no cover
    from utilities.phrase_data import denorm_duv  # type: ignore
except Exception:  # pragma: no cover - simple fallback
    def denorm_duv(vel_reg, dur_reg):  # type: ignore[override]
        """Fallback: map vel in [0,1] -> [0,127]; dur is expm1-regressed beats."""
        import torch as _torch  # local import to avoid global hard dep
        vel = _torch.clamp((_torch.tensor(vel_reg) * 127.0).round(), 0, 127)
        dur = _torch.expm1(_torch.tensor(dur_reg))
        return vel, dur


# Instrument pitch range presets --------------------------------------------
PITCH_PRESETS = {
    "bass": (28, 52),
    "piano": (21, 108),
    "strings": (40, 84),
}


# ----------------------- helpers -------------------------------------------

def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_hparams_sidecar(ckpt_path: Path) -> dict[str, Any] | None:
    """Try to load training hparams from a sidecar JSON next to the ckpt.

    The training script writes ``hparams.json`` in the checkpoint directory.
    We use it here to reconstruct model hyperparameters that are not stored
    in the checkpoint itself.
    """
    cand = ckpt_path.parent / "hparams.json"
    return _load_json(cand) if cand.is_file() else None


def load_checkpoint(path: Path) -> dict[str, Any]:
    if torch is None:
        raise SystemExit("torch is required for sampling")
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        return obj
    raise SystemExit(
        f"Unexpected checkpoint format: {list(obj.keys()) if isinstance(obj, dict) else type(obj)}"
    )


def build_model_from_meta(state: dict[str, Any], hparams: dict[str, Any] | None) -> Any:
    """Rebuild the model using checkpoint state and (optional) sidecar hparams.

    The training script stores architectural metadata inside the checkpoint
    (``state['meta']``).  Some historical runs additionally wrote a
    ``hparams.json`` sidecar.  We prefer explicit metadata from the checkpoint,
    fall back to the sidecar, and lastly rely on library defaults.  This keeps
    toy models (e.g., ``d_model=16``) loadable without relaxing
    ``load_state_dict``.
    """

    if PhraseTransformer is None:
        raise SystemExit(
            "Could not import project modules. Run 'pip install -e .' or set ALLOW_LOCAL_IMPORT=1"
        )

    meta = state.get("meta", {}) or {}
    hp = hparams or {}
    duv_cfg = state.get("duv_cfg", {}) or {}

    def _pick(key: str, *fallback_keys: str, default: Any = None) -> Any:
        """Return the first present value from metadata, hparams, or state."""

        sources = (meta, hp, state)
        keys = (key, *fallback_keys)
        for src in sources:
            if not isinstance(src, dict):
                continue
            for k in keys:
                if k in src and src[k] is not None:
                    return src[k]
        return default

    def _pick_int(key: str, *fallback_keys: str, default: int = 0) -> int:
        value = _pick(key, *fallback_keys, default=default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _pick_float(key: str, *fallback_keys: str, default: float = 0.0) -> float:
        value = _pick(key, *fallback_keys, default=default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _pick_bool(key: str, default: bool = False) -> bool:
        value = _pick(key, default=default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    d_model = _pick_int("d_model", default=512)
    max_len = _pick_int("max_len", default=128)
    n_layers = _pick_int("n_layers", "num_layers", "layers", default=4)
    n_heads = _pick_int("n_heads", "nhead", "num_heads", default=8)
    dropout = _pick_float("dropout", default=0.1)

    duv_mode = _pick("duv_mode", default=duv_cfg.get("mode", "reg")) or "reg"
    vel_bins = _pick_int("vel_bins", default=duv_cfg.get("vel_bins", 0))
    dur_bins = _pick_int("dur_bins", default=duv_cfg.get("dur_bins", 0))

    ctor = dict(
        d_model=d_model,
        max_len=max_len,
        nhead=n_heads,
        num_layers=n_layers,
        dropout=dropout,
        duv_mode=duv_mode,
        vel_bins=vel_bins,
        dur_bins=dur_bins,
        section_vocab_size=_pick_int("section_vocab_size"),
        mood_vocab_size=_pick_int("mood_vocab_size"),
        vel_bucket_size=_pick_int("vel_bucket_size"),
        dur_bucket_size=_pick_int("dur_bucket_size"),
        use_sinusoidal_posenc=_pick_bool("use_sinusoidal_posenc"),
        use_bar_beat=_pick_bool("use_bar_beat"),
        pitch_vocab_size=_pick_int("pitch_vocab_size", "vocab_pitch", default=0),
    )

    model = PhraseTransformer(**ctor)
    return model


def decode_duv(
    out_dict: dict[str, Any],
    vel_mode: str,
    dur_mode: str,
    meta: dict[str, Any],
    dur_max_beats: float,
) -> tuple[float, float]:
    """Return real velocity and duration from model outputs.

    Supports mixed regression/bucket decoding. If neither head exists, falls
    back to sensible defaults.
    """
    if torch is None or F is None:
        return 64.0, 0.25

    if vel_mode == "reg" and "vel_reg" in out_dict:
        vel = float(denorm_duv(out_dict["vel_reg"], 0.0)[0])
    elif "vel_cls" in out_dict:
        vel_idx = int(F.softmax(out_dict["vel_cls"], dim=-1).multinomial(1).item())
        vel_bins = int(meta.get("vel_bins", 8))
        vel = (vel_idx + 0.5) * (127.0 / max(1, vel_bins))
    else:
        vel = 64.0

    if dur_mode == "reg" and "dur_reg" in out_dict:
        dur = float(denorm_duv(0.0, out_dict["dur_reg"])[1])
    elif "dur_cls" in out_dict:
        dur_idx = int(F.softmax(out_dict["dur_cls"], dim=-1).multinomial(1).item())
        dur_bins = int(meta.get("dur_bins", 16))
        dur = 4.0 * (dur_idx + 1) / max(1, dur_bins)
    else:
        dur = 0.25

    vel = float(max(1.0, min(127.0, vel)))
    dur = float(min(dur_max_beats, max(1e-3, dur)))
    return vel, dur


def sample_logits(logits, temperature: float, topk: int, topp: float) -> int:
    """Sample an index from *logits* with temperature/top-k/top-p filtering."""
    if torch is None or F is None:
        return 60

    if temperature is None or temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / float(temperature)
    probs = F.softmax(logits, dim=-1)

    # top-k
    if topk and topk > 0:
        k = min(topk, probs.numel())
        topk_probs, topk_idx = torch.topk(probs, k)
        keep = torch.zeros_like(probs)
        keep.scatter_(0, topk_idx, topk_probs)
        probs = keep
        probs = probs / probs.sum()

    # top-p (nucleus)
    if topp and topp > 0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        mask = cdf <= topp
        if sorted_probs.numel() > 0:
            mask[0] = True
        filtered = sorted_probs * mask
        filtered = filtered / filtered.sum()
        idx = sorted_idx[torch.multinomial(filtered, 1)]
        return int(idx.item())

    idx = torch.multinomial(probs, 1)
    return int(idx.item())


def events_to_prettymidi(
    events: list[dict[str, float | int]],
    bpm: float,
    *,
    gm_program: int = 33,
    is_drum: bool = False,
    humanize_timing: float = 0.0,
    humanize_vel: float = 0.0,
):
    """Convert simple event dicts to a PrettyMIDI object (or None if unavailable)."""
    if pretty_midi is None:
        return None
    try:
        pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    except Exception:  # pragma: no cover - fallback path
        pm = pretty_midi.PrettyMIDI()
        if np is not None:
            pm._tempo_changes = np.array([0.0])  # type: ignore[attr-defined]
            pm._tempos = np.array([float(bpm)])  # type: ignore[attr-defined]
    inst = pretty_midi.Instrument(program=int(gm_program), name="phrase", is_drum=is_drum)
    t = 0.0
    sec_per_beat = 60.0 / float(bpm)
    for ev in events:
        pitch = int(ev["pitch"]) if "pitch" in ev else 60
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


def _resolve_steps_per_bar(state: dict[str, Any], hparams: dict[str, Any] | None) -> int:
    """Resolve steps-per-bar with priority: hparams → state → state.meta → default."""

    sources: tuple[Any, ...] = (hparams or {}, state, {})
    if isinstance(state, dict):
        sources = (hparams or {}, state, state.get("meta", {}) or {})
    for source in sources:
        if not isinstance(source, dict):
            continue
        value = source.get("steps_per_bar")
        if value is None:
            continue
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            continue
        if candidate:
            return candidate
    return 4


def _compute_step_limits(
    *, length: int, model_max_len: int, bars: int | None, steps_per_bar: int
) -> tuple[int, int | None]:
    """Return ``(max_steps, bar_step_limit)`` for the sampling loop."""

    max_steps = min(int(length), int(model_max_len))
    bar_step_limit: int | None = None
    if bars is not None:
        bar_step_limit = max(0, int(bars) * int(steps_per_bar))
        if bar_step_limit > 0:
            max_steps = min(max_steps, bar_step_limit)
    return max(0, max_steps), bar_step_limit


def _generate_fallback_events(
    *, max_steps: int, pitch_min: int, pitch_max: int
) -> list[dict[str, float | int]]:
    """Return random placeholder events used when Torch is unavailable."""

    events: list[dict[str, float | int]] = []
    for _ in range(max_steps):
        pitch = random.randint(pitch_min, pitch_max)
        events.append({"pitch": pitch, "velocity": 64, "duration_beats": 0.25})
    return events


# ----------------------- main inference ------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        epilog=(
            "If project modules fail to import, run 'pip install -e .' or set ALLOW_LOCAL_IMPORT=1"
        )
    )
    ap.add_argument(
        "--ckpt",
        required=True,
        type=Path,
        help="checkpoint from train_phrase.py (install package or set ALLOW_LOCAL_IMPORT=1)",
    )
    ap.add_argument("--length", type=int, default=64, help="number of steps to generate")
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="deprecated; use --temperature-start/--temperature-end",
    )
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
    ap.add_argument(
        "--bars", type=int, default=None, help="stop after N bars (approx 4 beats each)"
    )
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
    args = ap.parse_args(argv)

    temp_start = args.temperature if args.temperature is not None else args.temperature_start
    temp_end = args.temperature if args.temperature is not None else args.temperature_end

    if args.instrument_program < 0 or args.instrument_program > 127:
        raise SystemExit("--instrument-program must be in [0,127]")

    if args.instrument_name:
        args.pitch_min, args.pitch_max = PITCH_PRESETS[args.instrument_name]

    if args.pitch_min > args.pitch_max:
        raise SystemExit("--pitch-min must be <= --pitch-max")

    if args.seed is not None:
        random.seed(args.seed)
        if torch is not None:
            torch.manual_seed(args.seed)
        if np is not None:
            np.random.seed(args.seed)

    # Load checkpoint and hparams
    state = load_checkpoint(args.ckpt)
    hparams = load_hparams_sidecar(args.ckpt)

    # Reconstruct model and heads
    model = build_model_from_meta(state, hparams)
    model.load_state_dict(state["model"])  # type: ignore[arg-type]
    device = torch.device(args.device) if torch is not None else None
    if device is not None and hasattr(model, "to"):
        model = model.to(device)  # type: ignore[assignment]
    if hasattr(model, "eval"):
        model.eval()

    # Decide decoding modes
    duv_mode_tr = (hparams or {}).get("duv_mode", state.get("duv_cfg", {}).get("mode", "reg"))
    eff_duv = args.duv_decode if args.duv_decode != "auto" else duv_mode_tr
    vel_mode = args.vel_decode or ("reg" if eff_duv == "reg" else "bucket")
    dur_mode = args.dur_decode or ("reg" if eff_duv == "reg" else "bucket")

    # Seed events (optional)
    seq: list[dict[str, float | int]] = []
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
    # Truncate to model context if needed
    max_len = getattr(model, "max_len", 128)
    if seq and len(seq) > int(max_len):
        raise SystemExit("seed longer than model max_len")
    if args.seed_csv and not seq:
        raise SystemExit("seed CSV had no rows")

    # Encode seed if the model supports it; otherwise, we'll keep a local buffer
    if hasattr(model, "encode_seed"):
        state_enc = model.encode_seed(seq)  # type: ignore[attr-defined]
    else:
        state_enc = None
        print("NOTE: Adapt 'encode_seed' and 'step' calls to your model API.", file=sys.stderr)

    # Sampling loop ----------------------------------------------------------
    out_events: list[dict[str, float | int]] = []
    total_beats = 0.0
    steps_per_bar = _resolve_steps_per_bar(state, hparams)
    meta = {
        "vel_bins": (hparams or {}).get("vel_bins", state.get("duv_cfg", {}).get("vel_bins", 8)),
        "dur_bins": (hparams or {}).get("dur_bins", state.get("duv_cfg", {}).get("dur_bins", 16)),
        "steps_per_bar": steps_per_bar,
    }

    model_max_len = int(getattr(model, "max_len", args.length))
    max_steps, bar_step_limit = _compute_step_limits(
        length=int(args.length),
        model_max_len=model_max_len,
        bars=args.bars,
        steps_per_bar=steps_per_bar,
    )
    logger.debug(
        "max_steps=%d bar_step_limit=%s model_max_len=%d",
        max_steps,
        bar_step_limit,
        model_max_len,
    )

    if torch is None:
        # Fallback: generate a constant-length, random-velocity sequence
        out_events.extend(
            _generate_fallback_events(
                max_steps=max_steps,
                pitch_min=args.pitch_min,
                pitch_max=args.pitch_max,
            )
        )
        logging.warning("Torch not available; generated fallback events without model")
    else:
        with torch.no_grad():
            for step in range(max_steps):
                # Query model step API if available; otherwise synthesize empty outputs
                if hasattr(model, "step"):
                    out_dict = model.step(state_enc)  # type: ignore[attr-defined]
                else:
                    out_dict = {}

                # Temperature schedule (linear)
                alpha = step / max(max_steps - 1, 1)
                temp = temp_start + (temp_end - temp_start) * alpha

                # Pitch sampling
                if "pitch_logits" in out_dict:
                    logits = out_dict["pitch_logits"].squeeze(0)
                    if device is not None:
                        logits = logits.to(device)
                    pitch = sample_logits(logits, temp, args.topk, args.topp)
                else:
                    pitch = random.randint(36, 51)
                pitch = int(max(args.pitch_min, min(args.pitch_max, pitch)))

                # Velocity/Duration decoding
                vel, dur = decode_duv(out_dict, vel_mode, dur_mode, meta, args.dur_max_beats)

                ev = {"pitch": pitch, "velocity": vel, "duration_beats": dur}
                out_events.append(ev)
                total_beats += float(dur)

                # Advance model state if supported
                if hasattr(model, "update_state"):
                    state_enc = model.update_state(state_enc, ev)  # type: ignore[attr-defined]

                # Optional early stop by bars
                if bar_step_limit is not None and len(out_events) >= bar_step_limit:
                    break

    # Outputs ---------------------------------------------------------------
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
