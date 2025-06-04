"""utilities/prettymidi_sync.py
=================================================
Extract / apply vocal‑groove micro‑timing profiles
-------------------------------------------------
CLI Usage (examples)
--------------------
1) **Extract groove profile** (8‑th grid)
   ```bash
   python utilities/prettymidi_sync.py \
          --mode extract \
          --input data/vocal.mid \
          --subdiv 8 \
          --outfile data/groove.json
   ```

2) **Apply existing profile**
   ```bash
   python utilities/prettymidi_sync.py \
          --mode apply \
          --input drums.mid \
          --groove data/groove.json \
          --outfile drums_synced.mid
   ```

3) **One‑shot** (extract from vocal & apply to band track)
   ```bash
   python utilities/prettymidi_sync.py \
          --mode extract_apply \
          --input band.mid \
          --vocal vocal.mid \
          --subdiv 16 \
          --outfile band_synced.mid
   ```
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import statistics
import sys
from typing import Dict, List, Tuple

try:
    import pretty_midi
except ImportError:
    print("ERROR: pretty_midi is not installed.  `pip install pretty_midi`", file=sys.stderr)
    sys.exit(1)

################################################################################
# ── Grid helpers ──────────────────────────────────────────────────────────────
################################################################################

def _sec_per_subdivision(pm: "pretty_midi.PrettyMIDI", subdiv: int) -> float:
    """Return seconds per subdivision (8 => eighth‑note grid, 16 => 16th, etc.)."""
    # Tempo – use first tempo event
    tempi = pm.get_tempo_changes()[1]
    bpm = tempi[0] if len(tempi) else 120.0
    sec_per_beat = 60.0 / bpm
    # 4 subdivisions per beat ⇒ sixteenth grid, etc.
    sec_per_sub = sec_per_beat / (subdiv / 4)
    return sec_per_sub


def _grid_index_and_shift(time_s: float, sec_per_sub: float) -> Tuple[int, float]:
    idx = round(time_s / sec_per_sub)
    return idx, time_s - idx * sec_per_sub


def _collect_note_onsets(pm: "pretty_midi.PrettyMIDI") -> List[float]:
    onsets: List[float] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        onsets.extend(n.start for n in inst.notes)
    return sorted(onsets)

################################################################################
# ── Profile IO ────────────────────────────────────────────────────────────────
################################################################################

def _write_profile(path: pathlib.Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_profile(path: pathlib.Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))

################################################################################
# ── Extraction ────────────────────────────────────────────────────────────────
################################################################################

def extract_groove(pm: "pretty_midi.PrettyMIDI", subdiv: int) -> Dict:
    sec_per_sub = _sec_per_subdivision(pm, subdiv)
    onsets = _collect_note_onsets(pm)
    shifts: List[float] = []
    hist: Dict[str, int] = {}
    for t in onsets:
        _, shift = _grid_index_and_shift(t, sec_per_sub)
        shifts.append(shift)
        bucket = f"{shift:.4f}"
        hist[bucket] = hist.get(bucket, 0) + 1

    bpm = pm.get_tempo_changes()[1][0] if pm.get_tempo_changes()[1].size else 120.0
    return {
        "bpm": bpm,
        "subdiv": subdiv,
        "sec_per_sub": sec_per_sub,
        "mean_shift_sec": statistics.mean(shifts) if shifts else 0.0,
        "stdev_shift_sec": statistics.stdev(shifts) if len(shifts) > 1 else 0.0,
        "histogram": hist,
    }

################################################################################
# ── Application ───────────────────────────────────────────────────────────────
################################################################################

def apply_groove(pm: "pretty_midi.PrettyMIDI", profile: Dict, *, strength: float = 1.0, min_shift_sec: float = 1e-3):
    subdiv = int(profile.get("subdiv", 16))
    sec_per_sub = float(profile.get("sec_per_sub", _sec_per_subdivision(pm, subdiv)))
    hist = profile.get("histogram", {})
    if not hist:
        return
    buckets = [float(k) for k in hist.keys()]
    weights = [hist[k] for k in hist.keys()]

    for inst in pm.instruments:
        for n in inst.notes:
            idx, _ = _grid_index_and_shift(n.start, sec_per_sub)
            target_shift = random.choices(buckets, weights)[0] * strength
            new_start = idx * sec_per_sub + target_shift
            delta = new_start - n.start
            if abs(delta) < min_shift_sec:
                continue
            n.start += delta
            n.end += delta

################################################################################
# ── CLI ───────────────────────────────────────────────────────────────────────
################################################################################

def main():
    p = argparse.ArgumentParser(description="Vocal groove extractor / applier (PrettyMIDI)")
    p.add_argument("--mode", choices=["extract", "apply", "extract_apply"], required=True)
    p.add_argument("--input", required=True, help="Input MIDI file (for extract/apply)")
    p.add_argument("--vocal", help="Vocal MIDI (when extract_apply)")
    p.add_argument("--groove", help="Groove JSON (when apply)")
    p.add_argument("--subdiv", type=int, default=16)
    p.add_argument("--outfile", required=True)
    p.add_argument("--strength", type=float, default=1.0, help="Groove strength 0‑1")
    p.add_argument("--quantize", type=float, default=1e-3, help="Minimum shift to apply (sec)")
    args = p.parse_args()

    in_path = pathlib.Path(args.input)
    out_path = pathlib.Path(args.outfile)

    if args.mode == "extract":
        pm = pretty_midi.PrettyMIDI(str(in_path))
        prof = extract_groove(pm, args.subdiv)
        _write_profile(out_path, prof)
        print(f"[Groove] profile extracted to {out_path}")

    elif args.mode == "apply":
        if not args.groove:
            p.error("--groove required in apply mode")
        prof = _read_profile(pathlib.Path(args.groove))
        pm = pretty_midi.PrettyMIDI(str(in_path))
        apply_groove(pm, prof, strength=args.strength, min_shift_sec=args.quantize)
        pm.write(str(out_path))
        print(f"[Groove] applied groove -> {out_path}")

    elif args.mode == "extract_apply":
        if not args.vocal:
            p.error("--vocal required in extract_apply mode")
        vocal_pm = pretty_midi.PrettyMIDI(str(pathlib.Path(args.vocal)))
        prof = extract_groove(vocal_pm, args.subdiv)
        band_pm = pretty_midi.PrettyMIDI(str(in_path))
        apply_groove(band_pm, prof, strength=args.strength, min_shift_sec=args.quantize)
        band_pm.write(str(out_path))
        print(f"[Groove] extracted from {args.vocal} and applied -> {out_path}")

if __name__ == "__main__":
    main()
