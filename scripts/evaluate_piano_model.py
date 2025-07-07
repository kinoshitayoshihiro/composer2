from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pretty_midi


def _extract(midi: pretty_midi.PrettyMIDI) -> tuple[list[int], list[float], list[int]]:
    pitches: list[int] = []
    onsets: list[float] = []
    velocities: list[int] = []
    for inst in midi.instruments:
        for note in inst.notes:
            pitches.append(int(note.pitch))
            onsets.append(float(note.start))
            velocities.append(int(note.velocity))
    return pitches, onsets, velocities


def _pitch_precision_recall(ref: Iterable[int], gen: Iterable[int]) -> tuple[float, float]:
    ref_set = set(ref)
    gen_set = set(gen)
    if not gen_set:
        return 0.0, 0.0
    correct = len(ref_set & gen_set)
    precision = correct / len(gen_set)
    recall = correct / len(ref_set) if ref_set else 0.0
    return float(precision), float(recall)


def _npvi(onsets: list[float]) -> float:
    if len(onsets) < 3:
        return 0.0
    onsets.sort()
    intervals = np.diff(onsets)
    ratios = []
    for a, b in zip(intervals[:-1], intervals[1:]):
        if a + b == 0:
            continue
        ratios.append(abs(a - b) / ((a + b) / 2))
    return float(np.mean(ratios) * 100.0) if ratios else 0.0


def _groove_similarity(ref_onsets: list[float], gen_onsets: list[float]) -> float:
    ref_v = _npvi(ref_onsets)
    gen_v = _npvi(gen_onsets)
    if ref_v == 0.0 and gen_v == 0.0:
        return 1.0
    diff = abs(ref_v - gen_v)
    denom = max(ref_v, gen_v, 1e-5)
    return float(max(1.0 - diff / denom, 0.0))


def _vel_kl(ref_v: list[int], gen_v: list[int]) -> float:
    hist_r, _ = np.histogram(ref_v, bins=32, range=(0, 127))
    hist_g, _ = np.histogram(gen_v, bins=32, range=(0, 127))
    pr = hist_r / hist_r.sum() if hist_r.sum() else np.ones_like(hist_r) / len(hist_r)
    pg = hist_g / hist_g.sum() if hist_g.sum() else np.ones_like(hist_g) / len(hist_g)
    eps = 1e-8
    return float(np.sum(pr * np.log((pr + eps) / (pg + eps))))


def evaluate_piece(ref_path: Path, gen_path: Path) -> dict[str, float]:
    ref_midi = pretty_midi.PrettyMIDI(str(ref_path))
    gen_midi = pretty_midi.PrettyMIDI(str(gen_path))
    ref_p, ref_o, ref_v = _extract(ref_midi)
    gen_p, gen_o, gen_v = _extract(gen_midi)
    prec, recall = _pitch_precision_recall(ref_p, gen_p)
    groove = _groove_similarity(ref_o, gen_o)
    vel_kl = _vel_kl(ref_v, gen_v)
    return {
        "pitch_precision": prec,
        "pitch_recall": recall,
        "groove_similarity": groove,
        "velocity_kl": vel_kl,
    }


def evaluate_dirs(ref_dir: Path, gen_dir: Path, *, out_dir: Path) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, float] = {
        "pitch_precision": 0.0,
        "pitch_recall": 0.0,
        "groove_similarity": 0.0,
        "velocity_kl": 0.0,
    }
    pieces = 0
    for ref_file in sorted(ref_dir.glob("*.mid")):
        gen_file = gen_dir / ref_file.name
        if not gen_file.exists():
            continue
        metrics = evaluate_piece(ref_file, gen_file)
        with (out_dir / f"{ref_file.stem}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)
        for k, v in metrics.items():
            summary[k] += v
        pieces += 1
    if pieces:
        for k in summary:
            summary[k] /= pieces
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated piano MIDIs")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluate_dirs(args.reference, args.generated, out_dir=args.out)


if __name__ == "__main__":
    main()
