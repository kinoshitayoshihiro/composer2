from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import pretty_midi


def _scan_midi_files(paths: List[Path]) -> Tuple[List[List[float]], List[List[str]]]:
    """Return per-note velocity rows and simple track statistics."""
    rows: List[List[float]] = []
    stats: List[List[str]] = []
    for path in paths:
        try:
            pm = pretty_midi.PrettyMIDI(str(path))
        except Exception:
            continue
        total = 0
        prev_vel = 64
        for inst in pm.instruments:
            for note in inst.notes:
                rows.append([
                    note.pitch,
                    note.end - note.start,
                    prev_vel,
                    note.velocity,
                ])
                prev_vel = note.velocity
                total += 1
        stats.append([path.name, str(total)])
    return rows, stats


def build_velocity_csv(tracks_dir: Path, drums_dir: Path, csv_out: Path, stats_out: Path) -> None:
    midi_paths = sorted(tracks_dir.rglob("*.mid")) + sorted(drums_dir.rglob("*.mid"))
    rows, stats = _scan_midi_files(midi_paths)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pitch", "duration", "prev_vel", "velocity"])
        writer.writerows(rows)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    with stats_out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "events"])
        writer.writerows(stats)

__all__ = ["build_velocity_csv"]

