from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Tuple, Iterable

try:
    import pretty_midi
except Exception:  # pragma: no cover - optional
    pretty_midi = None  # type: ignore


def scan_midi_files(paths: Iterable[Path]) -> tuple[list[list[float]], list[list[str]]]:
    """Return per-note velocity rows and simple track statistics."""
    if pretty_midi is None:
        raise RuntimeError("pretty_midi required")
    rows: list[list[float]] = []
    stats: list[list[str]] = []
    for path in paths:
        try:
            pm = pretty_midi.PrettyMIDI(str(path))  # type: ignore[arg-type]
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


def validate_build_inputs(tracks_dir: Path, drums_dir: Path, csv_out: Path, stats_out: Path) -> None:
    """Verify directories exist and output locations are writable."""
    if not tracks_dir.is_dir():
        raise FileNotFoundError(f"tracks-dir not found: {tracks_dir}")
    if not drums_dir.is_dir():
        raise FileNotFoundError(f"drums-dir not found: {drums_dir}")
    for p in [csv_out.parent, stats_out.parent]:
        p.mkdir(parents=True, exist_ok=True)
        if not os.access(p, os.W_OK):
            raise PermissionError(f"cannot write to {p}")


def build_velocity_csv(
    tracks_dir: Path,
    drums_dir: Path,
    csv_out: Path,
    stats_out: Path,
) -> None:
    """Build velocity CSV and simple track statistics."""
    validate_build_inputs(tracks_dir, drums_dir, csv_out, stats_out)
    midi_paths = sorted(tracks_dir.rglob("*.mid")) + sorted(drums_dir.rglob("*.mid"))
    rows, stats = scan_midi_files(midi_paths)
    with csv_out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pitch", "duration", "prev_vel", "velocity"])
        writer.writerows(rows)
    with stats_out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "events"])
        writer.writerows(stats)


__all__ = ["build_velocity_csv"]
