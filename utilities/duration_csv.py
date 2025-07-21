from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

try:
    import pretty_midi
except Exception:  # pragma: no cover
    pretty_midi = None  # type: ignore


def scan_midi_files(paths: Iterable[Path]) -> list[list]:
    """Return rows of note data with duration, bar, position, pitch, and velocity."""
    if pretty_midi is None:
        raise RuntimeError("pretty_midi required")
    rows = []
    bar_counter = 0
    position_counter = 0

    for path in paths:
        try:
            pm = pretty_midi.PrettyMIDI(str(path))
        except Exception:
            continue
        for inst in pm.instruments:
            for note in inst.notes:
                duration = note.end - note.start
                pitch = note.pitch  # 音高情報
                velocity = note.velocity  # 音の強さ情報
                # Add all required note information
                rows.append([duration, bar_counter, position_counter, pitch, velocity])
                position_counter += 1
                # Every 16 notes, increment bar (adjustable)
                if position_counter >= 16:
                    position_counter = 0
                    bar_counter += 1
    return rows


def build_duration_csv(src: Path, out: Path) -> None:
    """Extract complete note data from ``src`` MIDI folder into ``out`` CSV."""
    midi_paths = sorted(src.rglob("*.mid"))
    rows = scan_midi_files(midi_paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        # Add all required columns including pitch and velocity
        writer.writerow(["duration", "bar", "position", "pitch", "velocity"])
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Build complete duration CSV with note data"
    )
    p.add_argument("src", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    build_duration_csv(args.src, args.out)
    print(f"✅ Complete duration CSV with pitch/velocity generated: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
