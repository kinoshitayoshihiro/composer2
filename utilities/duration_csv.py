from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

try:
    import pretty_midi
except Exception:  # pragma: no cover
    pretty_midi = None  # type: ignore


def scan_midi_files(paths: Iterable[Path]) -> list[list[float]]:
    """Return rows of note durations in seconds."""
    if pretty_midi is None:
        raise RuntimeError("pretty_midi required")
    rows: list[list[float]] = []
    for path in paths:
        try:
            pm = pretty_midi.PrettyMIDI(str(path))
        except Exception:
            continue
        for inst in pm.instruments:
            for note in inst.notes:
                rows.append([note.end - note.start])
    return rows


def build_duration_csv(src: Path, out: Path) -> None:
    """Extract durations from ``src`` MIDI folder into ``out`` CSV."""
    midi_paths = sorted(src.rglob("*.mid"))
    rows = scan_midi_files(midi_paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["duration"])
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Build duration CSV")
    p.add_argument("src", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    build_duration_csv(args.src, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
