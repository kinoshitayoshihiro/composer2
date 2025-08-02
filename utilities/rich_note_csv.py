from __future__ import annotations

import csv
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import pretty_midi
except Exception:  # pragma: no cover - optional dependency
    pretty_midi = None  # type: ignore


@dataclass
class NoteRow:
    pitch: int
    duration: float
    bar: int
    position: int
    velocity: int
    chord_symbol: str
    articulation: str
    q_onset: float
    q_duration: float
    cc64: Optional[int] = None
    bend: Optional[int] = None


def _last_value(times: List[float], values: List[int], t: float) -> int:
    """Return the last value at or before ``t`` using binary search."""
    idx = bisect_right(times, t) - 1
    return values[idx] if idx >= 0 else 0


def scan_midi_files(
    paths: Iterable[Path], include_cc: bool, include_bend: bool
) -> List[NoteRow]:
    """Return rows of rich note data from ``paths``."""
    if pretty_midi is None:  # pragma: no cover - handled at runtime
        raise RuntimeError("pretty_midi required")

    rows: List[NoteRow] = []
    for path in paths:
        try:
            pm = pretty_midi.PrettyMIDI(str(path))
        except Exception:
            continue

        if pm.time_signature_changes:
            ts = pm.time_signature_changes[0]
            beats_per_bar = ts.numerator * 4 / ts.denominator
        else:
            beats_per_bar = 4
        # TODO: support mid-piece time signature changes; only the first is used.
        subdiv = int(beats_per_bar * 4)

        for inst in pm.instruments:
            cc_times: List[float] = []
            cc_vals: List[int] = []
            if include_cc:
                cc_times = [cc.time for cc in inst.control_changes if cc.number == 64]
                cc_vals = [cc.value for cc in inst.control_changes if cc.number == 64]
            # Large CC/pitch-bend lists may still incur seconds of preprocessing in
            # rare extreme cases (>1M messages) but lookups remain O(log n).
            bend_times: List[float] = []
            bend_vals: List[int] = []
            if include_bend:
                bend_times = [b.time for b in inst.pitch_bends]
                bend_vals = [b.pitch for b in inst.pitch_bends]

            for note in inst.notes:
                tick = pm.time_to_tick(note.start)
                end_tick = pm.time_to_tick(note.end)
                q_onset = tick / pm.resolution
                q_duration = (end_tick - tick) / pm.resolution
                sixteenth = int(tick * 4 // pm.resolution)
                bar = sixteenth // subdiv
                position = sixteenth % subdiv
                cc64 = (
                    _last_value(cc_times, cc_vals, note.start) if include_cc else None
                )
                bend = (
                    _last_value(bend_times, bend_vals, note.start)
                    if include_bend
                    else None
                )
                rows.append(
                    NoteRow(
                        pitch=note.pitch,
                        duration=note.end - note.start,
                        bar=bar,
                        position=position,
                        velocity=note.velocity,
                        chord_symbol="",
                        articulation="",
                        q_onset=q_onset,
                        q_duration=q_duration,
                        cc64=cc64,
                        bend=bend,
                    )
                )
    return rows


def build_note_csv(
    src: Path, out: Path, include_cc: bool = True, include_bend: bool = True
) -> None:
    """Extract rich note data from ``src`` MIDI folder into ``out`` CSV."""
    midi_paths = sorted(src.rglob("*.mid"))
    rows = scan_midi_files(midi_paths, include_cc, include_bend)

    out.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "pitch",
        "duration",
        "bar",
        "position",
        "velocity",
        "chord_symbol",
        "articulation",
        "q_onset",
        "q_duration",
    ]
    if include_cc:
        headers.append("CC64")
    if include_bend:
        headers.append("bend")

    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            if include_cc:
                data["CC64"] = data.pop("cc64")
            else:
                data.pop("cc64")
            if not include_bend:
                data.pop("bend")
            writer.writerow(data)


def coverage_stats(csv_path: Path) -> None:
    """Print percentage of non-null values for each column using pandas."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    total = len(df)
    for col in df.columns:
        non_null = df[col].notna().sum()
        pct = (non_null / total * 100) if total else 0
        print(f"{col}: {pct:.1f}% coverage ({non_null}/{total})")


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Build rich note CSV from MIDI files")
    p.add_argument("src", type=Path, nargs="?", help="Directory containing MIDI files")
    p.add_argument("--out", type=Path, help="Output CSV path")
    p.add_argument("--no-cc", action="store_true", help="Exclude sustain pedal column")
    p.add_argument("--no-bend", action="store_true", help="Exclude pitch bend column")
    p.add_argument(
        "--coverage", type=Path, help="Compute coverage stats for an existing CSV"
    )
    args = p.parse_args(argv)

    if args.coverage:
        coverage_stats(args.coverage)
        return 0

    if args.src is None or args.out is None:
        p.error("src and --out required unless --coverage is specified")

    build_note_csv(
        args.src, args.out, include_cc=not args.no_cc, include_bend=not args.no_bend
    )
    print(f"✅ Rich note CSV generated: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
