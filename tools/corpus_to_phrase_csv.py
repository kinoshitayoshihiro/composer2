from __future__ import annotations

"""Convert MIDI or corpus samples to phrase CSVs."""

import argparse
import csv
import json
import random
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Pattern

import pretty_midi
import yaml


FIELDS = [
    "pitch",
    "velocity",
    "duration",
    "pos",
    "boundary",
    "bar",
    "instrument",
]


def bin_duration(duration_beats: float, bins: int) -> int:
    """Map a duration in beats to a discrete bin index."""

    return max(0, min(bins - 1, round(duration_beats * bins)))


def bin_velocity(velocity: int, bins: int) -> int:
    """Map velocity 0-127 to a discrete bin index."""

    return max(0, min(bins - 1, velocity * bins // 128))


def deterministic_split(
    files: list[Path], valid_ratio: float, seed: int
) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    files = files.copy()
    rng.shuffle(files)
    split = max(1, int(len(files) * (1 - valid_ratio)))
    return files[:split], files[split:]


def midi_to_rows(
    path: Path,
    gap: float,
    sections: list[tuple[float, str]] | None,
    instrument_filter: str | None,
    instrument_regex: Pattern[str] | None,
    pitch_range: tuple[int, int] | None,
    *,
    emit_buckets: bool = False,
    dur_bins: int = 16,
    vel_bins: int = 8,

) -> tuple[list[dict[str, object]], bool]:
    """Parse *path* into phrase rows using simple boundary heuristics.

    Returns the parsed rows and whether any instrument matched *instrument_filter*.
    """

    pm = pretty_midi.PrettyMIDI(str(path))
    ticks_per_beat = pm.resolution
    notes: list[tuple[float, float, int, int, str]] = []
    inst_match = False
    for inst in pm.instruments:
        name = inst.name or pretty_midi.program_to_instrument_name(inst.program)
        if instrument_filter and instrument_filter not in name.lower():
            continue
        if instrument_regex and not instrument_regex.search(name):
            continue
        inst_match = True
        for n in inst.notes:
            if pitch_range and not (pitch_range[0] <= n.pitch <= pitch_range[1]):
                continue
            start_tick = pm.time_to_tick(n.start)
            end_tick = pm.time_to_tick(n.end)
            start_beats = start_tick / ticks_per_beat
            dur_beats = (end_tick - start_tick) / ticks_per_beat
            notes.append((start_beats, dur_beats, n.pitch, n.velocity, name))
    notes.sort()
    rows: list[dict[str, object]] = []
    prev_start = None
    prev_bar = None
    prev_section = None
    for start, dur, pitch, vel, name in notes:
        bar = int(start // 4)
        pos = int((start % 4) * ticks_per_beat)
        boundary = 0
        cur_section = None
        if sections:
            for t, s in sections:
                if start >= t:
                    cur_section = s
                else:
                    break
        if (
            prev_start is None
            or bar != prev_bar
            or start - prev_start >= gap
            or cur_section != prev_section
        ):
            boundary = 1
        row = {
            "pitch": pitch,
            "velocity": vel,
            "duration": dur,
            "pos": pos,
            "boundary": boundary,
            "bar": bar,
            "instrument": name,
        }
        if emit_buckets:
            row["velocity_bucket"] = bin_velocity(vel, vel_bins)
            row["duration_bucket"] = bin_duration(dur, dur_bins)
        rows.append(row)
        prev_start = start
        prev_bar = bar
        prev_section = cur_section
    return rows, inst_match


def write_csv(
    rows: Iterable[dict[str, object]], path: Path, fields: list[str] = FIELDS
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_samples_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def corpus_mode(
    root: Path,
    *,
    emit_buckets: bool = False,
    dur_bins: int = 16,
    vel_bins: int = 8,
    instrument: str | None = None,
    instrument_regex: Pattern[str] | None = None,
    pitch_range: tuple[int, int] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    instrument = instrument.lower() if instrument else None

    def load_split(split: str) -> list[dict[str, object]]:
        base = root / split
        files: list[Path] = []
        if (base / "samples.jsonl").is_file():
            files = [base / "samples.jsonl"]
        else:
            files = sorted((base / "samples").glob("*.jsonl"))
        rows: list[dict[str, object]] = []
        stats = Counter()
        for p in files:
            for obj in read_samples_jsonl(p):
                name = str(obj.get("instrument", ""))
                pitch = int(obj["pitch"])
                if instrument or instrument_regex:
                    matched = False
                    name_l = name.lower()
                    if instrument and instrument in name_l:
                        matched = True
                    elif instrument_regex and instrument_regex.search(name):
                        matched = True
                    elif name_l in {"", "unknown"}:
                        fname = str(
                            obj.get("path")
                            or obj.get("source_path")
                            or obj.get("filename")
                            or obj.get("file")
                            or ""
                        )
                        stem = Path(fname).stem.lower()
                        if instrument and instrument in stem:
                            matched = True
                        elif instrument_regex and instrument_regex.search(stem):
                            matched = True
                        elif pitch_range and pitch_range[0] <= pitch <= pitch_range[1]:
                            matched = True
                    if not matched:
                        stats["removed_by_name"] += 1
                        continue
                if pitch_range and not (pitch_range[0] <= pitch <= pitch_range[1]):
                    stats["removed_by_pitch"] += 1
                    continue
                row = {
                    "pitch": pitch,
                    "velocity": obj.get("velocity", 0),
                    "duration": obj.get("duration", 0),
                    "pos": obj.get("pos", 0),
                    "boundary": obj.get("boundary", 0),
                    "bar": obj.get("bar", 0),
                    "instrument": name,
                }
                if emit_buckets:
                    row["velocity_bucket"] = bin_velocity(
                        int(row["velocity"]), vel_bins
                    )
                    row["duration_bucket"] = bin_duration(
                        float(row["duration"]), dur_bins
                    )
                rows.append(row)
                stats["kept"] += 1
        logging.info("%s stats %s", split, dict(stats))
        return rows

    train_rows = load_split("train")
    valid_rows = load_split("valid")
    if not train_rows or not valid_rows:
        raise SystemExit(
            "No rows matched filters. Try removing --instrument filters or use --pitch-range."
        )
    return train_rows, valid_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", type=Path)
    parser.add_argument("--out-train", type=Path, required=True)
    parser.add_argument("--out-valid", type=Path, required=True)
    parser.add_argument(
        "--instrument",
        help="case-insensitive substring filter (OR with --pitch-range in corpus mode)",
    )
    parser.add_argument(
        "--instrument-regex",
        help="regex instrument filter (OR with --pitch-range in corpus mode)",
    )
    parser.add_argument("--boundary-gap-beats", type=float, default=0.5)
    parser.add_argument("--boundary-on-section-change", action="store_true")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-notes", type=int, default=0)
    parser.add_argument(
        "--pitch-range",
        nargs=2,
        type=int,
        metavar=("LOW", "HIGH"),
        help="inclusive pitch filter; typical bass range 28-60",
    )
    parser.add_argument("--max-bars", type=int, default=10**9)
    parser.add_argument("--from-corpus", dest="from_corpus", type=Path)
    parser.add_argument(
        "--emit-buckets",
        action="store_true",
        help="emit velocity_bucket/duration_bucket columns",
    )
    parser.add_argument("--dur-bins", type=int, default=16)
    parser.add_argument("--vel-bins", type=int, default=8)
    args = parser.parse_args(argv)
    inst_re = re.compile(args.instrument_regex, re.I) if args.instrument_regex else None
    pitch_range = tuple(args.pitch_range) if args.pitch_range else None

    if args.from_corpus:
        train_rows, valid_rows = corpus_mode(
            args.from_corpus,
            emit_buckets=args.emit_buckets,
            dur_bins=args.dur_bins,
            vel_bins=args.vel_bins,
            instrument=args.instrument,
            instrument_regex=inst_re,
            pitch_range=pitch_range,
        )
        fields = (
            FIELDS + ["velocity_bucket", "duration_bucket"]
            if args.emit_buckets
            else FIELDS
        )
        write_csv(train_rows, args.out_train, fields)
        write_csv(valid_rows, args.out_valid, fields)
        return 0

    if args.src is None:
        raise SystemExit("--in is required when not using --from-corpus")

    files = [
        p
        for p in args.src.rglob("*")
        if p.suffix.lower() in {".mid", ".midi"}
    ]
    tags: dict[str, list[tuple[float, str]]] = {}
    if args.boundary_on_section_change:
        tag_path = args.src / "tags.yaml"
        if tag_path.is_file():
            raw = yaml.safe_load(tag_path.read_text()) or {}
            for fname, info in raw.items():
                sections = info.get("section") or []
                tags[fname] = [(float(t), str(s)) for t, s in sections]
    train_files, valid_files = deterministic_split(
        files, args.valid_ratio, args.seed
    )
    train_rows: list[dict[str, object]] = []
    train_stats = Counter()
    for p in train_files:
        sec = tags.get(p.name)
        rows, matched = midi_to_rows(
            p,
            args.boundary_gap_beats,
            sec,
            args.instrument.lower() if args.instrument else None,
            inst_re,
            pitch_range,
            emit_buckets=args.emit_buckets,
            dur_bins=args.dur_bins,
            vel_bins=args.vel_bins,
        )
        if not rows:
            if args.instrument and not matched:
                train_stats["instrument"] += 1
            else:
                train_stats["empty"] += 1
            continue
        if len(rows) < args.min_notes:
            train_stats["min_notes"] += 1
            continue
        if (rows[-1]["bar"] + 1) > args.max_bars:
            train_stats["max_bars"] += 1
            continue
        train_rows.extend(rows)
        train_stats["kept"] += 1

    valid_rows: list[dict[str, object]] = []
    valid_stats = Counter()
    for p in valid_files:
        sec = tags.get(p.name)
        rows, matched = midi_to_rows(
            p,
            args.boundary_gap_beats,
            sec,
            args.instrument.lower() if args.instrument else None,
            inst_re,
            pitch_range,
            emit_buckets=args.emit_buckets,
            dur_bins=args.dur_bins,
            vel_bins=args.vel_bins,
        )
        if not rows:
            if args.instrument and not matched:
                valid_stats["instrument"] += 1
            else:
                valid_stats["empty"] += 1
            continue
        if len(rows) < args.min_notes:
            valid_stats["min_notes"] += 1
            continue
        if (rows[-1]["bar"] + 1) > args.max_bars:
            valid_stats["max_bars"] += 1
            continue
        valid_rows.extend(rows)
        valid_stats["kept"] += 1

    logging.info(
        "train stats %s | valid stats %s",
        dict(train_stats),
        dict(valid_stats),
    )
    if not train_rows or not valid_rows:
        raise SystemExit(
            f"no rows after filtering: train={dict(train_stats)} valid={dict(valid_stats)}"
        )
    fields = (
        FIELDS + ["velocity_bucket", "duration_bucket"]
        if args.emit_buckets
        else FIELDS
    )
    write_csv(train_rows, args.out_train, fields)
    write_csv(valid_rows, args.out_valid, fields)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
