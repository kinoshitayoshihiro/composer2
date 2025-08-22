from __future__ import annotations

"""Convert MIDI or corpus samples to phrase CSVs."""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable, Iterator

import pretty_midi
import yaml


FIELDS = ["pitch", "velocity", "duration", "pos", "boundary", "bar", "instrument"]


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
) -> list[dict[str, object]]:
    """Parse *path* into phrase rows using simple boundary heuristics."""

    pm = pretty_midi.PrettyMIDI(str(path))
    ticks_per_beat = pm.resolution
    notes: list[tuple[float, float, int, int, str]] = []
    for inst in pm.instruments:
        name = inst.name or pretty_midi.program_to_instrument_name(inst.program)
        if instrument_filter and instrument_filter not in name.lower():
            continue
        for n in inst.notes:
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
        rows.append(
            {
                "pitch": pitch,
                "velocity": vel,
                "duration": dur,
                "pos": pos,
                "boundary": boundary,
                "bar": bar,
                "instrument": name,
            }
        )
        prev_start = start
        prev_bar = bar
        prev_section = cur_section
    return rows


def write_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_samples_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def corpus_mode(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    def load_split(split: str) -> list[dict[str, object]]:
        base = root / split
        files: list[Path] = []
        if (base / "samples.jsonl").is_file():
            files = [base / "samples.jsonl"]
        else:
            files = sorted((base / "samples").glob("*.jsonl"))
        rows: list[dict[str, object]] = []
        for p in files:
            for obj in read_samples_jsonl(p):
                rows.append(
                    {
                        "pitch": obj["pitch"],
                        "velocity": obj.get("velocity", 0),
                        "duration": obj.get("duration", 0),
                        "pos": obj.get("pos", 0),
                        "boundary": obj.get("boundary", 0),
                        "bar": obj.get("bar", 0),
                        "instrument": obj.get("instrument", ""),
                    }
                )
        return rows

    return load_split("train"), load_split("valid")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", type=Path)
    parser.add_argument("--out-train", type=Path, required=True)
    parser.add_argument("--out-valid", type=Path, required=True)
    parser.add_argument("--instrument")
    parser.add_argument("--boundary-gap-beats", type=float, default=0.5)
    parser.add_argument("--boundary-on-section-change", action="store_true")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-notes", type=int, default=0)
    parser.add_argument("--max-bars", type=int, default=10**9)
    parser.add_argument("--from-corpus", dest="from_corpus", type=Path)
    args = parser.parse_args(argv)

    if args.from_corpus:
        train_rows, valid_rows = corpus_mode(args.from_corpus)
        write_csv(train_rows, args.out_train)
        write_csv(valid_rows, args.out_valid)
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
    for p in train_files:
        sec = tags.get(p.name)
        rows = midi_to_rows(
            p,
            args.boundary_gap_beats,
            sec,
            args.instrument.lower() if args.instrument else None,
        )
        if rows and len(rows) >= args.min_notes and (rows[-1]["bar"] + 1) <= args.max_bars:
            train_rows.extend(rows)
    valid_rows: list[dict[str, object]] = []
    for p in valid_files:
        sec = tags.get(p.name)
        rows = midi_to_rows(
            p,
            args.boundary_gap_beats,
            sec,
            args.instrument.lower() if args.instrument else None,
        )
        if rows and len(rows) >= args.min_notes and (rows[-1]["bar"] + 1) <= args.max_bars:
            valid_rows.extend(rows)
    write_csv(train_rows, args.out_train)
    write_csv(valid_rows, args.out_valid)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
