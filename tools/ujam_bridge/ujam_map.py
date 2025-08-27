from __future__ import annotations

"""Convert generic MIDI files into UJAM-ready MIDI with key switches."""

import argparse
import csv
import pathlib
from collections import Counter, defaultdict
from typing import Dict, List

import pretty_midi
try:  # optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore
import groove_profile as gp

from . import utils

PATTERN_PATH = pathlib.Path(__file__).with_name("patterns") / "strum_library.yaml"


def pattern_to_keyswitches(pattern: str, library: Dict[str, List[str]], keymap: Dict[str, int]) -> List[int]:
    """Resolve a space separated *pattern* to key switch MIDI notes."""
    names = library.get(pattern, [])
    return [keymap[n] for n in names if n in keymap]


def _parse_simple(text: str) -> Dict[str, Dict[str, List[str] | int | str]]:
    data: Dict[str, Dict[str, List[str] | int | str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.split('#')[0].strip()
        if not line:
            continue
        if line.endswith(':'):
            current = line[:-1]
            data[current] = {}
            continue
        key, val = [x.strip() for x in line.split(':', 1)]
        key = key.strip('"')
        if current:
            if val.startswith('['):
                val_list = [v.strip().strip('"') for v in val.strip('[]').split(',') if v.strip()]
                data[current][key] = val_list
            else:
                data[current][key] = int(val) if val.isdigit() else val.strip('"')
        else:
            data[key] = int(val) if val.isdigit() else val.strip('"')
    return data


def _load_yaml(path: pathlib.Path) -> Dict:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return _parse_simple(text)


def _load_patterns() -> Dict[str, List[str]]:
    data = _load_yaml(PATTERN_PATH)
    return data.get("patterns", {})  # type: ignore[return-value]


def _group_chords(notes: List[pretty_midi.Note], window: float = 0.03) -> List[List[pretty_midi.Note]]:
    blocks: List[List[pretty_midi.Note]] = []
    current: List[pretty_midi.Note] = []
    for n in sorted(notes, key=lambda x: x.start):
        if not current or n.start - current[0].start <= window:
            current.append(n)
        else:
            blocks.append(current)
            current = [n]
    if current:
        blocks.append(current)
    return blocks


def _load_sections(path: pathlib.Path) -> Dict[int, str]:
    """Load bar index to section name mapping from *path*."""
    data = _load_yaml(path)
    raw = data.get("sections", data)
    sections: Dict[int, str] = {}
    if isinstance(raw, list):
        for item in raw:
            start = int(item.get("bar", item.get("start", 0)))
            name = str(item.get("section", item.get("name", "")))
            sections[start] = name
    elif isinstance(raw, dict):
        for k, v in raw.items():
            try:
                sections[int(k)] = str(v)
            except Exception:
                continue
    return sections


def _section_for_bar(bar: int, sections: Dict[int, str]) -> str | None:
    current: str | None = None
    for start in sorted(sections):
        if bar >= start:
            current = sections[start]
        else:
            break
    return current


def convert(args: argparse.Namespace) -> None:
    mapping_path = pathlib.Path(args.mapping)
    mapping = _load_yaml(mapping_path)
    keymap: Dict[str, int] = mapping.get("keyswitch", {})
    play_low = int(mapping.get("play_range", {}).get("low", 0))
    play_high = int(mapping.get("play_range", {}).get("high", 127))
    section_styles: Dict[str, List[str]] = mapping.get("section_styles", {})

    pattern_lib = _load_patterns()

    sections: Dict[int, str] = {}
    if args.tags:
        sections = _load_sections(pathlib.Path(args.tags))

    pm = pretty_midi.PrettyMIDI(args.in_midi)
    utils.quantize(pm, int(args.quant), float(args.swing))
    if args.use_groove_profile:
        vocal = pretty_midi.PrettyMIDI(args.use_groove_profile)
        events = [{"offset": vocal.time_to_tick(n.start) / vocal.resolution} for n in vocal.instruments[0].notes]
        profile = gp.extract_groove_profile(events)  # type: ignore[arg-type]
        utils.apply_groove_profile(pm, profile, max_ms=35.0)
    utils.humanize(pm, float(args.humanize))

    instrument = pm.instruments[0]
    chord_blocks = _group_chords(instrument.notes)
    beats_per_bar = pm.time_signature_changes[0].numerator if pm.time_signature_changes else 4

    bar_blocks: Dict[int, List[Dict]] = defaultdict(list)
    for idx, block in enumerate(chord_blocks):
        start = block[0].start
        duration = min(n.end - n.start for n in block)
        strum = "D" if idx % 2 == 0 else "U"
        pitches = [n.pitch for n in block]
        beat = pm.time_to_tick(start) / pm.resolution
        bar = int(beat // beats_per_bar)
        bar_blocks[bar].append({"start": start, "duration": duration, "pitches": pitches, "strum": strum})

    out_pm = pretty_midi.PrettyMIDI(resolution=pm.resolution)
    ks_inst = pretty_midi.Instrument(program=0, name="Keyswitches")
    perf_inst = pretty_midi.Instrument(program=instrument.program, name="Performance")

    ks_hist: Counter[int] = Counter()
    clamp_notes = 0
    total_notes = 0
    approx: List[str] = []
    csv_rows: List[Dict[str, object]] = []

    for bar in sorted(bar_blocks):
        blocks = bar_blocks[bar]
        bar_start = pm.tick_to_time(bar * beats_per_bar * pm.resolution)
        if args.section_aware == "on" and sections:
            sec = _section_for_bar(bar, sections)
            if sec:
                for name in section_styles.get(sec, []):
                    pitch = keymap.get(name)
                    if pitch is not None:
                        ks_inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=bar_start, end=bar_start + 0.05))
                        ks_hist[pitch] += 1
        pattern = " ".join(b["strum"] for b in blocks)
        ks_notes = pattern_to_keyswitches(pattern, pattern_lib, keymap)
        if not ks_notes:
            approx.append(pattern)
            ks_notes = []
            for b in blocks:
                ks_notes.extend(pattern_to_keyswitches(b["strum"], pattern_lib, keymap))
        for pitch in ks_notes:
            ks_inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=bar_start, end=bar_start + 0.05))
            ks_hist[pitch] += 1
        for b in blocks:
            chord = utils.chordify(b["pitches"], (play_low, play_high))
            total_notes += len(b["pitches"])
            clamp_notes += sum(1 for p in b["pitches"] if p < play_low or p > play_high)
            for p in chord:
                perf_inst.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=b["start"], end=b["start"] + b["duration"]))
            if args.dry_run:
                csv_rows.append({"start": b["start"], "chord": chord, "keyswitches": ks_notes})

    if args.dry_run:
        csv_path = pathlib.Path(args.out_midi).with_suffix(".csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["start", "chord", "keyswitches"])
            for row in csv_rows:
                writer.writerow([f"{row['start']:.3f}", " ".join(map(str, row['chord'])), " ".join(map(str, row['keyswitches']))])
    else:
        out_pm.instruments.append(ks_inst)
        out_pm.instruments.append(perf_inst)
        out_pm.write(args.out_midi)

    if ks_hist:
        print("Keyswitch usage:")
        for pitch, count in sorted(ks_hist.items()):
            print(f"  {pitch}: {count}")
    if total_notes:
        pct = (clamp_notes / total_notes) * 100.0
        print(f"Clamped notes: {clamp_notes}/{total_notes} ({pct:.1f}%)")
    if approx:
        print("Approximate patterns for bars:", ", ".join(approx))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--in", dest="in_midi", required=True)
    parser.add_argument("--out", dest="out_midi", required=True)
    parser.add_argument("--tags")
    parser.add_argument("--section-aware", choices=["on", "off"], default="on")
    parser.add_argument("--quant", type=int, default=120)
    parser.add_argument("--swing", type=float, default=0.0)
    parser.add_argument("--humanize", type=float, default=0.0)
    parser.add_argument("--use-groove-profile")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":  # pragma: no cover
    main()
