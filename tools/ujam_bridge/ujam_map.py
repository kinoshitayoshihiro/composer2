from __future__ import annotations

"""Convert generic MIDI files into UJAM-ready MIDI with key switches."""

import argparse
import csv
import pathlib
from bisect import bisect_right
from collections import Counter, defaultdict
from typing import Dict, List

try:  # optional dependency for writing MIDI; re-export for tests monkeypatching
    import mido  # type: ignore
except Exception:  # pragma: no cover - allow monkeypatch in tests
    mido = None  # type: ignore[assignment]

try:  # optional dependencies
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover
    pretty_midi = None  # type: ignore
try:  # optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore
import groove_profile as gp


PATTERN_PATH = pathlib.Path(__file__).with_name("patterns") / "strum_library.yaml"
MAP_DIR = pathlib.Path(__file__).with_name("maps")
KS_MIN, KS_MAX = 36, 88


def pattern_to_keyswitches(pattern: str, library: Dict[str, List[str]], keymap: Dict[str, int]) -> List[int]:
    """Resolve a space separated *pattern* to key switch MIDI notes."""
    names = library.get(pattern, [])
    return [keymap[n] for n in names if n in keymap]


def _parse_simple(text: str) -> Dict:
    """Very small YAML subset parser for map files."""
    lines = [l.rstrip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    data: Dict[str, object] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("keyswitches:"):
            i += 1
            items: List[Dict[str, object]] = []
            while i < len(lines) and lines[i].startswith("  -"):
                item: Dict[str, object] = {}
                first = lines[i][3:]
                if first:
                    k, v = first.split(":", 1)
                    item[k.strip()] = _coerce(v.strip())
                i += 1
                while i < len(lines) and lines[i].startswith("    "):
                    k, v = lines[i].strip().split(":", 1)
                    item[k.strip()] = _coerce(v.strip())
                    i += 1
                items.append(item)
            data["keyswitches"] = items
        else:
            k, v = line.split(":", 1)
            data[k.strip()] = _coerce(v.strip())
            i += 1
    return data


def _coerce(val: str) -> object:
    if val.lower() in {"true", "false"}:
        return val.lower() == "true"
    try:
        return int(val)
    except ValueError:
        return val.strip('"')


def _load_yaml(path: pathlib.Path) -> Dict:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return _parse_simple(text)


def _validate_map(data: Dict) -> List[str]:
    """Return a list of problems found in *data* mapping."""
    issues: List[str] = []
    key_lookup: dict[str, int] = {}
    names: set[str] = set()
    notes: set[int] = set()

    def _register(name: str | None, note: object, *, require_hold: bool = False, hold=None) -> None:
        if not isinstance(name, str):
            issues.append("keyswitch missing name")
            return
        if name in names:
            issues.append(f"duplicate name '{name}'")
        names.add(name)
        if not isinstance(note, int):
            issues.append(f"keyswitch '{name}' missing note")
            return
        if note in notes:
            issues.append(f"duplicate note {note}")
        notes.add(note)
        if not 0 <= note <= 127:
            issues.append(f"keyswitch '{name}' out of range 0..127 (got {note})")
        key_lookup[name] = note
        if require_hold and hold is None:
            issues.append(f"keyswitch '{name}' missing hold")

    ks_list = data.get("keyswitches", []) or []
    for idx, item in enumerate(ks_list):
        if not isinstance(item, dict):
            issues.append(f"keyswitch #{idx} not a mapping")
            continue
        _register(item.get("name"), item.get("note"), require_hold=True, hold=item.get("hold"))

    ks_dict = data.get("keyswitch")
    if isinstance(ks_dict, dict):
        for name, value in ks_dict.items():
            _register(name, value, require_hold=False)

    play = data.get("play_range", {}) or {}
    try:
        low = int(play.get("low", KS_MIN))
        high = int(play.get("high", KS_MAX))
    except Exception:
        low, high = KS_MIN, KS_MAX
    ks_low = min(low, KS_MIN)
    ks_high = max(high, KS_MAX)

    for name, note in key_lookup.items():
        if note < ks_low or note > ks_high:
            issues.append(
                f"keyswitch '{name}' out of range {ks_low}..{ks_high} (got {note})"
            )

    section_styles = data.get("section_styles", {}) or {}
    if isinstance(section_styles, dict):
        for section, names_list in section_styles.items():
            if not isinstance(names_list, (list, tuple, set)):
                continue
            for name in names_list:
                if name not in key_lookup:
                    issues.append(
                        f"section '{section}' references undefined keyswitch '{name}'"
                    )
    return issues


def load_map(product: str) -> Dict:
    """Load *product* YAML map from :data:`MAP_DIR` and validate."""
    path = MAP_DIR / f"{product}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"mapping not found: {product}")
    data = _load_yaml(path)
    problems = _validate_map(data)
    if problems:
        raise ValueError(", ".join(problems))
    return data


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
    from . import utils
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

    pm = pretty_midi.PrettyMIDI(str(args.in_midi))
    # 初期テンポ注入（無テンポや非正テンポの救済）
    try:
        _times, bpms = pm.get_tempo_changes()
        if len(bpms) == 0 or not float(bpms[0]) > 0:
            scale = 60.0 / (120.0 * pm.resolution)
            pm._tick_scales = [(0, scale)]
            if hasattr(pm, "_update_tick_to_time"):
                pm._update_tick_to_time(pm.resolution)
    except Exception:
        pass
    utils.quantize(pm, int(args.quant), float(args.swing))
    beats_per_bar = pm.time_signature_changes[0].numerator if pm.time_signature_changes else 4
    if args.use_groove_profile:
        vocal = pretty_midi.PrettyMIDI(str(args.use_groove_profile))
        events = [{"offset": vocal.time_to_tick(n.start) / vocal.resolution} for n in vocal.instruments[0].notes]
        profile = gp.extract_groove_profile(events)  # type: ignore[arg-type]
        utils.apply_groove_profile(
            pm,
            profile,
            beats_per_bar=beats_per_bar,
            clip_head_ms=float(args.groove_clip_head),
            clip_other_ms=float(args.groove_clip_other),
        )
    utils.humanize(pm, float(args.humanize))
    
    downbeats = pm.get_downbeats()
    ts_changes = pm.time_signature_changes
    ts_idx = 0
    bar_beats: Dict[int, int] = {}
    for i, db in enumerate(downbeats):
        while ts_idx + 1 < len(ts_changes) and ts_changes[ts_idx + 1].time <= db:
            ts_idx += 1
        bar_beats[i] = ts_changes[ts_idx].numerator if ts_changes else beats_per_bar

    instrument = pm.instruments[0]
    chord_blocks = _group_chords(instrument.notes)

    bar_blocks: Dict[int, List[Dict]] = defaultdict(list)
    all_blocks: List[Dict] = []
    prev_beat: float | None = None
    prev_strum: str | None = None
    for block in chord_blocks:
        start = block[0].start
        duration = min(n.end - n.start for n in block)
        beat = pm.time_to_tick(start) / pm.resolution
        bar = max(0, bisect_right(downbeats, start) - 1)
        bar_start_tick = pm.time_to_tick(downbeats[bar])
        beat_in_bar = (pm.time_to_tick(start) - bar_start_tick) / pm.resolution
        if prev_beat is None or beat_in_bar < 0.1 or (beat - prev_beat) > 1.5:
            strum = "D"
        else:
            strum = "U" if prev_strum != "U" else "D"
        pitches = [n.pitch for n in block]
        info = {
            "start": start,
            "duration": duration,
            "pitches": pitches,
            "strum": strum,
            "beat": beat,
            "bar": bar,
        }
        all_blocks.append(info)
        prev_beat = beat
        prev_strum = strum

    for i, info in enumerate(all_blocks):
        info["next_start"] = all_blocks[i + 1]["start"] if i + 1 < len(all_blocks) else info["start"] + info["duration"]
        bar_blocks[info["bar"]].append(info)

    out_pm = pretty_midi.PrettyMIDI(resolution=pm.resolution)
    ks_inst = pretty_midi.Instrument(program=0, name="Keyswitches")
    ks_inst.midi_channel = int(args.ks_channel) - 1
    perf_inst = pretty_midi.Instrument(program=instrument.program, name="Performance")

    ks_hist: Counter[int] = Counter()
    clamp_notes = 0
    total_notes = 0
    approx: List[str] = []
    csv_rows: List[Dict[str, object]] = []
    last_ks: List[int] | None = None
    bar_index = 0

    for bar in sorted(bar_blocks):
        blocks = bar_blocks[bar]
        bar_start = downbeats[bar]
        pattern = " ".join(b["strum"] for b in blocks)
        ks_notes = pattern_to_keyswitches(pattern, pattern_lib, keymap)
        if not ks_notes:
            approx.append(pattern)
            ks_notes = []
            for b in blocks:
                ks_notes.extend(pattern_to_keyswitches(b["strum"], pattern_lib, keymap))
        send = True
        if args.no_redundant_ks and last_ks == ks_notes:
            send = False
        if args.periodic_ks and bar_index % int(args.periodic_ks) == 0:
            send = True
        if send:
            ks_time = max(0.0, bar_start - (float(args.ks_lead) + 20.0) / 1000.0)
            sec: str | None = None
            if args.section_aware == "on" and sections:
                sec = _section_for_bar(bar, sections)
                if sec:
                    for name in section_styles.get(sec, []):
                        pitch = keymap.get(name)
                        if pitch is not None:
                            ks_inst.notes.append(
                                pretty_midi.Note(velocity=int(args.ks_vel), pitch=pitch, start=ks_time, end=ks_time + 0.05)
                            )
                            ks_hist[pitch] += 1
            for pitch in ks_notes:
                ks_inst.notes.append(
                    pretty_midi.Note(velocity=int(args.ks_vel), pitch=pitch, start=ks_time, end=ks_time + 0.05)
                )
                ks_hist[pitch] += 1
        last_ks = ks_notes
        for b in blocks:
            chord = utils.chordify(b["pitches"], (play_low, play_high))
            total_notes += len(b["pitches"])
            clamp_notes += sum(1 for p in b["pitches"] if p < play_low or p > play_high)
            end_time = b["start"] + b["duration"]
            next_start = b.get("next_start", end_time)
            head = float(args.ks_headroom) / 1000.0
            if next_start - head < end_time:
                end_time = next_start - head
            if end_time < b["start"]:
                end_time = b["start"]
            for p in chord:
                perf_inst.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=b["start"], end=end_time))
            if args.dry_run:
                csv_rows.append({"start": b["start"], "chord": chord, "keyswitches": ks_notes})
        bar_index += 1

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
        out_pm.write(str(args.out_midi))
        if getattr(mido, "MidiFile", None) is not None:
            try:
                mf = mido.MidiFile(str(args.out_midi))  # type: ignore[attr-defined]
                ks_ch = int(args.ks_channel) - 1
                for track in mf.tracks:
                    name = None
                    for msg in track:
                        if msg.type == "track_name":
                            name = msg.name
                        elif name == "Keyswitches" and msg.type in {"note_on", "note_off"}:
                            msg.channel = ks_ch
                mf.save(str(args.out_midi))
            except Exception:
                pass

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
    parser.add_argument("--ks-lead", type=float, default=60.0)
    parser.add_argument("--no-redundant-ks", action="store_true")
    parser.add_argument("--periodic-ks", type=int, default=0)
    parser.add_argument("--ks-headroom", type=float, default=10.0)
    parser.add_argument("--ks-channel", type=int, default=16)
    parser.add_argument("--ks-vel", type=int, default=100)
    parser.add_argument("--groove-clip-head", type=float, default=10.0)
    parser.add_argument("--groove-clip-other", type=float, default=35.0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":  # pragma: no cover
    main()
