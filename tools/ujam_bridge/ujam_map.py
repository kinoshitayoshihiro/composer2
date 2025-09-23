from __future__ import annotations

"""Convert generic MIDI files into UJAM-ready MIDI with key switches."""

import argparse
import csv
import math
import pathlib
from bisect import bisect_right
from collections import Counter, defaultdict
from types import SimpleNamespace
from typing import Dict, List, Sequence

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
from . import utils

__all__ = [
    "pattern_to_keyswitches",
    "load_map",
    "convert",
    "build_arg_parser",
    "main",
]

PATTERN_PATH = pathlib.Path(__file__).with_name("patterns") / "strum_library.yaml"
MAP_DIR = pathlib.Path(__file__).with_name("maps")
KS_MIN, KS_MAX = 36, 88



def convert(args: SimpleNamespace) -> None:
    """Convert a generic MIDI file into a UJAM-friendly arrangement."""

    if pretty_midi is None or mido is None:
        raise SystemExit("pretty_midi/mido not available")

    in_path = pathlib.Path(getattr(args, "in_midi"))
    out_path = pathlib.Path(getattr(args, "out_midi"))

    map_data: Dict[str, object] | None = None
    mapping_arg = getattr(args, "mapping", None)
    if mapping_arg:
        map_path = pathlib.Path(str(mapping_arg))
        if map_path.is_file():
            map_data = _load_yaml(map_path)

    if map_data is None:
        plugin = getattr(args, "plugin", None)
        if plugin:
            map_data = load_map(str(plugin))
        else:
            raise SystemExit("mapping path or plugin name required")

    problems = _validate_map(map_data)
    if problems:
        raise ValueError(", ".join(problems))

    keymap: Dict[str, int] = {}
    ks_entries = map_data.get("keyswitches", []) if isinstance(map_data, dict) else []
    if isinstance(ks_entries, list):
        for entry in ks_entries:
            if isinstance(entry, dict):
                name = entry.get("name")
                note = entry.get("note")
                if isinstance(name, str) and isinstance(note, int):
                    keymap[name] = int(note)

    ks_dict = map_data.get("keyswitch") if isinstance(map_data, dict) else None
    if isinstance(ks_dict, dict):
        for name, note in ks_dict.items():
            if isinstance(name, str) and isinstance(note, int):
                keymap[name] = int(note)

    for name, note in keymap.items():
        if not (KS_MIN <= int(note) <= KS_MAX):
            raise ValueError(
                f"keyswitch '{name}' out of range {KS_MIN}..{KS_MAX} (got {note})"
            )

    play = map_data.get("play_range", {}) if isinstance(map_data, dict) else {}
    try:
        play_low = int(play.get("low", KS_MIN))
    except Exception:
        play_low = KS_MIN
    try:
        play_high = int(play.get("high", KS_MAX))
    except Exception:
        play_high = KS_MAX
    play_low = max(0, min(127, play_low))
    play_high = max(play_low, min(127, play_high))

    section_styles = map_data.get("section_styles", {}) if isinstance(map_data, dict) else {}
    if not isinstance(section_styles, dict):
        section_styles = {}

    pattern_lib = _load_patterns()
    if not isinstance(pattern_lib, dict):
        pattern_lib = {}

    sections: Dict[int, str] = {}
    tags_arg = getattr(args, "tags", None)
    if tags_arg:
        tag_path = pathlib.Path(str(tags_arg))
        if tag_path.is_file():
            sections = _load_sections(tag_path)

    pm = pretty_midi.PrettyMIDI(str(in_path))
    try:
        times, tempi = pm.get_tempo_changes()
        times_seq = times.tolist() if hasattr(times, "tolist") else list(times)
        if len(tempi) == 0 or not (float(tempi[0]) > 0):
            scale = 60.0 / (120.0 * pm.resolution)
            pm._tick_scales = [(0, scale)]
            if hasattr(pm, "_update_tick_to_time"):
                pm._update_tick_to_time(pm.resolution)
        elif len(tempi) == 1:
            t0 = float(times_seq[0]) if times_seq else 0.0
            if t0 > 0.0 and getattr(pm, "_tick_scales", None) is None:
                scale = 60.0 / (float(tempi[0]) * pm.resolution)
                pm._tick_scales = [(0, scale)]
                if hasattr(pm, "_update_tick_to_time"):
                    pm._update_tick_to_time(pm.resolution)
    except Exception:
        pass

    utils.quantize(pm, int(getattr(args, "quant", 120)), float(getattr(args, "swing", 0.0)))

    ts_raw = list(getattr(pm, "time_signature_changes", []))
    if not ts_raw:
        ts_raw = [SimpleNamespace(numerator=4, denominator=4, time=0.0)]
    ts_raw = sorted(ts_raw, key=lambda ts: float(getattr(ts, "time", 0.0) or 0.0))

    def _beats_per_bar(ts_obj: SimpleNamespace) -> float:
        num = getattr(ts_obj, "numerator", 4)
        den = getattr(ts_obj, "denominator", 4)
        try:
            num = int(num)
        except Exception:
            num = 4
        try:
            den = int(den)
        except Exception:
            den = 4
        if den <= 0:
            den = 4
        beats = float(num) * (4.0 / float(den))
        return beats if beats > 0 else 4.0

    beats_per_bar = max(1, int(round(_beats_per_bar(ts_raw[0]))))

    groove_profile_path = getattr(args, "use_groove_profile", None)
    if groove_profile_path:
        vocal_pm = pretty_midi.PrettyMIDI(str(groove_profile_path))
        if vocal_pm.instruments:
            events = [
                {
                    "offset": vocal_pm.time_to_tick(n.start) / vocal_pm.resolution,
                }
                for n in vocal_pm.instruments[0].notes
            ]
            profile = gp.extract_groove_profile(events)  # type: ignore[arg-type]
            utils.apply_groove_profile(
                pm,
                profile,
                beats_per_bar=beats_per_bar,
                clip_head_ms=float(getattr(args, "groove_clip_head", 10.0)),
                clip_other_ms=float(getattr(args, "groove_clip_other", 35.0)),
            )

    utils.humanize(pm, float(getattr(args, "humanize", 0.0)))

    resolution = float(getattr(pm, "resolution", 480) or 480.0)

    def _time_to_beats(value: float) -> float:
        try:
            tick_val = pm.time_to_tick(value)
        except Exception:
            tick_val = value * resolution
        return float(tick_val) / resolution if resolution else float(value)

    segments: List[Dict[str, float]] = []
    prev_start_beat = 0.0
    prev_beats = _beats_per_bar(ts_raw[0])
    bar_offset = 0
    for idx, ts in enumerate(ts_raw):
        start_time = float(getattr(ts, "time", 0.0) or 0.0)
        start_beat = _time_to_beats(start_time)
        if idx > 0:
            beats_since_prev = max(0.0, start_beat - prev_start_beat)
            if prev_beats > 0:
                bar_offset += int(math.floor(beats_since_prev / prev_beats + 1e-9))
        beats_cur = max(_beats_per_bar(ts), 1.0)
        segments.append(
            {
                "start_time": start_time,
                "start_beat": start_beat,
                "beats_per_bar": beats_cur,
                "bar_offset": bar_offset,
            }
        )
        prev_start_beat = start_beat
        prev_beats = beats_cur
    if not segments:
        segments.append(
            {
                "start_time": 0.0,
                "start_beat": 0.0,
                "beats_per_bar": float(beats_per_bar),
                "bar_offset": 0.0,
            }
        )

    def _locate_bar(time: float) -> tuple[int, float, float, float]:
        beat_pos = _time_to_beats(time)
        segment = segments[0]
        for candidate in segments[1:]:
            if beat_pos + 1e-9 >= candidate["start_beat"]:
                segment = candidate
            else:
                break
        beats_in_bar = max(segment["beats_per_bar"], 1.0)
        rel_beats = max(0.0, beat_pos - segment["start_beat"])
        bar_in_segment = int(math.floor(rel_beats / beats_in_bar + 1e-9))
        bar_index = int(segment["bar_offset"]) + bar_in_segment
        bar_start_beat = segment["start_beat"] + bar_in_segment * beats_in_bar
        tick_value = bar_start_beat * resolution
        try:
            bar_start_time = pm.tick_to_time(tick_value)
        except Exception:
            bar_start_time = float(bar_start_beat)
        return bar_index, float(bar_start_time), bar_start_beat, beats_in_bar

    if not pm.instruments:
        raise ValueError("input MIDI has no instruments to convert")

    instrument = pm.instruments[0]
    chord_blocks = _group_chords(instrument.notes)

    bar_blocks: Dict[int, List[Dict]] = defaultdict(list)
    all_blocks: List[Dict] = []
    prev_beat: float | None = None
    prev_strum: str | None = None
    for block in chord_blocks:
        if not block:
            continue
        start = block[0].start
        duration = min(n.end - n.start for n in block)
        beat = _time_to_beats(start)
        bar, bar_start_time, bar_start_beat, _beats_in_bar = _locate_bar(start)
        beat_in_bar = beat - bar_start_beat
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
        next_start = all_blocks[i + 1]["start"] if i + 1 < len(all_blocks) else info["start"] + info["duration"]
        info["next_start"] = next_start
        bar_blocks[info["bar"]].append(info)

    out_pm = pretty_midi.PrettyMIDI(resolution=pm.resolution)
    ks_inst = pretty_midi.Instrument(program=0, name="Keyswitches")
    ks_channel = max(0, min(15, int(getattr(args, "ks_channel", 16)) - 1))
    ks_inst.midi_channel = ks_channel
    perf_inst = pretty_midi.Instrument(program=instrument.program, name="Performance")

    ks_hist: Counter[int] = Counter()
    clamp_notes = 0
    total_notes = 0
    approx: List[str] = []
    csv_rows: List[Dict[str, object]] = []
    last_sent: tuple[int, ...] | None = None
    last_sent_bar = -10**9
    emitted_at: dict[int, float] = {}
    bar_index = 0

    def _emit_keyswitch(pitch: int, when: float) -> bool:
        if no_redundant:
            prev = emitted_at.get(pitch)
            if prev is not None and abs(prev - when) <= 1e-6:
                return False
        ks_inst.notes.append(
            pretty_midi.Note(
                velocity=int(getattr(args, "ks_vel", 100)),
                pitch=pitch,
                start=when,
                end=when + 0.05,
            )
        )
        ks_hist[pitch] += 1
        emitted_at[pitch] = when
        return True

    try:
        periodic = int(getattr(args, "periodic_ks", 0))
    except Exception:
        periodic = 0
    ks_lead_ms = float(getattr(args, "ks_lead", 60.0))
    if not math.isfinite(ks_lead_ms):
        ks_lead_ms = 60.0
    ks_lead_ms = max(0.0, ks_lead_ms)
    lead_with_head_ms = ks_lead_ms + 20.0
    ks_headroom = float(getattr(args, "ks_headroom", 10.0)) / 1000.0
    section_mode = getattr(args, "section_aware", "off")
    no_redundant = bool(getattr(args, "no_redundant_ks", False))

    for bar in sorted(bar_blocks):
        blocks = bar_blocks[bar]
        bar_start = bar_starts[bar] if bar < len(bar_starts) else bar_starts[-1]
        pattern = " ".join(b["strum"] for b in blocks)
        ks_notes = pattern_to_keyswitches(pattern, pattern_lib, keymap)
        if not ks_notes:
            approx.append(pattern)
            ks_notes = []
            for b in blocks:
                ks_notes.extend(pattern_to_keyswitches(b["strum"], pattern_lib, keymap))
        ks_tuple = tuple(ks_notes)
        same_tuple = last_sent == ks_tuple
        if periodic > 0:
            periodic_due = (
                not same_tuple
                or last_sent_bar < 0
                or (bar_index - last_sent_bar) >= periodic
            )
        else:
            periodic_due = last_sent_bar < 0 or not same_tuple
        send = periodic_due
        if send and no_redundant and same_tuple and last_sent_bar == bar_index:
            send = False
        if send:
            bar_end = bar_start
            if blocks:
                for blk in blocks:
                    nxt = blk.get("next_start", blk["start"])
                    if nxt is None:
                        nxt = blk["start"]
                    try:
                        candidate = float(nxt)
                    except Exception:
                        candidate = float(blk["start"])
                    if candidate > bar_end:
                        bar_end = candidate
            bpm = _tempo_at(bar_end, tempo_times, tempo_values)
            lead_beats = (lead_with_head_ms / 1000.0) * (bpm / 60.0)
            ks_time = _ks_at_bar_end_with_headroom(bar_end, bpm, headroom_beats=lead_beats)
            def _record_emit() -> None:
                # Update guards immediately after emitting any KS event.
                nonlocal last_sent, last_sent_bar
                last_sent = ks_tuple
                last_sent_bar = bar_index
            if section_mode == "on" and sections:
                sec = _section_for_bar(bar, sections)
                if sec:
                    for name in section_styles.get(sec, []):
                        pitch = keymap.get(name)
                        if pitch is not None:
                            if _emit_keyswitch(pitch, ks_time):
                                _record_emit()
            for pitch in ks_notes:
                if _emit_keyswitch(pitch, ks_time):
                    _record_emit()
        for b in blocks:
            chord = utils.chordify(b["pitches"], (play_low, play_high))
            total_notes += len(b["pitches"])
            clamp_notes += sum(1 for p in b["pitches"] if p < play_low or p > play_high)
            end_time = b["start"] + b["duration"]
            next_start = b.get("next_start", end_time)
            if next_start - ks_headroom < end_time:
                end_time = next_start - ks_headroom
            if end_time < b["start"]:
                end_time = b["start"]
            for p in chord:
                perf_inst.notes.append(
                    pretty_midi.Note(velocity=100, pitch=p, start=b["start"], end=end_time)
                )
            if getattr(args, "dry_run", False):
                csv_rows.append({"start": b["start"], "chord": chord, "keyswitches": ks_notes})
        bar_index += 1

    if getattr(args, "dry_run", False):
        csv_path = out_path.with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["start", "chord", "keyswitches"])
            for row in csv_rows:
                writer.writerow(
                    [
                        f"{row['start']:.3f}",
                        " ".join(map(str, row["chord"])),
                        " ".join(map(str, row["keyswitches"])),
                    ]
                )
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_pm.instruments.append(ks_inst)
        out_pm.instruments.append(perf_inst)
        out_pm.write(str(out_path))
        if getattr(mido, "MidiFile", None) is not None:
            try:
                mf = mido.MidiFile(str(out_path))  # type: ignore[attr-defined]
                ks_track_index = None
                for idx, track in enumerate(mf.tracks):
                    track_name = None
                    for msg in track:
                        if msg.type == "track_name":
                            track_name = msg.name
                            break
                    if track_name == "Keyswitches":
                        ks_track_index = idx
                        break
                if ks_track_index is not None:
                    ks_track = mf.tracks[ks_track_index]
                    for msg in ks_track:
                        if msg.type in {"note_on", "note_off"}:
                            msg.channel = ks_channel
                    if ks_track_index != 0:
                        del mf.tracks[ks_track_index]
                        mf.tracks.insert(0, ks_track)
                    mf.save(str(out_path))
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
    range_issues: List[str] = []
    undefined_refs: List[str] = []
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
    play_defined = bool(play)
    try:
        low = int(play.get("low", KS_MIN))
        high = int(play.get("high", KS_MAX))
    except Exception:
        low, high = KS_MIN, KS_MAX
    play_low = min(low, high)
    play_high = max(low, high)
    play_low = max(0, min(play_low, 127))
    play_high = max(play_low, min(play_high, 127))
    ks_low = min(play_low, KS_MIN)
    ks_high = max(play_high, KS_MAX)

    for name, note in key_lookup.items():
        if note < ks_low or note > ks_high:
            range_issues.append(
                f"keyswitch '{name}' out of range {ks_low}..{ks_high} (got {note})"
            )
        elif play_defined and play_low <= note <= play_high:
            range_issues.append(
                f"keyswitch '{name}' overlaps play range {play_low}..{play_high} (got {note})"
            )

    section_styles = data.get("section_styles", {}) or {}
    if isinstance(section_styles, dict):
        for section, names_list in section_styles.items():
            if not isinstance(names_list, (list, tuple, set)):
                continue
            for name in names_list:
                if name not in key_lookup:
                    undefined_refs.append(
                        f"section '{section}' references undefined keyswitch '{name}'"
                    )
    issues.extend(undefined_refs)
    issues.extend(range_issues)
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


def _compute_bar_starts(pm: "pretty_midi.PrettyMIDI") -> List[float]:
    """Return bar start times accounting for time signature changes."""

    downbeats = list(pm.get_downbeats())
    if len(downbeats) >= 2:
        return [float(t) for t in downbeats]
    resolution = float(getattr(pm, "resolution", 480) or 480.0)
    ts_changes = list(getattr(pm, "time_signature_changes", []))
    if not ts_changes:
        ts_changes = [pretty_midi.TimeSignature(4, 4, 0.0)]
    ts_changes = sorted(ts_changes, key=lambda ts: float(getattr(ts, "time", 0.0) or 0.0))

    def _beats_per_bar(ts_obj: object) -> float:
        num = getattr(ts_obj, "numerator", 4)
        den = getattr(ts_obj, "denominator", 4)
        try:
            num = int(num)
        except Exception:
            num = 4
        try:
            den = int(den)
        except Exception:
            den = 4
        if den <= 0:
            den = 4
        beats = float(num) * (4.0 / float(den))
        return beats if beats > 0 else 4.0

    def _time_to_beats(value: float) -> float:
        try:
            tick_val = pm.time_to_tick(value)
        except Exception:
            tick_val = value * resolution
        return float(tick_val) / resolution if resolution else float(value)

    segments: List[Dict[str, float]] = []
    prev_start_beat = 0.0
    prev_beats = _beats_per_bar(ts_changes[0])
    bar_offset = 0
    for idx, ts in enumerate(ts_changes):
        start_time = float(getattr(ts, "time", 0.0) or 0.0)
        start_beat = _time_to_beats(start_time)
        if idx > 0:
            beats_since_prev = max(0.0, start_beat - prev_start_beat)
            if prev_beats > 0:
                bar_offset += int(math.floor(beats_since_prev / prev_beats + 1e-9))
        beats_cur = max(_beats_per_bar(ts), 1.0)
        segments.append(
            {
                "start_time": start_time,
                "start_beat": start_beat,
                "beats_per_bar": beats_cur,
                "bar_offset": bar_offset,
            }
        )
        prev_start_beat = start_beat
        prev_beats = beats_cur

    def _locate_bar_start(time: float) -> float:
        beat_pos = _time_to_beats(time)
        segment = segments[0]
        for candidate in segments[1:]:
            if beat_pos + 1e-9 >= candidate["start_beat"]:
                segment = candidate
            else:
                break
        beats_in_bar = max(segment["beats_per_bar"], 1.0)
        rel_beats = max(0.0, beat_pos - segment["start_beat"])
        bar_in_segment = int(math.floor(rel_beats / beats_in_bar + 1e-9))
        bar_start_beat = segment["start_beat"] + bar_in_segment * beats_in_bar
        tick_val = bar_start_beat * resolution
        try:
            return float(pm.tick_to_time(tick_val))
        except Exception:
            return float(bar_start_beat)

    bar_times = {0.0}
    for inst in getattr(pm, "instruments", []):
        for note in getattr(inst, "notes", []):
            bar_times.add(_locate_bar_start(float(getattr(note, "start", 0.0))))
    for ts in ts_changes:
        bar_times.add(_locate_bar_start(float(getattr(ts, "time", 0.0) or 0.0)))
    return sorted(bar_times)


def _tempo_at(time: float, tempo_times: Sequence[float], tempo_values: Sequence[float]) -> float:
    idx = bisect_right(tempo_times, time) - 1
    if idx < 0:
        idx = 0
    if idx >= len(tempo_values):
        idx = len(tempo_values) - 1
    bpm = float(tempo_values[idx]) if tempo_values else 120.0
    if not math.isfinite(bpm) or bpm <= 0.0:
        return 120.0
    return bpm


def _ks_at_bar_end_with_headroom(bar_end: float, bpm: float, headroom_beats: float = 0.08) -> float:
    if not math.isfinite(bar_end):
        return 0.0
    if not math.isfinite(bpm) or bpm <= 0.0:
        return max(0.0, bar_end)
    headroom = max(0.0, headroom_beats)
    return max(0.0, bar_end - (60.0 / bpm) * headroom)


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


def convert(args) -> None:
    if pretty_midi is None:
        raise RuntimeError("pretty_midi is required for conversion")
    mapping_arg = getattr(args, "mapping", "")
    mapping_path = pathlib.Path(mapping_arg)
    mapping_data: Dict
    if mapping_path.is_file():
        mapping_data = _load_yaml(mapping_path)
        issues = _validate_map(mapping_data)
        if issues:
            raise ValueError(", ".join(issues))
    else:
        candidate_name = mapping_path.name if mapping_path.suffix else f"{mapping_path.name}.yaml"
        candidate = MAP_DIR / candidate_name
        if candidate.is_file():
            mapping_data = _load_yaml(candidate)
            issues = _validate_map(mapping_data)
            if issues:
                raise ValueError(", ".join(issues))
        else:
            mapping_key = mapping_path.stem if mapping_path.suffix else mapping_path.name
            mapping_data = load_map(mapping_key)
    keymap: Dict[str, int] = {}
    for item in mapping_data.get("keyswitches", []) or []:
        if isinstance(item, dict):
            name = item.get("name")
            note = item.get("note")
            if isinstance(name, str) and isinstance(note, int):
                keymap[name] = int(note)
    extra = mapping_data.get("keyswitch")
    if isinstance(extra, dict):
        for name, note in extra.items():
            if isinstance(name, str) and isinstance(note, int):
                keymap[name] = int(note)
    if not keymap:
        raise ValueError("mapping did not define any keyswitch notes")
    pattern_lib = _load_patterns()
    section_styles = mapping_data.get("section_styles", {}) or {}
    play_cfg = mapping_data.get("play_range", {}) or {}
    try:
        play_low = int(play_cfg.get("low", KS_MIN))
        play_high = int(play_cfg.get("high", KS_MAX))
    except Exception:
        play_low, play_high = KS_MIN, KS_MAX
    sections: Dict[int, str] = {}
    tags_path = getattr(args, "tags", None)
    if tags_path:
        tag_path = pathlib.Path(tags_path)
        if tag_path.is_file():
            sections = _load_sections(tag_path)
    pm = pretty_midi.PrettyMIDI(str(args.in_midi))
    # 初期テンポ注入（無テンポや非正テンポの救済）
    try:
        # pretty_midi.get_tempo_changes() は (times, tempi) を返す
        times, tempi = pm.get_tempo_changes()
        if len(tempi) == 0 or not (float(tempi[0]) > 0):
            # 120 BPM を仮定して tick→sec のスケールを初期化
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

    try:
        tempo_times_arr, tempo_values_arr = pm.get_tempo_changes()
    except Exception:
        tempo_times_arr, tempo_values_arr = ([], [])
    tempo_times = [float(t) for t in tempo_times_arr] if len(tempo_times_arr) else [0.0]
    tempo_values = [float(v) for v in tempo_values_arr] if len(tempo_values_arr) else [120.0]

    bar_starts = _compute_bar_starts(pm)

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
        bar = max(0, bisect_right(bar_starts, start) - 1)
        if bar >= len(bar_starts):
            bar = len(bar_starts) - 1
        bar_start_tick = pm.time_to_tick(bar_starts[bar])
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
    last_sent: tuple[int, ...] | None = None
    last_sent_bar = -10**9
    emitted_at: dict[int, float] = {}
    bar_index = 0

    def _emit_keyswitch(pitch: int, when: float) -> bool:
        if args.no_redundant_ks:
            prev = emitted_at.get(pitch)
            if prev is not None and abs(prev - when) <= 1e-6:
                return False
        ks_inst.notes.append(
            pretty_midi.Note(
                velocity=int(args.ks_vel),
                pitch=pitch,
                start=when,
                end=when + 0.05,
            )
        )
        ks_hist[pitch] += 1
        emitted_at[pitch] = when
        return True

    for bar in sorted(bar_blocks):
        blocks = bar_blocks[bar]
        bar_start = bar_starts[bar] if bar < len(bar_starts) else bar_starts[-1]
        pattern = " ".join(b["strum"] for b in blocks)
        ks_notes = pattern_to_keyswitches(pattern, pattern_lib, keymap)
        if not ks_notes:
            approx.append(pattern)
            ks_notes = []
            for b in blocks:
                ks_notes.extend(pattern_to_keyswitches(b["strum"], pattern_lib, keymap))
        ks_tuple = tuple(ks_notes)
        periodic = max(0, int(args.periodic_ks))
        # periodic == 0 means the guard re-arms every bar.
        same_tuple = last_sent == ks_tuple
        due_periodic = (
            periodic == 0
            or not same_tuple
            or last_sent_bar < 0
            or (bar_index - last_sent_bar) >= periodic
        )
        send = due_periodic
        if send and args.no_redundant_ks and same_tuple and last_sent_bar == bar_index:
            send = False
        if send:
            ks_time = max(0.0, bar_start - (float(args.ks_lead) + 20.0) / 1000.0)
            emitted_any = False
            if args.section_aware == "on" and sections:
                sec = _section_for_bar(bar, sections)
                if sec:
                    for name in section_styles.get(sec, []):
                        pitch = keymap.get(name)
                        if pitch is not None:
                            emitted_any |= _emit_keyswitch(pitch, ks_time)
            for pitch in ks_notes:
                emitted_any |= _emit_keyswitch(pitch, ks_time)
            if emitted_any:
                # Track the last tuple only when a keyswitch actually went out.
                last_sent = ks_tuple
                last_sent_bar = bar_index
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
                ks_track_index = None
                for idx, track in enumerate(mf.tracks):
                    track_name = None
                    for msg in track:
                        if msg.type == "track_name":
                            track_name = msg.name
                            break
                    if track_name == "Keyswitches":
                        ks_track_index = idx
                        break
                if ks_track_index is not None:
                    ks_track = mf.tracks[ks_track_index]
                    for msg in ks_track:
                        if msg.type in {"note_on", "note_off"}:
                            msg.channel = ks_ch
                    if ks_track_index != 0:
                        del mf.tracks[ks_track_index]
                        mf.tracks.insert(0, ks_track)
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
