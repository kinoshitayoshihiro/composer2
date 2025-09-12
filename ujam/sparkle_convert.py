
#!/usr/bin/env python3
# sparkle_convert.py — Convert generic MIDI to UJAM Sparkle-friendly MIDI.
#
# Features
# - Reads an input MIDI and (optionally) a chord CSV/YAML timeline.
# - Emits:
#   1) Long chord notes (triads) in Sparkle’s "Chord" range (configurable octave).
#   2) Steady "common pulse" keys (left-hand phrase trigger) at a chosen subdivision.
#
# Assumptions / Notes
# - UJAM Virtual Guitarist (Sparkle) uses "left hand" keys to trigger patterns/phrases
#   and "chord area" notes to define the chord. Exact note ranges may vary by version
#   and preset. Therefore, ALL layout values are configurable via a mapping YAML.
# - Default time signature is 4/4 if not provided. Tempo is read from the file if present,
#   otherwise --bpm is used.
# - If no chord timeline is provided, a lightweight heuristic infers major/minor triads
#   by bar from active pitch classes.
#
# CLI
#     python sparkle_convert.py IN.mid --out OUT.mid \
#         --pulse 1/8 --chord-octave 4 --phrase-note 36 \
#         --mapping sparkle_mapping.yaml
#
#     # With explicit chord timeline (CSV):
#     python sparkle_convert.py IN.mid --out OUT.mid --pulse 1/8 \
#         --chords chords.csv
#
# Chord CSV format (times in seconds; headers required):
#     start,end,root,quality
#     0.0,2.0,C,maj
#     2.0,4.0,A,min
# Supported qualities: maj, min (others are passed through if triad mapping provided).
#
# Mapping YAML example is created alongside this script as 'sparkle_mapping.example.yaml'.
#
# (c) 2025 — Utility script for MIDI preprocessing. MIT License.

import argparse
import csv
import re
import bisect
import math
import random
import logging
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union

try:
    import pretty_midi  # type: ignore
except Exception as e:
    raise SystemExit("This tool requires pretty_midi. Try: pip install pretty_midi") from e

try:
    import yaml  # optional for mapping file
except Exception:
    yaml = None

PITCH_CLASS = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
    'E#': 5, 'F': 5, 'F#': 6, 'Gb': 6, 'Fb': 4,
    'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11, 'Cb': 11, 'B#': 0
}

NOTE_RE = re.compile(r'^([A-G][b#]?)(-?\d+)$')

EPS = 1e-9


def parse_midi_note(token: str) -> int:
    """Parse a MIDI note from integer or note name like C1 or F#2."""
    token = token.strip().replace('＃', '#').replace('♯', '#').replace('♭', 'b').replace('ｂ', 'b')
    fw = {chr(ord('Ａ') + i): chr(ord('A') + i) for i in range(26)}
    fw.update({chr(ord('０') + i): str(i) for i in range(10)})
    token = token.translate(str.maketrans(fw))
    try:
        v = int(token)
        if not (0 <= v <= 127):
            raise ValueError
        return v
    except ValueError:
        m = NOTE_RE.match(token)
        if not m:
            raise SystemExit(f"Invalid note token: {token} (use int 0-127 or names like C1)")
        name, octv = m.groups()
        pc = PITCH_CLASS.get(name)
        if pc is None:
            raise SystemExit(f"Unknown pitch class: {name}")
        val = (int(octv) + 1) * 12 + pc
        if not (0 <= val <= 127):
            raise SystemExit(f"Note out of MIDI range: {token}")
        return val


def validate_midi_note(v: int) -> int:
    """Validate MIDI note range 0..127."""
    v = int(v)
    if not (0 <= v <= 127):
        raise SystemExit(
            "cycle_phrase_notes must be MIDI 0..127, 'rest', or note names"
        )
    return v


def clip_note_interval(start_t: float, end_t: float, *, eps: float = 1e-4) -> Tuple[float, float]:
    """Ensure end_t is at least eps after start_t and clamp negatives."""
    if start_t < 0:
        start_t = 0.0
    if end_t < start_t + eps:
        end_t = start_t + eps
    return start_t, end_t


def validate_accent(accent: Optional[List[Union[int, float]]]) -> Optional[List[float]]:
    if accent is None:
        return None
    if not isinstance(accent, list):
        raise SystemExit("--accent must be JSON list of numbers")
    if not accent:
        return None
    cleaned: List[float] = []
    for x in accent:
        if not isinstance(x, (int, float)):
            raise SystemExit("--accent must be JSON list of numbers")
        v = float(x)
        if v < 0.1:
            v = 0.1
        if v > 2.0:
            v = 2.0
        cleaned.append(v)
    return cleaned


def parse_accent_arg(s: str) -> Optional[List[float]]:
    try:
        data = json.loads(s)
    except Exception:
        raise SystemExit("--accent must be JSON list of numbers")
    return validate_accent(data)


def _append_phrase(inst, pitch: int, start: float, end: float, vel: int,
                   merge_gap_sec: float, release_sec: float, min_len_sec: float):
    if end <= start + EPS:
        return
    end -= release_sec
    start, end = clip_note_interval(start, end, eps=EPS)
    if min_len_sec > 0.0 and end < start + min_len_sec - EPS:
        end = start + min_len_sec
    if inst.notes and inst.notes[-1].pitch == pitch and (start - inst.notes[-1].end) <= merge_gap_sec + EPS:
        inst.notes[-1].end = max(inst.notes[-1].end, end)
    else:
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))


def _legato_merge_chords(inst, merge_gap_sec: float):
    last_by_pitch = {}
    merged = []
    for n in sorted(inst.notes, key=lambda x: (x.pitch, x.start)):
        prev = last_by_pitch.get(n.pitch)
        if prev and (n.start - prev.end) <= merge_gap_sec + EPS:
            prev.end = max(prev.end, n.end)
        else:
            merged.append(n)
            last_by_pitch[n.pitch] = n
    inst.notes = sorted(merged, key=lambda x: x.start)

@dataclass
class ChordSpan:
    start: float
    end: float
    root_pc: int  # 0-11
    quality: str  # 'maj' or 'min' (extendable)

def parse_time_sig(default_num=4, default_den=4) -> Tuple[int,int]:
    # pretty_midi doesn't store TS per track reliably; keep configurable if needed
    return default_num, default_den

def parse_pulse(s: str) -> float:
    '''
    Parse a subdivision string like '1/8' -> 0.5 beats (if a beat is a quarter note).
    We define '1/8' as eighth-notes = 0.5 quarter-beats.
    '''
    s = s.strip()
    if '/' in s:
        num, den = s.split('/', 1)
        num = int(num)
        den = int(den)
        if num != 1:
            raise ValueError("Use forms like 1/8, 1/16, 1/4.")
        # relative to quarter-note = 1 beat
        return 4.0 / den
    else:
        # numeric beats directly
        return float(s)

def triad_pitches(root_pc: int, quality: str, octave: int, mapping: Dict) -> List[int]:
    '''Return MIDI numbers for a simple triad in the given octave based on mapping intervals.'''
    intervals = mapping.get('triad_intervals', {}).get(quality, [0,4,7])  # default maj
    base_c = (octave + 1) * 12  # C-octave base
    return [base_c + ((root_pc + iv) % 12) for iv in intervals]

def place_in_range(pitches: List[int], lo: int, hi: int, *, voicing_mode: str = 'stacked') -> List[int]:
    res: List[int] = []
    prev: Optional[int] = None
    if voicing_mode == 'closed':
        for p in pitches:
            while p < lo:
                p += 12
            while p > hi:
                p -= 12
            res.append(p)
        res.sort()
        changed = True
        while changed:
            changed = False
            res.sort()
            for i in range(1, len(res)):
                while res[i] - res[i-1] > 12 and res[i] - 12 >= lo:
                    res[i] -= 12
                    changed = True
        for i in range(len(res)):
            while res[i] > hi and res[i] - 12 >= lo:
                res[i] -= 12
        res.sort()
        return res

    for p in pitches:
        while p < lo:
            p += 12
        while p > hi:
            p -= 12
        if prev is not None:
            while p <= prev:
                p += 12
        prev = p
        res.append(p)
    if res and res[-1] > hi:
        while any(p > hi for p in res) and all(p - 12 >= lo for p in res):
            res = [p - 12 for p in res]
        if any(p > hi for p in res):
            logging.warning("place_in_range: notes fall outside range %s-%s", lo, hi)
    return res

def load_mapping(path: Optional[Path]) -> Dict:
    default = {
        "phrase_note": 36,          # Default left-hand "Common" phrase key (C2)
        "phrase_velocity": 96,
        "phrase_length_beats": 0.25,
        "phrase_hold": "off",
        "phrase_merge_gap": 0.02,
        "chord_merge_gap": 0.01,
        "chord_octave": 4,          # Place chord tones around C4-B4 by default
        "chord_velocity": 90,
        "triad_intervals": {
            "maj": [0,4,7],
            "min": [0,3,7]
        },
        "cycle_phrase_notes": [],  # e.g., [24, 26] to alternate per bar
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
        "chord_input_range": None,
        "voicing_mode": "stacked",
        "top_note_max": None,
        "phrase_channel": None,
        "chord_channel": None,
        "cycle_stride": 1,
        "accent": None,
        "silent_qualities": [],
        "clone_meta_only": False,
        "strict": False,
    }
    if path is None:
        return default
    if yaml is None:
        raise SystemExit("PyYAML is required to read mapping files. pip install pyyaml")
    data = yaml.safe_load(Path(path).read_text())
    default.update(data or {})
    rng = default.get("chord_input_range")
    if rng is not None:
        try:
            lo = int(rng.get("lo"))
            hi = int(rng.get("hi"))
        except Exception:
            raise SystemExit("chord_input_range must have integer lo/hi")
        if not (0 <= lo <= 127 and 0 <= hi <= 127 and lo <= hi):
            raise SystemExit("chord_input_range lo/hi must be 0..127 and lo<=hi")
        default["chord_input_range"] = {"lo": lo, "hi": hi}
    top = default.get("top_note_max")
    if top is not None:
        try:
            top_i = int(top)
        except Exception:
            raise SystemExit("top_note_max must be int")
        if not (0 <= top_i <= 127):
            raise SystemExit("top_note_max must be 0..127")
        default["top_note_max"] = top_i
    for key in ("phrase_channel", "chord_channel"):
        ch = default.get(key)
        if ch is not None:
            try:
                ch_i = int(ch)
            except Exception:
                raise SystemExit(f"{key} must be int 0..15")
            if not (0 <= ch_i <= 15):
                raise SystemExit(f"{key} must be 0..15")
            default[key] = ch_i
    cs = default.get("cycle_stride", 1)
    try:
        cs_i = int(cs)
    except Exception:
        raise SystemExit("cycle_stride must be int >=1")
    if cs_i <= 0:
        raise SystemExit("cycle_stride must be int >=1")
    default["cycle_stride"] = cs_i
    sq = default.get("silent_qualities")
    if sq is None:
        default["silent_qualities"] = []
    elif not isinstance(sq, list):
        raise SystemExit("silent_qualities must be list")
    default["strict"] = bool(default.get("strict", False))
    ph = default.get("phrase_hold", "off")
    if ph not in ("off", "bar", "chord"):
        raise SystemExit("phrase_hold must be off, bar, or chord")
    default["phrase_hold"] = ph
    for key in ("phrase_merge_gap", "chord_merge_gap"):
        try:
            val = float(default.get(key, 0.0))
        except Exception:
            raise SystemExit(f"{key} must be float")
        if val < 0.0:
            val = 0.0
        default[key] = val
    return default


def generate_mapping_template(full: bool) -> str:
    """Return a YAML mapping template string."""
    if full:
        return (
            "phrase_note: 36\n"
            "phrase_velocity: 96\n"
            "phrase_length_beats: 0.25\n"
            "phrase_hold: off  # off, bar, chord\n"
            "phrase_merge_gap: 0.02  # seconds\n"
            "chord_merge_gap: 0.01  # seconds\n"
            "chord_octave: 4\n"
            "chord_velocity: 90\n"
            "triad_intervals:\n"
            "  maj: [0,4,7]\n"
            "  min: [0,3,7]\n"
            "cycle_phrase_notes: []  # e.g., [24, rest, 26] to alternate per bar (小節ごとに切替)\n"
            "cycle_start_bar: 0\n"
            "cycle_mode: bar  # or 'chord'\n"
            "cycle_stride: 1  # number of bars/chords before advancing cycle\n"
            "voicing_mode: stacked  # or 'closed'\n"
            "top_note_max: null  # e.g., 72 to cap highest chord tone\n"
            "phrase_channel: null  # MIDI channel for phrase notes\n"
            "chord_channel: null  # MIDI channel for chord notes\n"
            "accent: []  # velocity multipliers per pulse\n"
            "skip_phrase_in_rests: false\n"
            "clone_meta_only: false\n"
            "silent_qualities: []\n"
            "swing: 0.0  # 0..1 swing feel\n"
            "swing_unit: 1/8  # subdivision for swing\n"
            "chord_input_range: {lo: 48, hi: 72}\n"
        )
    else:
        return (
            "phrase_note: 36\n"
            "cycle_phrase_notes: []  # e.g., [24, rest, 26] to alternate per bar (小節ごとに切替)\n"
            "phrase_hold: off\n"
            "phrase_merge_gap: 0.02\n"
            "chord_merge_gap: 0.01\n"
            "cycle_start_bar: 0\n"
            "cycle_mode: bar  # or 'chord'\n"
            "cycle_stride: 1\n"
            "voicing_mode: stacked  # or 'closed'\n"
            "top_note_max: null\n"
            "phrase_channel: null\n"
            "chord_channel: null\n"
            "accent: []\n"
            "skip_phrase_in_rests: false\n"
            "clone_meta_only: false\n"
            "silent_qualities: []\n"
            "swing: 0.0\n"
            "swing_unit: 1/8\n"
            "chord_input_range: {lo: 48, hi: 72}\n"
        )

def read_chords_csv(path: Path) -> List['ChordSpan']:
    spans: List[ChordSpan] = []
    with path.open(newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            start = float(row['start']); end = float(row['end'])
            root = row['root'].strip()
            quality = row['quality'].strip().lower()
            if root not in PITCH_CLASS:
                raise ValueError(f"Unknown root {root}")
            spans.append(ChordSpan(start, end, PITCH_CLASS[root], quality))
    return spans

def read_chords_yaml(path: Path) -> List['ChordSpan']:
    if yaml is None:
        raise SystemExit("PyYAML is required to read YAML chord files. pip install pyyaml")
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        return []
    if isinstance(raw, dict):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("Chord YAML must be a list or mapping")

    spans: List[ChordSpan] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Chord YAML item {i} must be a mapping")
        # 必須キー（欠損→KeyError）
        missing = [k for k in ("start", "end", "root") if k not in item]
        if missing:
            raise KeyError(missing[0])
        # 型変換（数値以外→ValueError）
        try:
            start = float(item["start"])
            end = float(item["end"])
        except Exception as e:
            raise ValueError("start/end must be numeric") from e
        root = str(item["root"]).strip()
        quality = str(item.get("quality", "maj")).strip().lower()
        # 未対応ルート→ValueError（テスト想定どおり）
        if root not in PITCH_CLASS:
            raise ValueError(f"Unknown root {root}")
        spans.append(ChordSpan(start, end, PITCH_CLASS[root], quality))
    return spans

def infer_chords_by_bar(pm: 'pretty_midi.PrettyMIDI', ts_num=4, ts_den=4) -> List['ChordSpan']:
    # Build a simplistic bar grid from downbeats. If absent, estimate from tempo.
    downbeats = pm.get_downbeats()
    if len(downbeats) < 2:
        beats = pm.get_beats()
        if len(beats) < 2:
            raise ValueError("Cannot infer beats/downbeats from this MIDI; please provide a chord CSV.")
        bar_beats = ts_num * (4.0 / ts_den)
        step = max(1, int(round(bar_beats)))
        downbeats = beats[::step]

    spans: List[ChordSpan] = []
    # Aggregate pitch-class histograms per bar
    for i in range(len(downbeats)):
        start = downbeats[i]
        end = downbeats[i+1] if i+1 < len(downbeats) else pm.get_end_time()
        if end - start <= 0.0:
            continue
        pc_weights = [0.0]*12
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for n in inst.notes:
                ns = max(n.start, start)
                ne = min(n.end, end)
                if ne <= ns:
                    continue
                dur = ne - ns
                pc_weights[n.pitch % 12] += dur * (n.velocity/127.0)
        # choose a root candidate
        root_pc = max(range(12), key=lambda pc: pc_weights[pc]) if any(pc_weights) else 0
        # score maj vs min by template match (0,4,7) vs (0,3,7)
        def score(intervals):
            return sum(pc_weights[(root_pc + iv) % 12] for iv in intervals)
        maj_s = score([0,4,7]); min_s = score([0,3,7])
        quality = 'maj' if maj_s >= min_s else 'min'
        spans.append(ChordSpan(start, end, root_pc, quality))
    return spans

def ensure_tempo(pm: 'pretty_midi.PrettyMIDI', fallback_bpm: Optional[float]) -> float:
    tempi = pm.get_tempo_changes()[1]
    if len(tempi):
        return float(tempi[0])
    if fallback_bpm is None:
        return 120.0
    return float(fallback_bpm)

def beats_to_seconds(beats: float, bpm: float) -> float:
    # beats are quarter-notes
    return (60.0 / bpm) * beats

def build_sparkle_midi(pm_in: 'pretty_midi.PrettyMIDI',
                       chords: List['ChordSpan'],
                       mapping: Dict,
                       pulse_subdiv_beats: float,
                       cycle_mode: str,
                       humanize_ms: float,
                       humanize_vel: int,
                       vel_curve: str,
                       bpm: float,
                       swing: float,
                       swing_unit_beats: float,
                       *,
                       phrase_channel: Optional[int] = None,
                       chord_channel: Optional[int] = None,
                       cycle_stride: int = 1,
                       accent: Optional[List[float]] = None,
                       skip_phrase_in_rests: bool = False,
                       silent_qualities: Optional[List[str]] = None,
                       clone_meta_only: bool = False,
                       stats: Optional[Dict] = None) -> 'pretty_midi.PrettyMIDI':
    if clone_meta_only:
        out = pretty_midi.PrettyMIDI()
        out.time_signature_changes = copy.deepcopy(pm_in.time_signature_changes)
        if hasattr(pm_in, '_tempo_changes') and hasattr(out, '_tempo_changes'):
            out._tempo_changes = copy.deepcopy(pm_in._tempo_changes)
            out._tick_scales = copy.deepcopy(pm_in._tick_scales)
            out._tick_to_time = copy.deepcopy(pm_in._tick_to_time)
            meta_src = "private"
        else:
            times, tempos = pm_in.get_tempo_changes()
            if hasattr(out, '_add_tempo_change'):
                for t, tempo in zip(times, tempos):
                    try:
                        out._add_tempo_change(tempo, t)
                    except Exception:
                        pass
            elif tempos:
                out.initial_tempo = tempos[0]
            meta_src = "public"
    else:
        pm_cls = getattr(pretty_midi, 'PrettyMIDI', None)
        if isinstance(pm_cls, type) and isinstance(pm_in, pm_cls):
            out = copy.deepcopy(pm_in)
            out.instruments = []
        else:
            try:
                out = pretty_midi.PrettyMIDI()
            except TypeError:
                out = pretty_midi.PrettyMIDI(None)
            out.time_signature_changes = copy.deepcopy(getattr(pm_in, 'time_signature_changes', []))
            if hasattr(pm_in, '_tempo_changes') and hasattr(out, '_tempo_changes'):
                out._tempo_changes = copy.deepcopy(getattr(pm_in, '_tempo_changes', []))
                out._tick_scales = copy.deepcopy(getattr(pm_in, '_tick_scales', []))
                out._tick_to_time = copy.deepcopy(getattr(pm_in, '_tick_to_time', []))
            else:
                try:
                    times, tempos = pm_in.get_tempo_changes()
                except Exception:
                    times, tempos = [], []
                if hasattr(out, '_add_tempo_change'):
                    for t, tempo in zip(times, tempos):
                        try:
                            out._add_tempo_change(tempo, t)
                        except Exception:
                            pass
                elif tempos:
                    out.initial_tempo = tempos[0]

    chord_inst = pretty_midi.Instrument(program=0, name="Sparkle Chords")
    phrase_inst = pretty_midi.Instrument(program=0, name="Sparkle Phrase (Common Pulse)")
    if chord_channel is not None:
        chord_inst.midi_channel = chord_channel
    if phrase_channel is not None:
        phrase_inst.midi_channel = phrase_channel

    chord_oct = int(mapping.get("chord_octave", 4))
    chord_vel = int(mapping.get("chord_velocity", 90))
    phrase_note = int(mapping.get("phrase_note", 36))
    phrase_vel = int(mapping.get("phrase_velocity", 96))
    phrase_len_beats = float(mapping.get("phrase_length_beats", 0.25))
    if phrase_len_beats <= 0 or pulse_subdiv_beats <= 0:
        raise SystemExit("phrase_length_beats and pulse_subdiv_beats must be positive")
    phrase_hold = str(mapping.get("phrase_hold", "off"))
    phrase_merge_gap = max(0.0, float(mapping.get("phrase_merge_gap", 0.02)))
    chord_merge_gap = max(0.0, float(mapping.get("chord_merge_gap", 0.01)))
    release_sec = max(0.0, float(mapping.get("phrase_release_ms", 0.0))) / 1000.0
    min_phrase_len_sec = max(0.0, float(mapping.get("min_phrase_len_ms", 0.0))) / 1000.0
    held_vel_mode = str(mapping.get("held_vel_mode", "first"))
    cycle_notes: List[Optional[int]] = list(mapping.get("cycle_phrase_notes", []) or [])
    cycle_start_bar = int(mapping.get("cycle_start_bar", 0))
    if cycle_notes:
        L = len(cycle_notes)
        cycle_start_bar = ((cycle_start_bar % L) + L) % L
    chord_range = mapping.get("chord_input_range")
    voicing_mode = mapping.get("voicing_mode", "stacked")
    top_note_max = mapping.get("top_note_max")
    strict = bool(mapping.get("strict", False))

    beat_times = pm_in.get_beats()
    if len(beat_times) < 2:
        raise SystemExit("Could not determine beats from MIDI")

    def beat_to_time(b: float) -> float:
        idx = int(math.floor(b))
        frac = b - idx
        if idx >= len(beat_times) - 1:
            last = beat_times[-1] - beat_times[-2]
            return beat_times[-1] + (b - (len(beat_times) - 1)) * last
        return beat_times[idx] + frac * (beat_times[idx + 1] - beat_times[idx])

    def time_to_beat(t: float) -> float:
        idx = bisect.bisect_right(beat_times, t) - 1
        if idx < 0:
            return 0.0
        if idx >= len(beat_times) - 1:
            last = beat_times[-1] - beat_times[-2]
            return (len(beat_times) - 1) + (t - beat_times[-1]) / last
        span = beat_times[idx + 1] - beat_times[idx]
        return idx + (t - beat_times[idx]) / span

    ts_changes = pm_in.time_signature_changes
    meter_map: List[Tuple[float, int, int]] = []
    estimated_4_4 = False
    if ts_changes:
        for ts in ts_changes:
            meter_map.append((ts.time, ts.numerator, ts.denominator))
    else:
        meter_map.append((0.0, 4, 4))
        estimated_4_4 = True
    downbeats = pm_in.get_downbeats()
    if ts_changes and len(downbeats) >= 2:
        ts0 = ts_changes[0]
        expected_bar = beat_to_time(ts0.numerator * (4.0 / ts0.denominator))
        if abs((downbeats[1] - downbeats[0]) - expected_bar) > 1e-6:
            downbeats = []
            for i, ts in enumerate(ts_changes):
                next_t = ts_changes[i + 1].time if i + 1 < len(ts_changes) else pm_in.get_end_time()
                start_b = time_to_beat(ts.time)
                end_b = time_to_beat(next_t)
                bar_beats = ts.numerator * (4.0 / ts.denominator)
                bar = start_b
                while bar < end_b - 1e-9:
                    downbeats.append(beat_to_time(bar))
                    bar += bar_beats
            downbeats.sort()
    if len(downbeats) < 2:
        beats = beat_times
        if ts_changes:
            downbeats = []
            for i, ts in enumerate(ts_changes):
                next_t = ts_changes[i + 1].time if i + 1 < len(ts_changes) else pm_in.get_end_time()
                start_b = time_to_beat(ts.time)
                end_b = time_to_beat(next_t)
                bar_beats = ts.numerator * (4.0 / ts.denominator)
                bar = start_b
                while bar < end_b - 1e-9:
                    downbeats.append(beat_to_time(bar))
                    bar += bar_beats
            downbeats.sort()
        else:
            downbeats = beats[::4]
    if cycle_notes and len(downbeats) < 2 and cycle_mode == 'bar':
        logging.info("cycle disabled; using fixed phrase_note=%d", phrase_note)
        cycle_notes = []
        if stats is not None:
            stats["cycle_disabled"] = True

    if stats is not None:
        stats["downbeats"] = downbeats
        stats["bar_pulses"] = {}
        stats["bar_phrase_notes"] = {}
        stats["bar_velocities"] = {}
        stats["triads"] = []
        stats["meters"] = meter_map
        if estimated_4_4:
            stats["estimated_4_4"] = True

    # precompute pulses per bar for velocity curves
    bar_counts: Dict[int, int] = {}
    for i, start in enumerate(downbeats):
        end = downbeats[i + 1] if i + 1 < len(downbeats) else pm_in.get_end_time()
        sb = time_to_beat(start)
        eb = time_to_beat(end)
        bar_counts[i] = int(math.ceil((eb - sb) / pulse_subdiv_beats))

    if stats is not None and phrase_hold != 'off':
        for i, start in enumerate(downbeats):
            end = downbeats[i + 1] if i + 1 < len(downbeats) else pm_in.get_end_time()
            sb = time_to_beat(start)
            eb = time_to_beat(end)
            pulses: List[Tuple[float, float]] = []
            b = sb
            idx = 0
            while b < eb - EPS:
                t = beat_to_time(b)
                pulses.append((b, t))
                interval = pulse_subdiv_beats
                if swing > 0.0 and abs(pulse_subdiv_beats - swing_unit_beats) < EPS:
                    interval *= (1 + swing) if idx % 2 == 0 else (1 - swing)
                b += interval
                idx += 1
            stats["bar_pulses"][i] = pulses

    bar_progress: Dict[int, int] = {}

    def vel_factor(mode: str, idx: int, total: int) -> float:
        if total <= 1:
            x = 0.0
        else:
            x = idx / (total - 1)
        if mode == 'up':
            return x
        if mode == 'down':
            return 1.0 - x
        if mode == 'sine':
            return math.sin(math.pi * x)
        return 1.0

    def pick_phrase_note(t: float, chord_idx: int) -> Optional[int]:
        if not cycle_notes:
            return phrase_note
        if cycle_mode == 'bar':
            base = max(0, bisect.bisect_right(downbeats, t) - 1)
        else:
            base = chord_idx
        idx = ((base + cycle_start_bar) // max(1, cycle_stride)) % len(cycle_notes)
        return cycle_notes[idx]

    def pulses_in_range(start_t: float, end_t: float) -> List[Tuple[int, int]]:
        indices: List[Tuple[int, int]] = []
        b = time_to_beat(start_t)
        eb = time_to_beat(end_t)
        while b < eb - EPS:
            t = beat_to_time(b)
            bar_idx = max(0, bisect.bisect_right(downbeats, t) - 1)
            bar_start_b = time_to_beat(downbeats[bar_idx])
            idx = int(math.floor((b - bar_start_b + EPS) / pulse_subdiv_beats))
            indices.append((bar_idx, idx))
            interval = pulse_subdiv_beats
            if swing > 0.0 and abs(pulse_subdiv_beats - swing_unit_beats) < EPS:
                interval *= (1 + swing) if idx % 2 == 0 else (1 - swing)
            b += interval
        return indices

    silent_qualities = set(silent_qualities or [])
    for c_idx, span in enumerate(chords):
        is_silent = span.quality in silent_qualities or span.quality == 'rest'
        triad: List[int] = []
        if not is_silent:
            triad = triad_pitches(span.root_pc, span.quality, chord_oct, mapping)
            if chord_range:
                triad = place_in_range(triad, chord_range['lo'], chord_range['hi'], voicing_mode=voicing_mode)
            if top_note_max is not None:
                while max(triad) > top_note_max and all(n - 12 >= 0 for n in triad):
                    triad = [n - 12 for n in triad]
                if triad and max(triad) > top_note_max:
                    msg = f"top_note_max={top_note_max} cannot be satisfied for triad {triad}"
                    if strict:
                        raise SystemExit(msg)
                    logging.warning(msg)
            s_t, e_t = clip_note_interval(span.start, span.end)
            for p in triad:
                chord_inst.notes.append(pretty_midi.Note(velocity=chord_vel, pitch=p, start=s_t, end=e_t))
        if stats is not None:
            stats["triads"].append(triad)
        if skip_phrase_in_rests and is_silent:
            continue

        sb = time_to_beat(span.start)
        eb = time_to_beat(span.end)
        if phrase_hold == 'chord':
            pn = pick_phrase_note(span.start, c_idx)
            if pn is not None:
                bar_idx = max(0, bisect.bisect_right(downbeats, span.start) - 1)
                total = bar_counts.get(bar_idx, 1)
                vf = vel_factor(vel_curve, 0, total)
                pulse_idx = pulses_in_range(span.start, span.end)
                if accent and pulse_idx:
                    if held_vel_mode == 'max':
                        af = max(accent[i % len(accent)] for _, i in pulse_idx)
                    elif held_vel_mode == 'mean':
                        af = sum(accent[i % len(accent)] for _, i in pulse_idx) / len(pulse_idx)
                    else:
                        af = accent[pulse_idx[0][1] % len(accent)]
                else:
                    af = accent[0] if accent else 1.0
                base_vel = max(1, min(127, int(round(phrase_vel * vf * af))))
                if humanize_vel > 0:
                    base_vel = max(1, min(127, int(round(base_vel + random.uniform(-humanize_vel, humanize_vel)))))
                if humanize_ms > 0.0:
                    delta_s = random.uniform(-humanize_ms, humanize_ms) / 1000.0
                    delta_e = random.uniform(-humanize_ms, humanize_ms) / 1000.0
                else:
                    delta_s = delta_e = 0.0
                start_t = span.start + delta_s
                if cycle_mode == 'bar':
                    start_t = max(downbeats[bar_idx], span.start, start_t)
                else:
                    start_t = max(span.start, start_t)
                end_t = min(span.end, span.end + delta_e)
                start_t, end_t = clip_note_interval(start_t, end_t)
                _append_phrase(phrase_inst, pn, start_t, end_t, base_vel, phrase_merge_gap, release_sec, min_phrase_len_sec)
                if stats is not None:
                    end_bar = max(0, bisect.bisect_right(downbeats, span.end - 1e-9) - 1)
                    for bi in range(bar_idx, end_bar + 1):
                        if bi not in stats["bar_phrase_notes"]:
                            stats["bar_phrase_notes"][bi] = pn
                        stats["bar_velocities"].setdefault(bi, []).append(base_vel)
        elif phrase_hold == 'bar':
            bar_idx = max(0, bisect.bisect_right(downbeats, span.start) - 1)
            while bar_idx < len(downbeats) and downbeats[bar_idx] < span.end:
                bar_start = downbeats[bar_idx]
                bar_end = downbeats[bar_idx + 1] if bar_idx + 1 < len(downbeats) else pm_in.get_end_time()
                start = max(span.start, bar_start)
                end = min(span.end, bar_end)
                pn = pick_phrase_note(start, c_idx)
                if pn is not None:
                    total = bar_counts.get(bar_idx, 1)
                    vf = vel_factor(vel_curve, 0, total)
                    pulse_idx = pulses_in_range(start, end)
                    if accent and pulse_idx:
                        if held_vel_mode == 'max':
                            af = max(accent[i % len(accent)] for _, i in pulse_idx)
                        elif held_vel_mode == 'mean':
                            af = sum(accent[i % len(accent)] for _, i in pulse_idx) / len(pulse_idx)
                        else:
                            af = accent[pulse_idx[0][1] % len(accent)]
                    else:
                        af = accent[0] if accent else 1.0
                    base_vel = max(1, min(127, int(round(phrase_vel * vf * af))))
                    if humanize_vel > 0:
                        base_vel = max(1, min(127, int(round(base_vel + random.uniform(-humanize_vel, humanize_vel)))))
                    if humanize_ms > 0.0:
                        delta_s = random.uniform(-humanize_ms, humanize_ms) / 1000.0
                        delta_e = random.uniform(-humanize_ms, humanize_ms) / 1000.0
                    else:
                        delta_s = delta_e = 0.0
                    start_t = start + delta_s
                    if cycle_mode == 'bar':
                        start_t = max(bar_start, span.start, start_t)
                    else:
                        start_t = max(span.start, start_t)
                    end_t = min(end, end + delta_e)
                    start_t, end_t = clip_note_interval(start_t, end_t)
                    _append_phrase(phrase_inst, pn, start_t, end_t, base_vel, phrase_merge_gap, release_sec, min_phrase_len_sec)
                    if stats is not None and bar_idx not in stats["bar_phrase_notes"]:
                        stats["bar_phrase_notes"][bar_idx] = pn
                    if stats is not None:
                        stats["bar_velocities"].setdefault(bar_idx, []).append(base_vel)
                bar_idx += 1
        else:
            b = sb
            while b < eb - 1e-9:
                t = beat_to_time(b)
                bar_idx = max(0, bisect.bisect_right(downbeats, t) - 1)
                total = bar_counts.get(bar_idx, 1)
                idx = bar_progress.get(bar_idx, 0)
                bar_progress[bar_idx] = idx + 1
                vf = vel_factor(vel_curve, idx, total)
                af = accent[idx % len(accent)] if accent else 1.0
                base_vel = max(1, min(127, int(round(phrase_vel * vf * af))))
                if humanize_vel > 0:
                    base_vel = max(1, min(127, int(round(base_vel + random.uniform(-humanize_vel, humanize_vel)))))

                pn = pick_phrase_note(t, c_idx)
                end_b = b + phrase_len_beats
                boundary = beat_to_time(end_b)
                if cycle_mode == 'bar' and bar_idx + 1 < len(downbeats):
                    boundary = min(boundary, downbeats[bar_idx + 1])
                boundary = min(boundary, span.end)

                if stats is not None:
                    stats["bar_pulses"].setdefault(bar_idx, []).append((b, t))
                if pn is not None:
                    if humanize_ms > 0.0:
                        delta_s = random.uniform(-humanize_ms, humanize_ms) / 1000.0
                        delta_e = random.uniform(-humanize_ms, humanize_ms) / 1000.0
                    else:
                        delta_s = delta_e = 0.0
                    start_t = t + delta_s
                    if cycle_mode == 'bar':
                        start_t = max(downbeats[bar_idx], span.start, start_t)
                    else:
                        start_t = max(span.start, start_t)
                    end_t = min(boundary, boundary + delta_e)
                    start_t, end_t = clip_note_interval(start_t, end_t)
                    _append_phrase(phrase_inst, pn, start_t, end_t, base_vel, phrase_merge_gap, release_sec, min_phrase_len_sec)
                    if stats is not None and bar_idx not in stats["bar_phrase_notes"]:
                        stats["bar_phrase_notes"][bar_idx] = pn
                    if stats is not None:
                        stats["bar_velocities"].setdefault(bar_idx, []).append(base_vel)
                interval = pulse_subdiv_beats
                if swing > 0.0 and abs(pulse_subdiv_beats - swing_unit_beats) < 1e-9:
                    interval *= (1 + swing) if idx % 2 == 0 else (1 - swing)
                b += interval

    _legato_merge_chords(chord_inst, chord_merge_gap)
    out.instruments.append(chord_inst)
    out.instruments.append(phrase_inst)
    if clone_meta_only and logging.getLogger().isEnabledFor(logging.INFO):
        logging.info("clone_meta_only tempo/time-signature via %s API", meta_src)
    if stats is not None:
        stats["pulse_count"] = len(phrase_inst.notes)
        stats["bar_count"] = len(downbeats)
    return out

def main():
    ap = argparse.ArgumentParser(description="Convert generic MIDI to UJAM Sparkle-friendly MIDI (chords + common pulse).")
    ap.add_argument("input_midi", type=str, help="Input MIDI file")
    ap.add_argument("--out", type=str, required=True, help="Output MIDI file")
    ap.add_argument("--pulse", type=str, default="1/8", help="Pulse subdivision (e.g., 1/8, 1/16, 1/4)")
    ap.add_argument("--bpm", type=float, default=None, help="Fallback BPM if input has no tempo")
    ap.add_argument("--chords", type=str, default=None, help="Chord CSV/YAML file. If omitted, infer per bar.")
    ap.add_argument("--mapping", type=str, default=None, help="YAML for Sparkle mapping (phrase note, chord octave, velocities, triad intervals).")
    ap.add_argument("--cycle-phrase-notes", type=str, default=None,
                    help="Comma-separated phrase trigger notes to cycle per bar (e.g., 24,26,C1,rest)")
    ap.add_argument("--cycle-start-bar", type=int, default=None,
                    help="Bar offset for cycling (default 0)")
    ap.add_argument("--cycle-mode", choices=["bar", "chord"], default=None, help="Cycle mode")
    ap.add_argument("--cycle-stride", type=int, default=None, help="Number of bars/chords before advancing cycle")
    ap.add_argument(
        "--phrase-channel",
        type=int,
        default=None,
        help="MIDI channel for phrase notes (0-15, best effort; instruments are split regardless)",
    )
    ap.add_argument(
        "--chord-channel",
        type=int,
        default=None,
        help="MIDI channel for chord notes (0-15, best effort; instruments are split regardless)",
    )
    ap.add_argument("--humanize-timing-ms", type=float, default=0.0, help="Randomize note timing +/- ms")
    ap.add_argument("--humanize-vel", type=int, default=0, help="Randomize velocity +/- value")
    ap.add_argument("--vel-curve", choices=["flat", "up", "down", "sine"], default="flat", help="Velocity curve within bar")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for humanization")
    ap.add_argument("--swing", type=float, default=0.0, help="Swing amount 0..1")
    ap.add_argument("--swing-unit", type=str, default="1/8", choices=["1/8", "1/16"], help="Subdivision for swing")
    ap.add_argument("--accent", type=str, default=None, help="JSON velocity multipliers per pulse")
    ap.add_argument("--skip-phrase-in-rests", action="store_true", help="Suppress phrase notes in rest spans")
    ap.add_argument("--phrase-hold", choices=["off", "bar", "chord"], default=None,
                    help="Hold phrase keys: off, bar, or chord (default: off)")
    ap.add_argument("--phrase-merge-gap", type=float, default=None,
                    help="Merge same-pitch phrase notes if gap <= seconds (default: 0.02)")
    ap.add_argument("--chord-merge-gap", type=float, default=None,
                    help="Merge same-pitch chord notes if gap <= seconds (default: 0.01)")
    ap.add_argument("--phrase-release-ms", type=float, default=None,
                    help="Shorten phrase note ends by ms (default: 0.0)")
    ap.add_argument("--min-phrase-len-ms", type=float, default=None,
                    help="Minimum phrase note length in ms (default: 0.0)")
    ap.add_argument("--held-vel-mode", choices=["first", "max", "mean"], default=None,
                    help="Velocity for held notes: first, max, or mean accent (default: first)")
    ap.add_argument(
        "--clone-meta-only",
        action="store_true",
        help="Clone only tempo/time-signature from input (best effort across pretty_midi versions)",
    )
    ap.add_argument("--write-mapping-template", action="store_true", help="Print mapping YAML template to stdout")
    ap.add_argument("--write-mapping-template-path", type=str, default=None, help="Write mapping YAML template to PATH")
    ap.add_argument("--template-style", choices=["full", "minimal"], default="full", help="Template style")
    ap.add_argument("--dry-run", action="store_true", help="Do not write output; log summary")
    ap.add_argument("--quiet", action="store_true", help="Reduce log output")
    ap.add_argument("--verbose", action="store_true", help="Increase log output")
    args, extras = ap.parse_known_args()

    if extras and args.write_mapping_template:
        legacy_tpl_args = []
        while extras and not extras[0].startswith('-'):
            legacy_tpl_args.append(extras.pop(0))
        logging.info("--write-mapping-template with arguments is deprecated; use --template-style/--write-mapping-template-path")
    else:
        legacy_tpl_args = None
    if extras:
        ap.error(f"unrecognized arguments: {' '.join(extras)}")

    if args.quiet:
        logging.basicConfig(level=logging.WARNING)
    elif args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.seed is not None:
        random.seed(args.seed)

    if args.write_mapping_template or args.write_mapping_template_path or legacy_tpl_args is not None:
        style = args.template_style
        path = args.write_mapping_template_path
        if legacy_tpl_args is not None:
            if legacy_tpl_args and legacy_tpl_args[0] in ("full", "minimal"):
                style = legacy_tpl_args[0]
                legacy_tpl_args = legacy_tpl_args[1:]
            if legacy_tpl_args:
                path = legacy_tpl_args[0]
        content = generate_mapping_template(style == 'full')
        if path:
            Path(path).write_text(content)
        else:
            print(content, end="")
        return

    pm = pretty_midi.PrettyMIDI(args.input_midi)

    ts_num, ts_den = parse_time_sig()  # currently fixed 4/4; extend as needed
    bpm = ensure_tempo(pm, args.bpm)
    pulse_beats = parse_pulse(args.pulse)
    swing_unit_beats = parse_pulse(args.swing_unit)
    if not (0.0 <= args.swing < 1.0):
        raise SystemExit("--swing must be 0.0<=s<1.0")
    swing = args.swing
    if swing > 0.0 and abs(swing_unit_beats - pulse_beats) >= 1e-9:
        logging.info("swing disabled: swing unit %s != pulse %s", args.swing_unit, args.pulse)
        swing = 0.0

    mapping = load_mapping(Path(args.mapping) if args.mapping else None)
    cycle_notes_raw = mapping.get("cycle_phrase_notes", [])
    cycle_notes: List[Optional[int]] = []
    for tok in cycle_notes_raw:
        if tok is None:
            cycle_notes.append(None)
        elif isinstance(tok, str):
            if tok.strip().lower() == 'rest':
                cycle_notes.append(None)
            else:
                cycle_notes.append(parse_midi_note(tok))
        else:
            cycle_notes.append(validate_midi_note(tok))
    cycle_start_bar = int(mapping.get("cycle_start_bar", 0))
    cycle_mode = mapping.get("cycle_mode", "bar")
    cycle_stride = int(mapping.get("cycle_stride", 1))
    phrase_channel = mapping.get("phrase_channel")
    chord_channel = mapping.get("chord_channel")
    accent = validate_accent(mapping.get("accent"))
    silent_qualities = mapping.get("silent_qualities", [])
    clone_meta_only = bool(mapping.get("clone_meta_only", False))
    if args.cycle_phrase_notes is not None:
        tokens = [t for t in args.cycle_phrase_notes.split(',') if t.strip()]
        parsed: List[Optional[int]] = []
        for tok in tokens:
            if tok.strip().lower() == 'rest':
                parsed.append(None)
            else:
                parsed.append(parse_midi_note(tok))
        cycle_notes = parsed
    if args.cycle_start_bar is not None:
        cycle_start_bar = args.cycle_start_bar
    if args.cycle_mode is not None:
        cycle_mode = args.cycle_mode
    if args.cycle_stride is not None:
        if args.cycle_stride <= 0:
            raise SystemExit("cycle-stride must be >=1")
        cycle_stride = args.cycle_stride
    for key, val in (("phrase_channel", args.phrase_channel), ("chord_channel", args.chord_channel)):
        if val is not None:
            if not (0 <= val <= 15):
                raise SystemExit(f"{key.replace('_', '-')} must be 0..15")
            if key == "phrase_channel":
                phrase_channel = val
            else:
                chord_channel = val
    if args.accent is not None:
        accent = parse_accent_arg(args.accent)
    mapping["cycle_phrase_notes"] = cycle_notes
    mapping["cycle_start_bar"] = cycle_start_bar
    mapping["cycle_mode"] = cycle_mode
    mapping["cycle_stride"] = cycle_stride
    mapping["phrase_channel"] = phrase_channel
    mapping["chord_channel"] = chord_channel
    mapping["accent"] = accent
    mapping["silent_qualities"] = silent_qualities
    phrase_hold = mapping.get("phrase_hold", "off")
    phrase_merge_gap = float(mapping.get("phrase_merge_gap", 0.02))
    chord_merge_gap = float(mapping.get("chord_merge_gap", 0.01))
    phrase_release_ms = float(mapping.get("phrase_release_ms", 0.0))
    min_phrase_len_ms = float(mapping.get("min_phrase_len_ms", 0.0))
    held_vel_mode = mapping.get("held_vel_mode", "first")
    if args.phrase_hold is not None:
        phrase_hold = args.phrase_hold
    if args.phrase_merge_gap is not None:
        phrase_merge_gap = args.phrase_merge_gap
    if args.chord_merge_gap is not None:
        chord_merge_gap = args.chord_merge_gap
    if args.phrase_release_ms is not None:
        phrase_release_ms = args.phrase_release_ms
    if args.min_phrase_len_ms is not None:
        min_phrase_len_ms = args.min_phrase_len_ms
    if args.held_vel_mode is not None:
        held_vel_mode = args.held_vel_mode
    phrase_merge_gap = max(0.0, phrase_merge_gap)
    chord_merge_gap = max(0.0, chord_merge_gap)
    phrase_release_ms = max(0.0, phrase_release_ms)
    min_phrase_len_ms = max(0.0, min_phrase_len_ms)
    mapping["phrase_hold"] = phrase_hold
    mapping["phrase_merge_gap"] = phrase_merge_gap
    mapping["chord_merge_gap"] = chord_merge_gap
    mapping["phrase_release_ms"] = phrase_release_ms
    mapping["min_phrase_len_ms"] = min_phrase_len_ms
    mapping["held_vel_mode"] = held_vel_mode
    clone_meta_only = bool(args.clone_meta_only or clone_meta_only)

    if args.chords:
        chord_path = Path(args.chords)
        if chord_path.suffix in {'.yaml', '.yml'}:
            chords = read_chords_yaml(chord_path)
        else:
            chords = read_chords_csv(chord_path)
    else:
        chords = infer_chords_by_bar(pm, ts_num, ts_den)

    stats: Dict = {}
    out_pm = build_sparkle_midi(pm, chords, mapping, pulse_beats, cycle_mode,
                                args.humanize_timing_ms, args.humanize_vel,
                                args.vel_curve, bpm, swing, swing_unit_beats,
                                phrase_channel=phrase_channel, chord_channel=chord_channel,
                                cycle_stride=cycle_stride, accent=accent,
                                skip_phrase_in_rests=args.skip_phrase_in_rests,
                                silent_qualities=silent_qualities,
                                clone_meta_only=clone_meta_only, stats=stats)
    if args.dry_run:
        phrase_inst = None
        for inst in out_pm.instruments:
            if inst.name == "Sparkle Phrase (Common Pulse)":
                phrase_inst = inst
                break
        bar_pulses = stats.get("bar_pulses", {})
        B = len(bar_pulses)
        P = sum(len(p) for p in bar_pulses.values())
        T = len(phrase_inst.notes) if phrase_inst else 0
        logging.info("bars=%d pulses(theoretical)=%d triggers(emitted)=%d", B, P, T)
        logging.info("phrase_hold=%s phrase_merge_gap=%.3f chord_merge_gap=%.3f phrase_release_ms=%.1f min_phrase_len_ms=%.1f",
                     mapping.get("phrase_hold"),
                     mapping.get("phrase_merge_gap"),
                     mapping.get("chord_merge_gap"),
                     mapping.get("phrase_release_ms"),
                     mapping.get("min_phrase_len_ms"))
        if stats.get("cycle_disabled"):
            logging.info("cycle disabled; using fixed phrase_note=%d", mapping.get("phrase_note"))
        if stats.get("meters"):
            meters = [(float(t), n, d) for t, n, d in stats["meters"]]
            if stats.get("estimated_4_4"):
                logging.info("meter_map=%s (estimated 4/4 grid)", meters)
            else:
                logging.info("meter_map=%s", meters)
        if mapping.get("cycle_phrase_notes"):
            example = [stats["bar_phrase_notes"].get(i) for i in range(min(4, stats.get("bar_count", 0)))]
            logging.info("cycle_mode=%s notes=%s", cycle_mode, example)
        if mapping.get("chord_input_range"):
            logging.info("chord_input_range=%s first_triads=%s", mapping.get("chord_input_range"), stats.get("triads", [])[:2])
        for i in range(min(4, stats.get("bar_count", 0))):
            pn = stats["bar_phrase_notes"].get(i, mapping.get("phrase_note"))
            pulses = stats["bar_pulses"].get(i, [])
            vels = stats["bar_velocities"].get(i, [])
            if str(mapping.get("phrase_hold")) != "off" and phrase_inst is not None:
                bar_start = stats["downbeats"][i]
                bar_end = stats["downbeats"][i + 1] if i + 1 < len(stats["downbeats"]) else out_pm.get_end_time()
                trig = sum(1 for n in phrase_inst.notes if bar_start <= n.start < bar_end)
                logging.info("bar %d | phrase %s | triggers %d | vel %s", i, pn, trig, vels)
            else:
                logging.info("bar %d | phrase %s | pulses %d | vel %s", i, pn, len(pulses), vels)
        if args.verbose:
            for b_idx in sorted(stats["bar_pulses"].keys()):
                logging.info("bar %d pulses %s", b_idx, stats["bar_pulses"][b_idx])
        return
    out_pm.write(args.out)
    logging.info("Wrote %s", args.out)

if __name__ == "__main__":
    main()
