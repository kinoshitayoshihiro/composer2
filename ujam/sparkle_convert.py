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
from functools import lru_cache
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

from .consts import PHRASE_INST_NAME, CHORD_INST_NAME, DAMP_INST_NAME, PITCH_CLASS, SECTION_PRESETS
from .phrase_schedule import (
    schedule_phrase_keys,
    SectionLFO,
    StableChordGuard,
    VocalAdaptive,
    ChordSpan,
    markov_pick,
    DENSITY_PRESETS,
)
from .damping import _append_phrase, _legato_merge_chords, emit_damping
from .io import read_chords_yaml, read_section_profiles

NOTE_RE = re.compile(r"^([A-G][b#]?)(-?\d+)$")

NOTE_ALIASES: Dict[str, int] = {}
NOTE_ALIAS_INV: Dict[int, str] = {}

EPS = 1e-9


def parse_midi_note(token: str) -> int:
    """Parse a MIDI note from integer or note name like C1 or F#2."""
    token = token.strip().replace("＃", "#").replace("♯", "#").replace("♭", "b").replace("ｂ", "b")
    fw = {chr(ord("Ａ") + i): chr(ord("A") + i) for i in range(26)}
    fw.update({chr(ord("０") + i): str(i) for i in range(10)})
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
        raise SystemExit("cycle_phrase_notes must be MIDI 0..127, 'rest', or note names")
    return v


def parse_note_token(tok: Union[str, int], *, warn_unknown: bool = False) -> Optional[int]:
    """Normalize note token to MIDI int or None for rests."""
    if isinstance(tok, str):
        t = tok.strip()
        if t.lower() == "rest":
            return None
        if t in NOTE_ALIASES:
            return NOTE_ALIASES[t]
        try:
            return parse_midi_note(t)
        except SystemExit:
            if warn_unknown:
                logging.warning("unknown note alias: %s", tok)
                return None
            raise
    try:
        return validate_midi_note(int(tok))
    except Exception as e:  # pragma: no cover - unlikely
        raise SystemExit(f"Invalid note token: {tok}") from e


def clip_note_interval(start_t: float, end_t: float, *, eps: float = EPS) -> Tuple[float, float]:
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


def stretch_accent(accent: List[float], n: int) -> List[float]:
    if len(accent) == n:
        return accent
    if n <= 0:
        return []
    if len(accent) == 1:
        return accent * n
    out: List[float] = []
    for i in range(n):
        pos = i * (len(accent) - 1) / (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(accent) - 1)
        frac = pos - lo
        out.append(accent[lo] * (1 - frac) + accent[hi] * frac)
    return out


def write_reports(
    stats: Dict, json_path: Optional[str] = None, md_path: Optional[str] = None
) -> None:
    if json_path:
        Path(json_path).write_text(json.dumps(stats, indent=2))
    if md_path:
        lines = [
            "# Sparkle Report",
            f"bars: {stats.get('bar_count', 0)}",
            f"fills: {stats.get('fill_count', 0)}",
        ]
        Path(md_path).write_text("\n".join(lines) + "\n")


def write_debug_md(stats: Dict, path: str) -> None:
    sections = stats.get("section_tags", {})
    notes = stats.get("bar_phrase_notes_list", [])
    fills = set(stats.get("fill_bars", []))
    reasons = stats.get("bar_reason", {})
    lfo_pos = stats.get("lfo_pos", {})
    guards = stats.get("guard_hold_beats", {})
    lines = [
        "|bar|section|phrase|fill|accent|lfo_pos|guard_hold_beats|damping|reason|",
        "|-|-|-|-|-|-|-|-|-|",
    ]
    bar_count = stats.get("bar_count", len(notes))
    accent_scales = stats.get("accent_scales", {})
    for b in range(bar_count):
        note = notes[b] if b < len(notes) else None
        alias = NOTE_ALIAS_INV.get(note, str(note) if note is not None else "")
        fill_flag = "1" if b in fills else ""
        acc = f"{accent_scales.get(b, 1.0):.2f}"
        lfo = f"{lfo_pos.get(b, 0.0):.2f}" if lfo_pos else ""
        guard = f"{guards.get(b, 0.0):.2f}" if guards else ""
        damp = stats.get("damping", {}).get(b, "")
        src = reasons.get(b, {}).get("source", "")
        lines.append(
            f"|{b}|{sections.get(b, '')}|{alias}|{fill_flag}|{acc}|{lfo}|{guard}|{damp}|{src}|"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def parse_accent_arg(s: str) -> Optional[List[float]]:
    try:
        data = json.loads(s)
    except Exception:
        raise SystemExit("--accent must be JSON list of numbers")
    return validate_accent(data)


def parse_damp_arg(s: str) -> Tuple[str, Dict[str, float]]:
    """Parse --damp argument into mode and kwargs."""
    if not s:
        return "none", {}
    if ":" in s:
        mode, rest = s.split(":", 1)
    else:
        mode, rest = s, ""
    mode = mode.strip()
    kw: Dict[str, float] = {}
    for token in filter(None, (t.strip() for t in rest.split(","))):
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k in {"cc", "channel", "value", "smooth"}:
            kw[k] = int(v)
        elif k in {"deadband", "min_beats"}:
            kw[k] = float(v)
        elif k in {"clip_lo", "clip_hi"}:
            kw[k] = int(v)
        else:
            try:
                kw[k] = float(v)
            except ValueError:
                kw[k] = v
    lo = kw.pop("clip_lo", None)
    hi = kw.pop("clip_hi", None)
    if lo is not None or hi is not None:
        kw["clip"] = (int(lo) if lo is not None else 0, int(hi) if hi is not None else 127)
    return mode, kw


def validate_section_lfo_cfg(cfg: Dict) -> Dict:
    if not isinstance(cfg, dict):
        raise SystemExit("section_lfo must be object")
    if int(cfg.get("period", 0)) <= 0:
        raise SystemExit("section_lfo.period must be >0")
    shape = cfg.get("shape", "linear")
    if shape not in ("linear", "sine", "triangle"):
        raise SystemExit("section_lfo.shape invalid")
    vel = cfg.get("vel")
    if vel is not None:
        if not (isinstance(vel, list) and len(vel) == 2):
            raise SystemExit("section_lfo.vel must be [min,max]")
        if min(vel) <= 0:
            raise SystemExit("section_lfo.vel must be >0")
    fill = cfg.get("fill")
    if fill is not None:
        if not (isinstance(fill, list) and len(fill) == 2):
            raise SystemExit("section_lfo.fill must be [min,max]")
        if not (0.0 <= fill[0] <= 1.0 and 0.0 <= fill[1] <= 1.0):
            raise SystemExit("section_lfo.fill range 0..1")
    return cfg


def validate_stable_guard_cfg(cfg: Dict) -> Dict:
    if not isinstance(cfg, dict):
        raise SystemExit("stable_chord_guard must be object")
    if int(cfg.get("min_hold_beats", 0)) < 0:
        raise SystemExit("min_hold_beats must be >=0")
    strat = cfg.get("strategy", "skip")
    if strat not in ("skip", "alternate"):
        raise SystemExit("strategy must be skip or alternate")
    return cfg


def validate_vocal_adapt_cfg(cfg: Dict) -> Dict:
    if not isinstance(cfg, dict):
        raise SystemExit("vocal_adapt must be object")
    if "dense_onset" not in cfg:
        raise SystemExit("dense_onset required")
    if "dense_ratio" in cfg:
        dr = float(cfg["dense_ratio"])
        if not (0.0 <= dr <= 1.0):
            raise SystemExit("dense_ratio must be 0..1")
    if "smooth_bars" in cfg and int(cfg["smooth_bars"]) < 0:
        raise SystemExit("smooth_bars must be >=0")
    return cfg


def validate_style_inject_cfg(cfg: Dict) -> Dict:
    if not isinstance(cfg, dict):
        raise SystemExit("style_inject must be object")
    if int(cfg.get("period", 0)) < 1:
        raise SystemExit("style_inject.period must be >=1")
    note = cfg.get("note")
    if note is None or not (0 <= int(note) <= 127):
        raise SystemExit("style_inject.note 0..127 required")
    if float(cfg.get("duration_beats", 0)) <= 0:
        raise SystemExit("style_inject.duration_beats must be >0")
    if "min_gap_beats" in cfg and float(cfg["min_gap_beats"]) < 0:
        raise SystemExit("style_inject.min_gap_beats must be >=0")
    if "avoid_pitches" in cfg:
        if not isinstance(cfg["avoid_pitches"], list):
            raise SystemExit("style_inject.avoid_pitches must be list")
        for n in cfg["avoid_pitches"]:
            if not (0 <= int(n) <= 127):
                raise SystemExit("style_inject.avoid_pitches entries must be 0..127")
    return cfg


def vocal_features_from_midi(path: str) -> Tuple[List[int], List[float]]:
    vpm = pretty_midi.PrettyMIDI(path)
    vb = vpm.get_downbeats()
    if not vb:
        return [], []
    onsets = [0] * len(vb)
    voiced = [0.0] * len(vb)
    end_time = (
        vpm.get_end_time()
        if hasattr(vpm, "get_end_time")
        else max((n.end for inst in vpm.instruments for n in inst.notes), default=0.0)
    )
    bar_dur = [
        (vb[i + 1] - vb[i]) if i + 1 < len(vb) else (end_time - vb[i]) for i in range(len(vb))
    ]
    for inst in vpm.instruments:
        for n in inst.notes:
            idx = bisect.bisect_right(vb, n.start) - 1
            if 0 <= idx < len(onsets):
                onsets[idx] += 1
                end = min(n.end, vb[idx + 1] if idx + 1 < len(vb) else vpm.get_end_time())
                voiced[idx] += max(0.0, end - n.start)
    ratios = [voiced[i] / bar_dur[i] if bar_dur[i] > 0 else 0.0 for i in range(len(onsets))]
    return onsets, ratios


def vocal_onsets_from_midi(path: str) -> List[int]:
    onsets, _ = vocal_features_from_midi(path)
    return onsets


@dataclass
class ChordSpan:
    start: float
    end: float
    root_pc: int  # 0-11
    quality: str  # 'maj' or 'min' (extendable)


def parse_time_sig(default_num=4, default_den=4) -> Tuple[int, int]:
    # pretty_midi doesn't store TS per track reliably; keep configurable if needed
    return default_num, default_den


def parse_pulse(s: str) -> float:
    """
    Parse a subdivision string like '1/8' -> 0.5 beats (if a beat is a quarter note).
    We define '1/8' as eighth-notes = 0.5 quarter-beats.
    """
    s = s.strip()
    if "/" in s:
        num, den = s.split("/", 1)
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
    """Return MIDI numbers for a simple triad in the given octave based on mapping intervals."""
    intervals = mapping.get("triad_intervals", {}).get(quality, [0, 4, 7])  # default maj
    base_c = (octave + 1) * 12  # C-octave base
    return [base_c + ((root_pc + iv) % 12) for iv in intervals]


def place_in_range(
    pitches: List[int], lo: int, hi: int, *, voicing_mode: str = "stacked"
) -> List[int]:
    res: List[int] = []
    prev: Optional[int] = None
    if voicing_mode == "closed":
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
                while res[i] - res[i - 1] > 12 and res[i] - 12 >= lo:
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
        "phrase_note": 36,  # Default left-hand "Common" phrase key (C2)
        "phrase_velocity": 96,
        "phrase_length_beats": 0.25,
        "phrase_hold": "off",
        "phrase_merge_gap": 0.02,
        "chord_merge_gap": 0.01,
        "chord_octave": 4,  # Place chord tones around C4-B4 by default
        "chord_velocity": 90,
        "triad_intervals": {"maj": [0, 4, 7], "min": [0, 3, 7]},
        "cycle_phrase_notes": [],  # e.g., [24, 26] to alternate per bar
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
        "chord_input_range": None,
        "voicing_mode": "stacked",
        "top_note_max": None,
        "phrase_channel": None,
        "chord_channel": None,
        "cycle_stride": 1,
        "merge_reset_at": "none",
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
    areas = default.get("areas", {})
    aliases: Dict[str, int] = {}
    for area in ("common", "style"):
        anchors = areas.get(area, {}).get("anchors", {})
        for name, val in anchors.items():
            try:
                aliases[str(name)] = int(val)
            except Exception:
                continue
    default["note_aliases"] = aliases
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
    mra = str(default.get("merge_reset_at", "none")).lower()
    if mra not in ("none", "bar", "chord"):
        raise SystemExit("merge_reset_at must be none, bar, or chord")
    default["merge_reset_at"] = mra
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


def apply_section_preset(mapping: Dict, preset_name: Optional[str]) -> None:
    if not preset_name:
        return
    preset = SECTION_PRESETS.get(preset_name)
    if not preset:
        raise SystemExit(f"unknown section preset: {preset_name}")
    sections = mapping.get("sections") or []
    tag_map = {s.get("tag"): s for s in sections if isinstance(s, dict)}
    for tag, cfg in preset.items():
        sec = tag_map.get(tag)
        if not sec:
            continue
        for k, v in cfg.items():
            if k not in sec:
                if k == "phrase_pool":
                    sec[k] = [parse_note_token(t, warn_unknown=True) for t in v]
                else:
                    sec[k] = v


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
            "merge_reset_at: none  # none, bar, chord\n"
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
            "merge_reset_at: none\n"
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


def read_chords_csv(path: Path) -> List["ChordSpan"]:
    spans: List[ChordSpan] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            start = float(row["start"])
            end = float(row["end"])
            root = row["root"].strip()
            quality = row["quality"].strip().lower()
            if root not in PITCH_CLASS:
                raise ValueError(f"Unknown root {root}")
            spans.append(ChordSpan(start, end, PITCH_CLASS[root], quality))
    return spans


def infer_chords_by_bar(pm: "pretty_midi.PrettyMIDI", ts_num=4, ts_den=4) -> List["ChordSpan"]:
    # Build a simplistic bar grid from downbeats. If absent, estimate from tempo.
    downbeats = pm.get_downbeats()
    if len(downbeats) < 2:
        beats = pm.get_beats()
        if len(beats) < 2:
            raise ValueError(
                "Cannot infer beats/downbeats from this MIDI; please provide a chord CSV."
            )
        bar_beats = ts_num * (4.0 / ts_den)
        step = max(1, int(round(bar_beats)))
        downbeats = beats[::step]

    spans: List[ChordSpan] = []
    # Aggregate pitch-class histograms per bar
    for i in range(len(downbeats)):
        start = downbeats[i]
        end = downbeats[i + 1] if i + 1 < len(downbeats) else pm.get_end_time()
        if end - start <= 0.0:
            continue
        pc_weights = [0.0] * 12
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for n in inst.notes:
                ns = max(n.start, start)
                ne = min(n.end, end)
                if ne <= ns:
                    continue
                dur = ne - ns
                pc_weights[n.pitch % 12] += dur * (n.velocity / 127.0)
        # choose a root candidate
        root_pc = max(range(12), key=lambda pc: pc_weights[pc]) if any(pc_weights) else 0

        # score maj vs min by template match (0,4,7) vs (0,3,7)
        def score(intervals):
            return sum(pc_weights[(root_pc + iv) % 12] for iv in intervals)

        maj_s = score([0, 4, 7])
        min_s = score([0, 3, 7])
        quality = "maj" if maj_s >= min_s else "min"
        spans.append(ChordSpan(start, end, root_pc, quality))
    return spans


def ensure_tempo(pm: "pretty_midi.PrettyMIDI", fallback_bpm: Optional[float]) -> float:
    tempi = pm.get_tempo_changes()[1]
    if len(tempi):
        return float(tempi[0])
    if fallback_bpm is None:
        return 120.0
    return float(fallback_bpm)


def beats_to_seconds(beats: float, bpm: float) -> float:
    # beats are quarter-notes
    return (60.0 / bpm) * beats


def build_sparkle_midi(
    pm_in: "pretty_midi.PrettyMIDI",
    chords: List["ChordSpan"],
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
    stats: Optional[Dict] = None,
    merge_reset_at: str = "none",
    section_lfo: Optional[SectionLFO] = None,
    stable_guard: Optional[StableChordGuard] = None,
    fill_policy: str = "section",
    vocal_adapt: Optional[VocalAdaptive] = None,
    vocal_ducking: float = 0.0,
    lfo_targets: Tuple[str, ...] = ("phrase",),
    section_pool_weights: Optional[Dict[str, Dict[int, float]]] = None,
    rng: Optional[random.Random] = None,
    guide_onsets: Optional[List[int]] = None,
    guide_onset_th: int = 0,
    guide_style_note: Optional[int] = None,
) -> "pretty_midi.PrettyMIDI":
    rng = rng or random.Random()

    def _duck(bar_idx: int, vel: int) -> int:
        if vocal_ducking > 0 and vocal_adapt and vocal_adapt.dense_phrase is not None:
            if vocal_adapt.phrase_for_bar(bar_idx) == vocal_adapt.dense_phrase:
                return max(1, int(round(vel * (1.0 - vocal_ducking))))
        return vel

    if clone_meta_only:
        out = pretty_midi.PrettyMIDI()
        out.time_signature_changes = copy.deepcopy(pm_in.time_signature_changes)
        if hasattr(pm_in, "_tempo_changes") and hasattr(out, "_tempo_changes"):
            out._tempo_changes = copy.deepcopy(pm_in._tempo_changes)
            out._tick_scales = copy.deepcopy(pm_in._tick_scales)
            out._tick_to_time = copy.deepcopy(pm_in._tick_to_time)
            meta_src = "private"
        else:
            times, tempos = pm_in.get_tempo_changes()
            if hasattr(out, "_add_tempo_change"):
                for t, tempo in zip(times, tempos):
                    try:
                        out._add_tempo_change(tempo, t)
                    except Exception:
                        pass
            elif tempos:
                out.initial_tempo = tempos[0]
            meta_src = "public"
    else:
        pm_cls = getattr(pretty_midi, "PrettyMIDI", None)
        if isinstance(pm_cls, type) and isinstance(pm_in, pm_cls):
            out = copy.deepcopy(pm_in)
            out.instruments = []
        else:
            try:
                out = pretty_midi.PrettyMIDI()
            except TypeError:
                out = pretty_midi.PrettyMIDI(None)
            out.time_signature_changes = copy.deepcopy(getattr(pm_in, "time_signature_changes", []))
            if hasattr(pm_in, "_tempo_changes") and hasattr(out, "_tempo_changes"):
                out._tempo_changes = copy.deepcopy(getattr(pm_in, "_tempo_changes", []))
                out._tick_scales = copy.deepcopy(getattr(pm_in, "_tick_scales", []))
                out._tick_to_time = copy.deepcopy(getattr(pm_in, "_tick_to_time", []))
            else:
                try:
                    times, tempos = pm_in.get_tempo_changes()
                except Exception:
                    times, tempos = [], []
                if hasattr(out, "_add_tempo_change"):
                    for t, tempo in zip(times, tempos):
                        try:
                            out._add_tempo_change(tempo, t)
                        except Exception:
                            pass
                elif tempos:
                    out.initial_tempo = tempos[0]

    chord_inst = pretty_midi.Instrument(program=0, name=CHORD_INST_NAME)
    phrase_inst = pretty_midi.Instrument(program=0, name=PHRASE_INST_NAME)
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

    @lru_cache(maxsize=None)
    def beat_to_time(b: float) -> float:
        idx = int(math.floor(b))
        frac = b - idx
        if idx >= len(beat_times) - 1:
            last = beat_times[-1] - beat_times[-2]
            return beat_times[-1] + (b - (len(beat_times) - 1)) * last
        return beat_times[idx] + frac * (beat_times[idx + 1] - beat_times[idx])

    @lru_cache(maxsize=None)
    def time_to_beat(t: float) -> float:
        idx = bisect.bisect_right(beat_times, t) - 1
        if idx < 0:
            return 0.0
        if idx >= len(beat_times) - 1:
            last = beat_times[-1] - beat_times[-2]
            return (len(beat_times) - 1) + (t - beat_times[-1]) / last
        span = beat_times[idx + 1] - beat_times[idx]
        return idx + (t - beat_times[idx]) / span

    def maybe_merge_gap(inst, pitch, start_t, *, bar_start=None, chord_start=None):
        """Return merge gap or -1.0 to force new note at reset boundary."""
        mg = phrase_merge_gap
        if (
            merge_reset_at != "none"
            and inst.notes
            and inst.notes[-1].pitch == pitch
            and (start_t - inst.notes[-1].end) <= phrase_merge_gap + EPS
        ):
            if (
                merge_reset_at == "bar"
                and bar_start is not None
                and abs(start_t - bar_start) <= EPS
            ):
                return -1.0
            if (
                merge_reset_at == "chord"
                and chord_start is not None
                and abs(start_t - chord_start) <= EPS
            ):
                return -1.0
        return mg

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
        if abs((downbeats[1] - downbeats[0]) - expected_bar) > EPS:
            downbeats = []
            for i, ts in enumerate(ts_changes):
                next_t = ts_changes[i + 1].time if i + 1 < len(ts_changes) else pm_in.get_end_time()
                start_b = time_to_beat(ts.time)
                end_b = time_to_beat(next_t)
                bar_beats = ts.numerator * (4.0 / ts.denominator)
                bar = start_b
                while bar < end_b - EPS:
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
                while bar < end_b - EPS:
                    downbeats.append(beat_to_time(bar))
                    bar += bar_beats
            downbeats.sort()
        else:
            downbeats = beats[::4]
    if cycle_notes and len(downbeats) < 2 and cycle_mode == "bar":
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
        stats["bar_reason"] = {}
        stats["lfo_pos"] = {}
        stats["guard_hold_beats"] = {}
        stats["fill_sources"] = {}
        stats["merge_events"] = []
        stats["fill_conflicts"] = []
        if estimated_4_4:
            stats["estimated_4_4"] = True

    num_bars = len(downbeats) - 1
    density_map: Dict[int, str] = {}
    sections = mapping.get("sections")
    if sections:
        tag_map: Dict[int, str] = {}
        for sec in sections:
            dens = sec.get("density")
            start = int(sec.get("start_bar", 0))
            end = int(sec.get("end_bar", num_bars))
            if dens in ("low", "med", "high"):
                for b in range(max(0, start), min(num_bars, end)):
                    density_map[b] = dens
            tag = sec.get("tag")
            if tag:
                for b in range(max(0, start), min(num_bars, end)):
                    tag_map[b] = tag
        if stats is not None:
            stats["section_tags"] = tag_map
    if stats is not None:
        stats["bar_density"] = density_map
        stats["bar_count"] = num_bars
        stats["swing_unit"] = swing_unit_beats
        if section_lfo:
            stats["accent_scales"] = {b: section_lfo.vel_scale(b) for b in range(num_bars)}
    # precompute pulses per bar for velocity curves
    bar_info: Dict[int, Tuple[float, float, float, float, int]] = {}
    for i, start in enumerate(downbeats[:-1]):
        end = downbeats[i + 1]
        sb = time_to_beat(start)
        eb = time_to_beat(end)
        count = int(math.ceil((eb - sb) / pulse_subdiv_beats))
        bar_info[i] = (start, end, sb, eb, count)
    bar_counts = {i: info[4] for i, info in bar_info.items()}
    bar_accent_cache: Dict[int, List[float]] = {}

    def accent_for_bar(bi: int) -> Optional[List[float]]:
        if accent is None:
            return None
        arr = bar_accent_cache.get(bi)
        if arr is None:
            n = bar_counts.get(bi, len(accent))
            if n % len(accent) == 0:
                arr = accent * (n // len(accent))
            else:
                arr = stretch_accent(accent, n)
            bar_accent_cache[bi] = arr
        return arr

    bar_qualities: List[Optional[str]] = [None] * num_bars
    if chords:
        for i in range(num_bars):
            start, end, _, _, _ = bar_info[i]
            qs = [c.quality for c in chords if c.start <= start + EPS and c.end >= end - EPS]
            if len(qs) == 1:
                bar_qualities[i] = qs[0]

    phrase_plan: List[Optional[int]] = []
    fill_map: Dict[int, int] = {}
    if cycle_mode == "bar":
        sections = mapping.get("sections")
        if chords:
            last_idx = max(max(0, bisect.bisect_right(downbeats, c.end - EPS) - 1) for c in chords)
            num_bars = max(num_bars, last_idx + 1)
            if len(bar_qualities) < num_bars:
                bar_qualities.extend([None] * (num_bars - len(bar_qualities)))
        phrase_plan, fill_map, fill_src = schedule_phrase_keys(
            num_bars,
            cycle_notes,
            sections,
            mapping.get("style_fill"),
            cycle_start_bar=cycle_start_bar,
            cycle_stride=cycle_stride,
            lfo=section_lfo,
            style_inject=mapping.get("style_inject"),
            fill_policy=fill_policy,
            pulse_subdiv=pulse_subdiv_beats,
            markov=mapping.get("markov"),
            bar_qualities=bar_qualities,
            section_pool_weights=section_pool_weights,
            rng=rng,
            stats=stats,
        )
        bar_sources = ["cycle"] * len(phrase_plan)
        if stats is not None:
            stats["bar_phrase_notes_list"] = list(phrase_plan)
            stats["fill_bars"] = list(fill_map.keys())
        if sections:
            for sec in sections:
                pool = sec.get("pool")
                if pool:
                    start = int(sec.get("start_bar", 0))
                    end = int(sec.get("end_bar", num_bars))
                    for b in range(max(0, start), min(num_bars, end)):
                        bar_sources[b] = "section"
        if vocal_adapt and phrase_plan:
            for i in range(len(phrase_plan)):
                alt = vocal_adapt.phrase_for_bar(i)
                if alt is not None:
                    phrase_plan[i] = alt
                    if bar_sources[i] != "vocal":
                        bar_sources[i] = f"{bar_sources[i]}+vocal"
                    else:
                        bar_sources[i] = "vocal"
        if guide_onsets and guide_style_note is not None:
            for idx, cnt in enumerate(guide_onsets):
                if cnt >= guide_onset_th:
                    tgt = idx - 1
                    if tgt >= 0 and tgt not in fill_map:
                        fill_map[tgt] = (guide_style_note, pulse_subdiv_beats, 1.0)
                        fill_src[tgt] = "style"
                        if stats is not None:
                            stats["fill_bars"].append(tgt)
                    break
        if stats is not None:
            for i, pn in enumerate(phrase_plan):
                stats["bar_reason"][i] = {"source": bar_sources[i], "note": pn}
            stats["fill_sources"].update(fill_src)

    if stats is not None and phrase_hold != "off":
        for i, start in enumerate(downbeats):
            end = downbeats[i + 1] if i + 1 < len(downbeats) else pm_in.get_end_time()
            sb = time_to_beat(start)
            eb = time_to_beat(end)
            pulses: List[Tuple[float, float]] = []
            b = sb
            idx = 0
            preset = (
                DENSITY_PRESETS.get(
                    stats.get("bar_density", {}).get(i, "med"), DENSITY_PRESETS["med"]
                )
                if stats
                else DENSITY_PRESETS["med"]
            )
            while b < eb - EPS:
                t = beat_to_time(b)
                pulses.append((b, t))
                interval = pulse_subdiv_beats * preset["stride"]
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
        if mode == "up":
            return x
        if mode == "down":
            return 1.0 - x
        if mode == "sine":
            return math.sin(math.pi * x)
        return 1.0

    last_bar_idx = -1
    last_bar_note: Optional[int] = None

    def pick_phrase_note(t: float, chord_idx: int) -> Optional[int]:
        nonlocal last_bar_idx, last_bar_note
        if phrase_plan:
            bar_idx = max(0, bisect.bisect_right(downbeats, t) - 1)
            if bar_idx < len(phrase_plan):
                pn = phrase_plan[bar_idx]
            else:
                if cycle_notes:
                    idx = ((bar_idx + cycle_start_bar) // max(1, cycle_stride)) % len(cycle_notes)
                    pn = cycle_notes[idx]
                else:
                    pn = 36 + bar_idx
                phrase_plan.append(pn)
            if bar_idx != last_bar_idx:
                last_bar_idx = bar_idx
                if pn is None:
                    last_bar_note = None
                    return None
                if pn == last_bar_note:
                    return None
                last_bar_note = pn
            if pn is None:
                return None
            return pn
        if not cycle_notes:
            return phrase_note
        if cycle_mode == "bar":
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
            preset = (
                DENSITY_PRESETS.get(
                    stats.get("bar_density", {}).get(bar_idx, "med"), DENSITY_PRESETS["med"]
                )
                if stats
                else DENSITY_PRESETS["med"]
            )
            interval = pulse_subdiv_beats * preset["stride"]
            if swing > 0.0 and abs(pulse_subdiv_beats - swing_unit_beats) < EPS:
                interval *= (1 + swing) if idx % 2 == 0 else (1 - swing)
            b += interval
        return indices

    silent_qualities = set(silent_qualities or [])
    for c_idx, span in enumerate(chords):
        is_silent = span.quality in silent_qualities or span.quality == "rest"
        triad: List[int] = []
        if not is_silent:
            triad = triad_pitches(span.root_pc, span.quality, chord_oct, mapping)
            if chord_range:
                triad = place_in_range(
                    triad, chord_range["lo"], chord_range["hi"], voicing_mode=voicing_mode
                )
            if top_note_max is not None:
                while max(triad) > top_note_max and all(n - 12 >= 0 for n in triad):
                    triad = [n - 12 for n in triad]
                if triad and max(triad) > top_note_max:
                    msg = f"top_note_max={top_note_max} cannot be satisfied for triad {triad}"
                    if strict:
                        raise SystemExit(msg)
                    logging.warning(msg)
            s_t, e_t = clip_note_interval(span.start, span.end, eps=EPS)
            c_vel = chord_vel
            if section_lfo and "chord" in lfo_targets:
                bar_idx = max(0, bisect.bisect_right(downbeats, span.start) - 1)
                c_vel = max(1, min(127, int(round(c_vel * section_lfo.vel_scale(bar_idx)))))
                if stats is not None:
                    stats["lfo_pos"][bar_idx] = section_lfo._pos(bar_idx)
            for p in triad:
                chord_inst.notes.append(
                    pretty_midi.Note(velocity=c_vel, pitch=p, start=s_t, end=e_t)
                )
        if stats is not None:
            stats["triads"].append(triad)
        if skip_phrase_in_rests and is_silent:
            continue

        sb = time_to_beat(span.start)
        eb = time_to_beat(span.end)
        if phrase_hold == "chord":
            pn = pick_phrase_note(span.start, c_idx)
            if pn is not None:
                bar_idx = max(0, bisect.bisect_right(downbeats, span.start) - 1)
                total = bar_counts.get(bar_idx, 1)
                vf = vel_factor(vel_curve, 0, total)
                pulse_idx = pulses_in_range(span.start, span.end)
                preset = (
                    DENSITY_PRESETS.get(
                        stats.get("bar_density", {}).get(bar_idx, "med"), DENSITY_PRESETS["med"]
                    )
                    if stats
                    else DENSITY_PRESETS["med"]
                )
                acc_arr = accent_for_bar(bar_idx)
                if acc_arr and pulse_idx:
                    if held_vel_mode == "max":
                        af = max(acc_arr[i % len(acc_arr)] for _, i in pulse_idx) * preset["accent"]
                    elif held_vel_mode == "mean":
                        af = (
                            sum(acc_arr[i % len(acc_arr)] for _, i in pulse_idx) / len(pulse_idx)
                        ) * preset["accent"]
                    else:
                        af = acc_arr[pulse_idx[0][1] % len(acc_arr)] * preset["accent"]
                else:
                    af = (acc_arr[0] if acc_arr else 1.0) * preset["accent"]
                base_vel = max(1, min(127, int(round(phrase_vel * vf * af))))
                base_vel = _duck(bar_idx, base_vel)
                if section_lfo and "phrase" in lfo_targets:
                    base_vel = max(
                        1, min(127, int(round(base_vel * section_lfo.vel_scale(bar_idx))))
                    )
                    if stats is not None:
                        stats["lfo_pos"][bar_idx] = section_lfo._pos(bar_idx)
                if humanize_vel > 0:
                    base_vel = max(
                        1, min(127, int(round(base_vel + rng.uniform(-humanize_vel, humanize_vel))))
                    )
                if humanize_ms > 0.0:
                    delta_s = rng.uniform(-humanize_ms, humanize_ms) / 1000.0
                    delta_e = rng.uniform(-humanize_ms, humanize_ms) / 1000.0
                else:
                    delta_s = delta_e = 0.0
                start_t = span.start + delta_s
                if cycle_mode == "bar":
                    start_t = max(downbeats[bar_idx], span.start, start_t)
                else:
                    start_t = max(span.start, start_t)
                end_t = min(span.end, span.end + delta_e)
                start_t, end_t = clip_note_interval(start_t, end_t, eps=EPS)
                mg = maybe_merge_gap(
                    phrase_inst, pn, start_t, bar_start=downbeats[bar_idx], chord_start=span.start
                )
                _append_phrase(
                    phrase_inst,
                    pn,
                    start_t,
                    end_t,
                    base_vel,
                    mg,
                    release_sec,
                    min_phrase_len_sec,
                    stats,
                )
                if stats is not None:
                    end_bar = max(0, bisect.bisect_right(downbeats, span.end - EPS) - 1)
                    for bi in range(bar_idx, end_bar + 1):
                        if bi not in stats["bar_phrase_notes"]:
                            stats["bar_phrase_notes"][bi] = pn
                        stats["bar_velocities"].setdefault(bi, []).append(base_vel)
        elif phrase_hold == "bar":
            bar_idx = max(0, bisect.bisect_right(downbeats, span.start) - 1)
            while bar_idx < len(downbeats) and downbeats[bar_idx] < span.end:
                bar_start = downbeats[bar_idx]
                bar_end = (
                    downbeats[bar_idx + 1] if bar_idx + 1 < len(downbeats) else pm_in.get_end_time()
                )
                start = max(span.start, bar_start)
                end = min(span.end, bar_end)
                pn = pick_phrase_note(start, c_idx)
                if pn is not None:
                    total = bar_counts.get(bar_idx, 1)
                    vf = vel_factor(vel_curve, 0, total)
                    pulse_idx = pulses_in_range(start, end)
                    preset = (
                        DENSITY_PRESETS.get(
                            stats.get("bar_density", {}).get(bar_idx, "med"), DENSITY_PRESETS["med"]
                        )
                        if stats
                        else DENSITY_PRESETS["med"]
                    )
                    acc_arr = accent_for_bar(bar_idx)
                    if acc_arr and pulse_idx:
                        if held_vel_mode == "max":
                            af = (
                                max(acc_arr[i % len(acc_arr)] for _, i in pulse_idx)
                                * preset["accent"]
                            )
                        elif held_vel_mode == "mean":
                            af = (
                                sum(acc_arr[i % len(acc_arr)] for _, i in pulse_idx)
                                / len(pulse_idx)
                            ) * preset["accent"]
                        else:
                            af = acc_arr[pulse_idx[0][1] % len(acc_arr)] * preset["accent"]
                    else:
                        af = (acc_arr[0] if acc_arr else 1.0) * preset["accent"]
                    base_vel = max(1, min(127, int(round(phrase_vel * vf * af))))
                    base_vel = _duck(bar_idx, base_vel)
                    if section_lfo:
                        base_vel = max(
                            1, min(127, int(round(base_vel * section_lfo.vel_scale(bar_idx))))
                        )
                    if humanize_vel > 0:
                        base_vel = max(
                            1,
                            min(
                                127, int(round(base_vel + rng.uniform(-humanize_vel, humanize_vel)))
                            ),
                        )
                    if humanize_ms > 0.0:
                        delta_s = rng.uniform(-humanize_ms, humanize_ms) / 1000.0
                        delta_e = rng.uniform(-humanize_ms, humanize_ms) / 1000.0
                    else:
                        delta_s = delta_e = 0.0
                    start_t = start + delta_s
                    if cycle_mode == "bar":
                        start_t = max(bar_start, span.start, start_t)
                    else:
                        start_t = max(span.start, start_t)
                    end_t = min(end, end + delta_e)
                    start_t, end_t = clip_note_interval(start_t, end_t, eps=EPS)
                    mg = maybe_merge_gap(
                        phrase_inst, pn, start_t, bar_start=bar_start, chord_start=span.start
                    )
                    _append_phrase(
                        phrase_inst,
                        pn,
                        start_t,
                        end_t,
                        base_vel,
                        mg,
                        release_sec,
                        min_phrase_len_sec,
                        stats,
                    )
                    if stats is not None and bar_idx not in stats["bar_phrase_notes"]:
                        stats["bar_phrase_notes"][bar_idx] = pn
                    if stats is not None:
                        stats["bar_velocities"].setdefault(bar_idx, []).append(base_vel)
                bar_idx += 1
        else:
            b = sb
            while b < eb - EPS:
                t = beat_to_time(b)
                bar_idx = max(0, bisect.bisect_right(downbeats, t) - 1)
                preset = (
                    DENSITY_PRESETS.get(
                        stats.get("bar_density", {}).get(bar_idx, "med"), DENSITY_PRESETS["med"]
                    )
                    if stats
                    else DENSITY_PRESETS["med"]
                )
                total = bar_counts.get(bar_idx, 1)
                idx = bar_progress.get(bar_idx, 0)
                bar_progress[bar_idx] = idx + 1
                vf = vel_factor(vel_curve, idx, total)
                acc_arr = accent_for_bar(bar_idx)
                af = (acc_arr[idx % len(acc_arr)] if acc_arr else 1.0) * preset["accent"]
                base_vel = max(1, min(127, int(round(phrase_vel * vf * af))))
                base_vel = _duck(bar_idx, base_vel)
                if section_lfo and "phrase" in lfo_targets:
                    base_vel = max(
                        1, min(127, int(round(base_vel * section_lfo.vel_scale(bar_idx))))
                    )
                    if stats is not None:
                        stats["lfo_pos"][bar_idx] = section_lfo._pos(bar_idx)
                if humanize_vel > 0:
                    base_vel = max(
                        1, min(127, int(round(base_vel + rng.uniform(-humanize_vel, humanize_vel))))
                    )

                interval = pulse_subdiv_beats * preset["stride"]
                if swing > 0.0 and abs(pulse_subdiv_beats - swing_unit_beats) < EPS:
                    interval *= (1 + swing) if idx % 2 == 0 else (1 - swing)
                next_b = b + interval
                if cycle_mode == "bar" and bar_idx + 1 < len(downbeats):
                    next_b = min(next_b, time_to_beat(downbeats[bar_idx + 1]))
                next_b = min(next_b, time_to_beat(span.end))
                beat_inc = next_b - b
                if stable_guard:
                    stable_guard.step((span.root_pc, span.quality), beat_inc)
                    if stats is not None:
                        stats["guard_hold_beats"][bar_idx] = stable_guard.hold
                pn = pick_phrase_note(t, c_idx)
                if stable_guard:
                    pn = stable_guard.filter(pn)
                end_b = b + phrase_len_beats * preset["len"]
                boundary = beat_to_time(end_b)
                if cycle_mode == "bar" and bar_idx + 1 < len(downbeats):
                    boundary = min(boundary, downbeats[bar_idx + 1])
                boundary = min(boundary, span.end)

                if stats is not None:
                    stats["bar_pulses"].setdefault(bar_idx, []).append((b, t))
                if pn is not None:
                    if humanize_ms > 0.0:
                        delta_s = rng.uniform(-humanize_ms, humanize_ms) / 1000.0
                        delta_e = rng.uniform(-humanize_ms, humanize_ms) / 1000.0
                    else:
                        delta_s = delta_e = 0.0
                    start_t = t + delta_s
                    if cycle_mode == "bar":
                        start_t = max(downbeats[bar_idx], span.start, start_t)
                    else:
                        start_t = max(span.start, start_t)
                    end_t = min(boundary, boundary + delta_e)
                    start_t, end_t = clip_note_interval(start_t, end_t, eps=EPS)
                    mg = maybe_merge_gap(
                        phrase_inst,
                        pn,
                        start_t,
                        bar_start=downbeats[bar_idx],
                        chord_start=span.start,
                    )
                    _append_phrase(
                        phrase_inst,
                        pn,
                        start_t,
                        end_t,
                        base_vel,
                        mg,
                        release_sec,
                        min_phrase_len_sec,
                        stats,
                    )
                    if stats is not None and bar_idx not in stats["bar_phrase_notes"]:
                        stats["bar_phrase_notes"][bar_idx] = pn
                    if stats is not None:
                        stats["bar_velocities"].setdefault(bar_idx, []).append(base_vel)
                b += beat_inc

    for bar_idx, (pitch, dur, vscale) in fill_map.items():
        if pitch is None or bar_idx + 1 >= len(downbeats):
            continue
        end_b = bar_info.get(bar_idx, (0, 0, 0, time_to_beat(downbeats[bar_idx + 1]), 0))[3]
        start_b = end_b - dur
        start_t = beat_to_time(start_b)
        end_t = beat_to_time(end_b)
        vel = max(1, min(127, int(round(phrase_vel * vscale))))
        if section_lfo and "fill" in lfo_targets:
            vel = max(1, min(127, int(round(vel * section_lfo.vel_scale(bar_idx)))))
            if stats is not None:
                stats["lfo_pos"][bar_idx] = section_lfo._pos(bar_idx)
        _append_phrase(
            phrase_inst,
            pitch,
            start_t,
            end_t,
            vel,
            phrase_merge_gap,
            release_sec,
            min_phrase_len_sec,
            stats,
        )

    _legato_merge_chords(chord_inst, chord_merge_gap)
    out.instruments.append(chord_inst)
    out.instruments.append(phrase_inst)
    if clone_meta_only and logging.getLogger().isEnabledFor(logging.INFO):
        logging.info("clone_meta_only tempo/time-signature via %s API", meta_src)
    if stats is not None:
        stats["pulse_count"] = len(phrase_inst.notes)
        stats["bar_count"] = len(downbeats)
        stats["fill_count"] = len(fill_map)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Convert generic MIDI to UJAM Sparkle-friendly MIDI (chords + common pulse)."
    )
    ap.add_argument("input_midi", type=str, help="Input MIDI file")
    ap.add_argument("--out", type=str, required=True, help="Output MIDI file")
    ap.add_argument(
        "--pulse", type=str, default="1/8", help="Pulse subdivision (e.g., 1/8, 1/16, 1/4)"
    )
    ap.add_argument("--bpm", type=float, default=None, help="Fallback BPM if input has no tempo")
    ap.add_argument(
        "--chords", type=str, default=None, help="Chord CSV/YAML file. If omitted, infer per bar."
    )
    ap.add_argument(
        "--mapping",
        type=str,
        default=None,
        help="YAML for Sparkle mapping (phrase note, chord octave, velocities, triad intervals).",
    )
    ap.add_argument("--section-preset", type=str, default=None, help="Predefined section profile")
    ap.add_argument(
        "--cycle-phrase-notes",
        type=str,
        default=None,
        help="Comma-separated phrase trigger notes to cycle per bar (e.g., 24,26,C1,rest)",
    )
    ap.add_argument(
        "--cycle-start-bar", type=int, default=None, help="Bar offset for cycling (default 0)"
    )
    ap.add_argument("--cycle-mode", choices=["bar", "chord"], default=None, help="Cycle mode")
    ap.add_argument(
        "--cycle-stride",
        type=int,
        default=None,
        help="Number of bars/chords before advancing cycle",
    )
    ap.add_argument(
        "--merge-reset-at",
        choices=["none", "bar", "chord"],
        default=None,
        help="Reset phrase merge at bar or chord boundaries",
    )
    ap.add_argument(
        "--section-lfo",
        type=str,
        default=None,
        help='JSON periodic arc scaling velocities/fill {"period":4,"vel":[0.9,1.1],"fill":[0,1]}',
    )
    ap.add_argument(
        "--lfo-apply",
        type=str,
        default=None,
        help='JSON list of LFO targets e.g. ["phrase","chord","fill"]',
    )
    ap.add_argument(
        "--fill-policy",
        type=str,
        default="section",
        choices=["section", "lfo", "style", "first", "last"],
        help="Fill conflict resolution policy",
    )
    ap.add_argument(
        "--stable-guard",
        "--stable-chord-guard",
        dest="stable_guard",
        type=str,
        default=None,
        help='JSON stable chord guard {"min_hold_beats":4,"strategy":"alternate"}',
    )
    ap.add_argument("--vocal-adapt", type=str, default=None, help="JSON vocal density adapt")
    ap.add_argument("--vocal-guide", type=str, default=None, help="Vocal MIDI guiding density")
    ap.add_argument("--guide-vocal", type=str, default=None, help="Automatic vocal-aware mode")
    ap.add_argument("--guide-onset-th", type=int, default=4, help="Onset threshold for dense bars")
    ap.add_argument(
        "--guide-rest-th", type=float, default=0.5, help="Rest ratio threshold (unused)"
    )
    ap.add_argument("--guide-style-every", type=int, default=0, help="Style fill period (unused)")
    ap.add_argument(
        "--guide-chorus-boost", type=float, default=1.0, help="Chorus fill boost (unused)"
    )
    ap.add_argument(
        "--style-inject", type=str, default=None, help="JSON periodic style phrase injection"
    )
    ap.add_argument(
        "--section-pool-weights",
        type=str,
        default=None,
        help="JSON tag->{note:weight} override for section pools",
    )
    ap.add_argument(
        "--vocal-ducking",
        type=float,
        default=0.0,
        help="Scale phrase velocity in dense vocal bars (0-1)",
    )
    ap.add_argument("--debug-json", type=str, default=None, help="Write merged config to PATH")
    ap.add_argument("--debug-md", type=str, default=None, help="Write debug markdown table")
    ap.add_argument(
        "--debug-midi-out", type=str, default=None, help="Write phrase-only MIDI to PATH"
    )
    ap.add_argument("--print-plan", action="store_true", help="Print per-bar phrase plan")
    ap.add_argument(
        "--report-json",
        "--report",
        dest="report_json",
        type=str,
        default=None,
        help="Write stats JSON to PATH",
    )
    ap.add_argument("--report-md", type=str, default=None, help="Write stats markdown to PATH")
    ap.add_argument(
        "--damp",
        type=str,
        default="none",
        help="Damping spec e.g. 'fixed:cc=11,value=64' or 'vocal:cc=11,channel=1'",
    )
    ap.add_argument(
        "--log-level", type=str, default="info", choices=["debug", "info"], help="Logging level"
    )
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
    ap.add_argument(
        "--humanize-timing-ms", type=float, default=0.0, help="Randomize note timing +/- ms"
    )
    ap.add_argument("--humanize-vel", type=int, default=0, help="Randomize velocity +/- value")
    ap.add_argument(
        "--vel-curve",
        choices=["flat", "up", "down", "sine"],
        default="flat",
        help="Velocity curve within bar",
    )
    ap.add_argument("--seed", type=int, default=None, help="Random seed for humanization")
    ap.add_argument("--swing", type=float, default=0.0, help="Swing amount 0..1")
    ap.add_argument(
        "--swing-unit",
        type=str,
        default="1/8",
        choices=["1/8", "1/16"],
        help="Subdivision for swing",
    )
    ap.add_argument("--accent", type=str, default=None, help="JSON velocity multipliers per pulse")
    ap.add_argument(
        "--sections", type=str, default=None, help="JSON list of sections (start_bar,end_bar,tag)"
    )
    ap.add_argument(
        "--skip-phrase-in-rests", action="store_true", help="Suppress phrase notes in rest spans"
    )
    ap.add_argument(
        "--phrase-hold",
        choices=["off", "bar", "chord"],
        default=None,
        help="Hold phrase keys: off, bar, or chord (default: off)",
    )
    ap.add_argument(
        "--phrase-merge-gap",
        type=float,
        default=None,
        help="Merge same-pitch phrase notes if gap <= seconds (default: 0.02)",
    )
    ap.add_argument(
        "--chord-merge-gap",
        type=float,
        default=None,
        help="Merge same-pitch chord notes if gap <= seconds (default: 0.01)",
    )
    ap.add_argument(
        "--phrase-release-ms",
        type=float,
        default=None,
        help="Shorten phrase note ends by ms (default: 0.0)",
    )
    ap.add_argument(
        "--min-phrase-len-ms",
        type=float,
        default=None,
        help="Minimum phrase note length in ms (default: 0.0)",
    )
    ap.add_argument(
        "--held-vel-mode",
        choices=["first", "max", "mean"],
        default=None,
        help="Velocity for held notes: first, max, or mean accent (default: first)",
    )
    ap.add_argument(
        "--clone-meta-only",
        action="store_true",
        help="Clone only tempo/time-signature from input (best effort across pretty_midi versions)",
    )
    ap.add_argument(
        "--write-mapping-template",
        action="store_true",
        help="Print mapping YAML template to stdout",
    )
    ap.add_argument(
        "--write-mapping-template-path",
        type=str,
        default=None,
        help="Write mapping YAML template to PATH",
    )
    ap.add_argument(
        "--template-style", choices=["full", "minimal"], default="full", help="Template style"
    )
    ap.add_argument("--dry-run", action="store_true", help="Do not write output; log summary")
    ap.add_argument("--quiet", action="store_true", help="Reduce log output")
    ap.add_argument("--verbose", action="store_true", help="Increase log output")
    args, extras = ap.parse_known_args()

    if extras and args.write_mapping_template:
        legacy_tpl_args = []
        while extras and not extras[0].startswith("-"):
            legacy_tpl_args.append(extras.pop(0))
        logging.info(
            "--write-mapping-template with arguments is deprecated; use --template-style/--write-mapping-template-path"
        )
    else:
        legacy_tpl_args = None
    if extras:
        ap.error(f"unrecognized arguments: {' '.join(extras)}")

    if args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level)
    rng = random.Random(args.seed)

    if (
        args.write_mapping_template
        or args.write_mapping_template_path
        or legacy_tpl_args is not None
    ):
        style = args.template_style
        path = args.write_mapping_template_path
        if legacy_tpl_args is not None:
            if legacy_tpl_args and legacy_tpl_args[0] in ("full", "minimal"):
                style = legacy_tpl_args[0]
                legacy_tpl_args = legacy_tpl_args[1:]
            if legacy_tpl_args:
                path = legacy_tpl_args[0]
        content = generate_mapping_template(style == "full")
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
    if swing > 0.0 and abs(swing_unit_beats - pulse_beats) >= EPS:
        logging.info("swing disabled: swing unit %s != pulse %s", args.swing_unit, args.pulse)
        swing = 0.0
    swing = max(0.0, min(float(swing), 0.9))

    mapping = load_mapping(Path(args.mapping) if args.mapping else None)
    global NOTE_ALIASES, NOTE_ALIAS_INV
    NOTE_ALIASES = mapping.get("note_aliases", {})
    NOTE_ALIAS_INV = {v: k for k, v in NOTE_ALIASES.items()}
    cycle_notes_raw = mapping.get("cycle_phrase_notes", [])
    cycle_notes: List[Optional[int]] = []
    for tok in cycle_notes_raw:
        if tok is None:
            cycle_notes.append(None)
        else:
            cycle_notes.append(parse_note_token(tok))
    cycle_start_bar = int(mapping.get("cycle_start_bar", 0))
    cycle_mode = mapping.get("cycle_mode", "bar")
    cycle_stride = int(mapping.get("cycle_stride", 1))
    merge_reset_at = mapping.get("merge_reset_at", "none")
    phrase_channel = mapping.get("phrase_channel")
    chord_channel = mapping.get("chord_channel")
    accent = validate_accent(mapping.get("accent"))
    silent_qualities = mapping.get("silent_qualities", [])
    clone_meta_only = bool(mapping.get("clone_meta_only", False))
    if args.cycle_phrase_notes is not None:
        tokens = [t for t in args.cycle_phrase_notes.split(",") if t.strip()]
        cycle_notes = [parse_note_token(t) for t in tokens]
    if args.cycle_start_bar is not None:
        cycle_start_bar = args.cycle_start_bar
    if args.cycle_mode is not None:
        cycle_mode = args.cycle_mode
    if args.cycle_stride is not None:
        if args.cycle_stride <= 0:
            raise SystemExit("cycle-stride must be >=1")
        cycle_stride = args.cycle_stride
    if args.merge_reset_at is not None:
        merge_reset_at = args.merge_reset_at
    for key, val in (
        ("phrase_channel", args.phrase_channel),
        ("chord_channel", args.chord_channel),
    ):
        if val is not None:
            if not (0 <= val <= 15):
                raise SystemExit(f"{key.replace('_', '-')} must be 0..15")
            if key == "phrase_channel":
                phrase_channel = val
            else:
                chord_channel = val
    if args.accent is not None:
        accent = parse_accent_arg(args.accent)
    sections = mapping.get("sections")
    if args.sections is not None:
        try:
            sections = json.loads(args.sections)
        except Exception:
            raise SystemExit('--sections must be JSON list, e.g. {"start_bar":0}')
    if sections:
        for sec in sections:
            pool = sec.get("phrase_pool") or sec.get("pool")
            if pool:
                sec["phrase_pool" if "phrase_pool" in sec else "pool"] = [
                    parse_note_token(t, warn_unknown=True) for t in pool
                ]
            pbq = sec.get("pool_by_quality")
            if isinstance(pbq, dict):
                for q, lst in list(pbq.items()):
                    pbq[q] = [parse_note_token(t, warn_unknown=True) for t in lst]
            density = sec.get("density")
            if density is not None and density not in ("low", "med", "high"):
                raise SystemExit("sections density must be low|med|high")
    mapping["cycle_phrase_notes"] = cycle_notes
    mapping["cycle_start_bar"] = cycle_start_bar
    mapping["cycle_mode"] = cycle_mode
    mapping["sections"] = sections
    apply_section_preset(mapping, args.section_preset)
    mapping["cycle_stride"] = cycle_stride
    mapping["merge_reset_at"] = merge_reset_at
    mapping["phrase_channel"] = phrase_channel
    mapping["chord_channel"] = chord_channel
    mapping["accent"] = accent
    mapping["sections"] = sections
    mapping["silent_qualities"] = silent_qualities
    section_lfo_cfg = mapping.get("section_lfo")
    if args.section_lfo is not None:
        try:
            section_lfo_cfg = json.loads(args.section_lfo)
        except Exception:
            raise SystemExit('--section-lfo must be JSON e.g. {"period":4}')
    section_lfo_obj = None
    if section_lfo_cfg:
        section_lfo_cfg = validate_section_lfo_cfg(section_lfo_cfg)
        period = int(section_lfo_cfg.get("period", 0))
        vel = section_lfo_cfg.get("vel", [1.0, 1.0])
        fill = section_lfo_cfg.get("fill", [0.0, 0.0])
        section_lfo_obj = SectionLFO(period, vel_range=tuple(vel), fill_range=tuple(fill))
        mapping["section_lfo"] = section_lfo_cfg
    lfo_apply = mapping.get("lfo_apply", ["phrase"])
    if args.lfo_apply is not None:
        try:
            lfo_apply = json.loads(args.lfo_apply)
        except Exception:
            raise SystemExit('--lfo-apply must be JSON list e.g. ["phrase","chord","fill"]')
    mapping["lfo_apply"] = lfo_apply
    mapping["fill_policy"] = args.fill_policy
    stable_cfg = mapping.get("stable_chord_guard")
    if args.stable_guard is not None:
        try:
            stable_cfg = json.loads(args.stable_guard)
        except Exception:
            raise SystemExit('--stable-guard must be JSON e.g. {"min_hold_beats":4}')
    stable_obj = None
    if stable_cfg:
        stable_cfg = validate_stable_guard_cfg(stable_cfg)
        stable_obj = StableChordGuard(
            int(stable_cfg.get("min_hold_beats", 0)), stable_cfg.get("strategy", "skip")
        )
        mapping["stable_chord_guard"] = stable_cfg
    guide_onsets = None
    guide_style_note = NOTE_ALIASES.get("style_fill", 40)
    vocal_cfg = mapping.get("vocal_adapt")
    if args.vocal_adapt is not None:
        try:
            vocal_cfg = json.loads(args.vocal_adapt)
        except Exception:
            raise SystemExit('--vocal-adapt must be JSON e.g. {"dense_onset":4}')
    if args.vocal_guide and vocal_cfg is not None:
        try:
            on, rat = vocal_features_from_midi(args.vocal_guide)
            vocal_cfg["onsets"] = on
            vocal_cfg["ratios"] = rat
        except Exception:
            raise SystemExit("--vocal-guide must be valid MIDI")
    vocal_obj = None
    if args.guide_vocal:
        try:
            on, rat = vocal_features_from_midi(args.guide_vocal)
            guide_onsets = on
            dense_phrase = NOTE_ALIASES.get("open_1_16")
            sparse_phrase = NOTE_ALIASES.get("muted_1_8")
            vocal_obj = VocalAdaptive(
                int(args.guide_onset_th), dense_phrase, sparse_phrase, on, rat
            )
        except Exception:
            raise SystemExit("--guide-vocal must be valid MIDI")
    elif vocal_cfg:
        vocal_cfg = validate_vocal_adapt_cfg(vocal_cfg)
        onsets = vocal_cfg.get("onsets", [])
        ratios = vocal_cfg.get("ratios", [])
        vocal_obj = VocalAdaptive(
            int(vocal_cfg.get("dense_onset", 0)),
            vocal_cfg.get("dense_phrase"),
            vocal_cfg.get("sparse_phrase"),
            onsets,
            ratios,
            vocal_cfg.get("dense_ratio"),
            int(vocal_cfg.get("smooth_bars", 0)),
        )
        mapping["vocal_adapt"] = vocal_cfg
    style_cfg = mapping.get("style_inject")
    if args.style_inject is not None:
        try:
            style_cfg = json.loads(args.style_inject)
        except Exception:
            raise SystemExit('--style-inject must be JSON e.g. {"period":8,"note":30}')
    if style_cfg:
        style_cfg = validate_style_inject_cfg(style_cfg)
        mapping["style_inject"] = style_cfg
    spw = None
    if args.section_pool_weights:
        try:
            raw = json.loads(args.section_pool_weights)
            spw = {str(k): {int(n): float(w) for n, w in v.items()} for k, v in raw.items()}
        except Exception:
            raise SystemExit('--section-pool-weights must be JSON like {"verse":{"36":1.0}}')
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
    mapping["seed"] = args.seed
    clone_meta_only = bool(args.clone_meta_only or clone_meta_only)

    if args.debug_json:
        Path(args.debug_json).write_text(json.dumps(mapping, indent=2))

    if args.chords:
        chord_path = Path(args.chords)
        if chord_path.suffix in {".yaml", ".yml"}:
            chords = read_chords_yaml(chord_path)
        else:
            chords = read_chords_csv(chord_path)
    else:
        chords = infer_chords_by_bar(pm, ts_num, ts_den)

    stats: Dict = {}
    out_pm = build_sparkle_midi(
        pm,
        chords,
        mapping,
        pulse_beats,
        cycle_mode,
        args.humanize_timing_ms,
        args.humanize_vel,
        args.vel_curve,
        bpm,
        swing,
        swing_unit_beats,
        phrase_channel=phrase_channel,
        chord_channel=chord_channel,
        cycle_stride=cycle_stride,
        accent=accent,
        skip_phrase_in_rests=args.skip_phrase_in_rests,
        silent_qualities=silent_qualities,
        clone_meta_only=clone_meta_only,
        stats=stats,
        merge_reset_at=merge_reset_at,
        section_lfo=section_lfo_obj,
        stable_guard=stable_obj,
        fill_policy=args.fill_policy,
        vocal_adapt=vocal_obj,
        vocal_ducking=args.vocal_ducking,
        lfo_targets=tuple(lfo_apply),
        section_pool_weights=spw,
        rng=rng,
        guide_onsets=guide_onsets,
        guide_onset_th=args.guide_onset_th,
        guide_style_note=guide_style_note,
    )
    stats["seed_used"] = args.seed
    if args.debug_md:
        write_debug_md(stats, args.debug_md)
    if args.print_plan and stats.get("bar_reason"):
        for i in sorted(stats["bar_reason"]):
            note = stats["bar_reason"][i]["note"]
            src = stats["bar_reason"][i]["source"]
            alias = NOTE_ALIAS_INV.get(note, str(note) if note is not None else "rest")
            print(f"bar {i}: {alias} | {src}")
    mode, damp_kw = parse_damp_arg(args.damp)
    if mode == "vocal" and vocal_cfg:
        damp_kw.setdefault("vocal_ratios", vocal_cfg.get("ratios"))
        damp_kw.setdefault("downbeats", stats.get("downbeats"))
    emit_damping(out_pm, mode, **damp_kw)
    if args.debug_midi_out:
        dbg = pretty_midi.PrettyMIDI()
        for inst in out_pm.instruments:
            if inst.name == PHRASE_INST_NAME:
                dbg.instruments.append(inst)
        dbg.write(args.debug_midi_out)
    if args.report_json or args.report_md:
        write_reports(stats, args.report_json, args.report_md)
    if args.dry_run:
        phrase_inst = None
        for inst in out_pm.instruments:
            if inst.name == PHRASE_INST_NAME:
                phrase_inst = inst
                break
        bar_pulses = stats.get("bar_pulses", {})
        B = len(bar_pulses)
        P = sum(len(p) for p in bar_pulses.values())
        T = len(phrase_inst.notes) if phrase_inst else 0
        logging.info("bars=%d pulses(theoretical)=%d triggers(emitted)=%d", B, P, T)
        logging.info(
            "phrase_hold=%s phrase_merge_gap=%.3f chord_merge_gap=%.3f phrase_release_ms=%.1f min_phrase_len_ms=%.1f",
            mapping.get("phrase_hold"),
            mapping.get("phrase_merge_gap"),
            mapping.get("chord_merge_gap"),
            mapping.get("phrase_release_ms"),
            mapping.get("min_phrase_len_ms"),
        )
        if stats.get("cycle_disabled"):
            logging.info("cycle disabled; using fixed phrase_note=%d", mapping.get("phrase_note"))
        if stats.get("meters"):
            meters = [(float(t), n, d) for t, n, d in stats["meters"]]
            if stats.get("estimated_4_4"):
                logging.info("meter_map=%s (estimated 4/4 grid)", meters)
            else:
                logging.info("meter_map=%s", meters)
        if mapping.get("cycle_phrase_notes"):
            example = [
                stats["bar_phrase_notes"].get(i) for i in range(min(4, stats.get("bar_count", 0)))
            ]
            logging.info("cycle_mode=%s notes=%s", cycle_mode, example)
        if mapping.get("chord_input_range"):
            logging.info(
                "chord_input_range=%s first_triads=%s",
                mapping.get("chord_input_range"),
                stats.get("triads", [])[:2],
            )
        for i in range(min(4, stats.get("bar_count", 0))):
            pn = stats["bar_phrase_notes"].get(i, mapping.get("phrase_note"))
            pulses = stats["bar_pulses"].get(i, [])
            vels = stats["bar_velocities"].get(i, [])
            if str(mapping.get("phrase_hold")) != "off" and phrase_inst is not None:
                bar_start = stats["downbeats"][i]
                bar_end = (
                    stats["downbeats"][i + 1]
                    if i + 1 < len(stats["downbeats"])
                    else out_pm.get_end_time()
                )
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
