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


# Chord timeline formats:
#   A) Explicit seconds with headers: ``start,end,root,quality``.
#   B) Compact bars with headers: ``bar,chord`` (e.g., ``8,G:maj``).
#   C) Headerless compact bars: ``bar,chord`` per line.
#
# Compact rows expand to ``Root:Quality`` symbols (quality defaults to ``maj``) and
# convert bars to seconds using existing downbeats/meter maps where possible, falling
# back to the provided BPM hint (or 120 BPM) under a 4/4 assumption.


# Chord CSV formats:
#   (A) start,end,root,quality              # explicit timings (seconds)
#   (B) start,chord                         # timings (seconds) + chord symbols
#   (C) bar,chord                           # bar indices; converted via --bpm & default TS (4/4)
#   (D) headerless two columns               # auto-detect: integer -> bar,chord; otherwise start,chord
#
# For (B)-(D) the end time is inferred from the next start (or one bar past the final row).
# Supported chord qualities remain open-ended; common aliases normalise to "maj"/"min".


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
import json
import os
import unicodedata
import io
from fractions import Fraction
from functools import lru_cache
import collections
import itertools
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union, Set, Any, Callable, Iterable, Sequence, Type
from textwrap import dedent
from collections.abc import Mapping

# ujam/sparkle_convert.py のトップレベル付近
from .io import read_chords_yaml  # ← 追加
from .phrase_schedule import ChordSpan  # ← 追加（テストが sc.ChordSpan を使う場合に備え）

# 先頭付近（import 群のあと）
# star import などでテストに流れ込む名前を制御
__all__ = [
    # 公開したい関数・クラスだけを明示（例）
    "ChordSpan",
    "build_sparkle_midi",
    "parse_note_token",
    "parse_midi_note",
    "apply_section_preset",
    "schedule_phrase_keys",
    "vocal_onsets_from_midi",
    "vocal_features_from_midi",
    "main",
]
# ※ 下線始まりの名前は通常 * で公開されませんが、明示しておくと安全

try:
    import pretty_midi  # type: ignore
except Exception as e:
    raise SystemExit("This tool requires pretty_midi. Try: pip install pretty_midi") from e

try:
    import yaml  # optional for mapping file
except Exception:
    yaml = None

from .consts import (
    PHRASE_INST_NAME,
    CHORD_INST_NAME,
    DAMP_INST_NAME,
    PITCH_CLASS,
    SECTION_PRESETS,
)
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

__all__ = [
    "ChordSpan",
    "build_sparkle_midi",
    "parse_midi_note",
    "parse_note_token",
    "apply_section_preset",
    "schedule_phrase_keys",
    "vocal_onsets_from_midi",
    "vocal_features_from_midi",
    "read_chords_yaml",
    "main",
]

NOTE_RE = re.compile(r"^([A-Ga-g])([b#]{0,2})(-?\d+)$")

_DEFAULT_NOTE_ALIASES: Dict[str, int] = {
    "open_1_8": 24,
    "muted_1_8": 25,
    "open_1_16": 26,
}
NOTE_ALIASES: Dict[str, int] = dict(_DEFAULT_NOTE_ALIASES)
NOTE_ALIAS_INV: Dict[int, str] = {v: k for k, v in NOTE_ALIASES.items()}

EPS = 1e-9
MAX_ITERS = 1024

MIN_QPM = 60_000_000.0 / 0xFFFFFF + 1e-6
MAX_QPM = 1000.0

# Set SPARKLE_DETERMINISTIC=1 to force deterministic RNG defaults for tests.
# Opt-in keeps normal randomness intact while letting tests pin results.
_SPARKLE_DETERMINISTIC = os.getenv("SPARKLE_DETERMINISTIC") == "1"


# Helper utilities ---------------------------------------------------------

def _beats_per_bar(num: int, den: int) -> float:
    """Return the theoretical bar length in beats for ``num/den``."""

    if den == 0:
        raise ValueError("time signature denominator must be non-zero")
    return num * (4.0 / den)


def pulses_per_bar(num: int, den: int, unit: float) -> int:
    """Return the pulse count implied by a time signature and subdivision."""

    if unit <= 0.0:
        raise ValueError("pulse subdivision must be positive")
    bar_beats = _beats_per_bar(num, den)
    total = int(round(bar_beats / unit))
    return max(1, total)


def _legacy_pulses_per_bar(num: int, den: int, stats: Optional[Dict[str, Any]]) -> Optional[int]:
    """Return legacy pulse count when requested via ``stats``."""

    if not stats or not stats.get("_legacy_bar_pulses_grid"):
        return None
    if den <= 0:
        raise ValueError("time signature denominator must be positive")
    # 旧バージョンは 16 分音符分解能（および 12/8 などの複合拍子では
    # さらに細かい 24 分割）でグリッドを生成していた。
    pulses = int(round(num * (16.0 / den)))
    if den == 8 and num % 3 == 0:
        # 12/8, 9/8 などでは 8 分音符ベースに倍解像度の 24/18 パルスを出力する。
        legacy_triplet = int(round(num * (8.0 / den))) * 2
        pulses = max(pulses, legacy_triplet)
    return max(1, pulses)


def _insert_fill_note(
    phrase_inst: Optional["pretty_midi.Instrument"],
    mapping: Dict,
    start: float,
    end: float,
    *,
    velocity: Optional[int] = None,
    phrase_merge_gap: float = 0.0,
    release_sec: float = 0.0,
    min_phrase_len_sec: float = 0.0,
    stats: Optional[Dict[str, Any]] = None,
) -> bool:
    """Append a single fill note using ``style_fill`` or ``phrase_note``."""

    if phrase_inst is None:
        return False
    try:
        pitch_val: Optional[int] = None
        raw_style = mapping.get("style_fill") if isinstance(mapping, dict) else None
        if raw_style is not None:
            pitch_val = int(raw_style)
        elif isinstance(mapping, dict) and mapping.get("phrase_note") is not None:
            pitch_val = int(mapping.get("phrase_note"))
    except Exception:
        pitch_val = None
    if pitch_val is None:
        return False
    if velocity is None:
        vel_candidate = None
        if isinstance(mapping, dict):
            vel_candidate = mapping.get("phrase_velocity")
        try:
            velocity = int(vel_candidate) if vel_candidate is not None else 96
        except Exception:
            velocity = 96
    velocity = max(1, min(127, int(velocity)))
    start_f = float(start)
    end_f = float(end)
    if not math.isfinite(start_f) or not math.isfinite(end_f):
        return False
    if end_f <= start_f + EPS:
        end_f = start_f + max(1e-3, end_f - start_f + 1e-3)
    _append_phrase(
        phrase_inst,
        int(pitch_val),
        start_f,
        end_f,
        velocity,
        phrase_merge_gap,
        release_sec,
        min_phrase_len_sec,
        stats,
    )
    return True


def _safe_int(val: Any) -> Optional[int]:
    """Return ``int(val)`` if possible; otherwise ``None``."""

    if isinstance(val, bool):  # pragma: no cover - defensive
        return int(val)
    try:
        return int(val)
    except Exception:
        return None


def _note_name_to_midi(tok: str) -> Optional[int]:
    """Resolve note names like ``C3`` (including Unicode accidentals) to MIDI ints."""

    if not isinstance(tok, str):
        return None
    token = tok.strip()
    if not token:
        return None
    token = unicodedata.normalize("NFKC", token)
    token = (
        token.replace("♭", "b")
        .replace("♯", "#")
        .replace("ｂ", "b")
        .replace("＃", "#")
        .replace("𝄫", "bb")
        .replace("𝄪", "##")
    )
    match = NOTE_RE.match(token)
    if not match:
        return None
    letter, accidental, octave_raw = match.groups()
    try:
        octave = int(octave_raw)
    except Exception:
        return None
    base = {
        "C": 0,
        "D": 2,
        "E": 4,
        "F": 5,
        "G": 7,
        "A": 9,
        "B": 11,
    }
    semitone = base[letter.upper()]
    if accidental == "#":
        semitone += 1
    elif accidental == "##":
        semitone += 2
    elif accidental == "b":
        semitone -= 1
    elif accidental == "bb":
        semitone -= 2
    return semitone + (octave + 1) * 12


def _resolve_pitch_token(tok: Any, mapping: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Resolve various pitch token forms to an integer MIDI pitch."""

    value = tok

    if isinstance(value, dict) and "pitch" in value:
        value = value.get("pitch")

    if hasattr(value, "pitch"):
        pitch_attr = getattr(value, "pitch")
        resolved = _safe_int(pitch_attr)
        if resolved is not None:
            return resolved
        value = pitch_attr

    if isinstance(value, (int, float)):
        try:
            iv = int(round(float(value)))
        except Exception:
            iv = None
        if iv is not None:
            return max(0, min(127, iv))

    resolved = _safe_int(value)
    if resolved is not None:
        return resolved

    if isinstance(value, str):
        token = value.strip()
        num_val = _safe_int(token)
        if num_val is not None:
            return num_val

        if mapping:
            for key in (
                "phrase_note_map",
                "phrase_aliases",
                "aliases",
                "note_aliases",
                "phrase_token_pitch",
            ):
                mp = mapping.get(key)
                if isinstance(mp, dict) and token in mp:
                    mapped = _resolve_pitch_token(mp[token], None)
                    if mapped is not None:
                        return mapped

        alias = NOTE_ALIASES.get(token)
        if alias is not None:
            alias_resolved = _resolve_pitch_token(alias, None)
            if alias_resolved is not None:
                return alias_resolved

        note_val = _note_name_to_midi(token)
        if note_val is not None:
            return note_val

    return None


def resolve_phrase_alias(name_or_int: Any, mapping: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Resolve phrase tokens from mapping pools or aliases to concrete MIDI pitches."""

    resolved = _resolve_pitch_token(name_or_int, mapping)
    if resolved is not None:
        return resolved

    if not isinstance(name_or_int, str):
        return None

    token = name_or_int.strip()
    if not token:
        return None

    def _iter_pool_entries(pool: Any) -> Iterable[Tuple[List[str], Any]]:
        if pool is None:
            return
        if isinstance(pool, dict):
            for key in ("pool", "notes", "entries"):
                if key in pool:
                    yield from _iter_pool_entries(pool[key])
            # treat mapping of name -> value directly
            for key, value in pool.items():
                if key in {"pool", "notes", "weights", "entries", "T"}:
                    continue
                if isinstance(key, str):
                    yield [key], value
        elif isinstance(pool, list):
            for item in pool:
                if isinstance(item, dict):
                    names: List[str] = []
                    for name_key in ("name", "label", "alias", "token", "id"):
                        val = item.get(name_key)
                        if isinstance(val, str) and val.strip():
                            names.append(val.strip())
                    pitch_val = item.get("note")
                    if pitch_val is None:
                        pitch_val = item.get("pitch")
                    if pitch_val is None:
                        pitch_val = item.get("value")
                    if pitch_val is not None:
                        yield names or [str(pitch_val)], pitch_val
                elif isinstance(item, (tuple, list)) and item:
                    first = item[0]
                    names = [str(first)] if isinstance(first, str) else []
                    yield names or [str(first)], first
                else:
                    yield [str(item)], item

    if mapping:
        pool_cfg = mapping.get("phrase_pool")
        for names, target in _iter_pool_entries(pool_cfg):
            for name in names:
                if name.strip() == token:
                    candidate = _resolve_pitch_token(target, mapping)
                    if candidate is not None:
                        return candidate
        phrase_map = mapping.get("phrase_aliases")
        if isinstance(phrase_map, dict):
            for alias_name, value in phrase_map.items():
                if isinstance(alias_name, str) and alias_name.strip() == token:
                    candidate = _resolve_pitch_token(value, mapping)
                    if candidate is not None:
                        return candidate

    return None


def finalize_song_length(
    pm_out: "pretty_midi.PrettyMIDI",
    downbeats: Optional[Sequence[float]],
    sections: Optional[Sequence[Any]],
    fallback_end: Optional[float],
) -> Tuple[Optional[float], int]:
    """Clamp instrument note ends to the computed song end time."""

    def _extract_end_bar(items: Optional[Sequence[Any]]) -> Optional[int]:
        if not items:
            return None
        for entry in reversed(list(items)):
            if isinstance(entry, Mapping):
                end_bar = entry.get("end_bar")
                if end_bar is None:
                    continue
                try:
                    idx = int(end_bar)
                except Exception:
                    continue
                if idx >= 0:
                    return idx
        return None

    downbeats_list: List[float] = []
    if downbeats:
        for t in downbeats:
            try:
                downbeats_list.append(float(t))
            except Exception:
                continue

    song_end_time: Optional[float] = None
    end_bar = _extract_end_bar(sections)
    if end_bar is not None and downbeats_list:
        if end_bar >= len(downbeats_list):
            logging.warning(
                "finalize_song_length: section end bar %d exceeds downbeats %d; clamping",
                end_bar,
                len(downbeats_list) - 1,
            )
            end_bar = len(downbeats_list) - 1
        if end_bar >= 0:
            song_end_time = downbeats_list[end_bar]
    elif fallback_end is not None and math.isfinite(fallback_end):
        song_end_time = float(fallback_end)
    elif downbeats_list:
        song_end_time = downbeats_list[-1]

    if song_end_time is None or not math.isfinite(song_end_time):
        return (None, 0)

    clamp_to = max(0.0, song_end_time - EPS)
    notes_clipped = 0
    for inst in getattr(pm_out, "instruments", []) or []:
        for note in getattr(inst, "notes", []) or []:
            try:
                start = float(getattr(note, "start", 0.0))
                end = float(getattr(note, "end", start))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(end):
                continue
            if end > clamp_to + EPS:
                note.end = clamp_to if clamp_to >= start else start
                notes_clipped += 1
    if notes_clipped:
        logging.info(
            "finalize_song_length: clipped %d note(s) at %.3fs",
            notes_clipped,
            song_end_time,
        )
    return (song_end_time, notes_clipped)


def _coerce_note_alias_map(source: Any) -> Dict[str, Any]:
    """Return a dict copy of ``source`` if it behaves like a mapping."""

    if isinstance(source, Mapping):
        try:
            return dict(source)
        except Exception:
            return {str(k): source[k] for k in source}
    if hasattr(source, "items"):
        try:
            return dict(source.items())
        except Exception:
            pass
    if isinstance(source, list):
        try:
            return {str(k): v for k, v in source}
        except Exception:
            pass
    return {}


def _current_note_alias_maps() -> Tuple[Dict[str, Any], Dict[int, str]]:
    """Return (alias_by_name, alias_by_midi) using current globals."""

    global NOTE_ALIAS_INV
    alias_map = dict(_DEFAULT_NOTE_ALIASES)
    override_map = _coerce_note_alias_map(NOTE_ALIASES)
    if override_map:
        alias_map.update(override_map)
    inv_source = NOTE_ALIAS_INV
    if isinstance(inv_source, Mapping):
        alias_inv: Dict[int, str] = dict(inv_source)
    elif hasattr(inv_source, "items"):
        try:
            alias_inv = dict(inv_source.items())
        except Exception:
            alias_inv = {}
    else:
        alias_inv = {}

    updated = False
    for name, target in alias_map.items():
        midi_val = _resolve_pitch_token(target, None)
        if midi_val is None:
            continue
        if midi_val not in alias_inv:
            alias_inv[midi_val] = name
            updated = True

    if updated:
        NOTE_ALIAS_INV = alias_inv

    return alias_map, alias_inv


def build_avoid_set(
    mapping: Dict[str, Any], explicit_avoid: Optional[Iterable[Any]] = None
) -> Set[int]:
    """Return a set of MIDI pitches to avoid based on mapping/explicit tokens."""

    tokens: List[Any] = []

    def _extend(source: Any) -> None:
        if source is None:
            return
        if isinstance(source, (list, tuple, set)):
            tokens.extend(source)
        else:
            tokens.append(source)

    _extend(mapping.get("phrase_note"))
    _extend(mapping.get("cycle_phrase_notes"))
    _extend(mapping.get("style_fill"))
    if explicit_avoid is not None:
        _extend(list(explicit_avoid))

    resolved: Set[int] = set()
    for tok in tokens:
        val = _resolve_pitch_token(tok, mapping)
        if val is not None:
            resolved.add(val)
    return resolved


def _seed_tick_tables(pm: "pretty_midi.PrettyMIDI", qpm: float) -> bool:
    """Seed PrettyMIDI tick tables using a sanitized tempo; return True if privates touched."""

    try:
        safe_qpm = float(qpm)
    except Exception:
        safe_qpm = 120.0
    if not math.isfinite(safe_qpm) or safe_qpm <= 0.0:
        safe_qpm = 120.0
    safe_qpm = max(MIN_QPM, min(safe_qpm, MAX_QPM))

    resolution = getattr(pm, "resolution", 220)
    try:
        resolution = float(resolution)
    except Exception:
        resolution = 220.0
    if not math.isfinite(resolution) or resolution <= 0.0:
        resolution = 220.0

    seconds_per_tick = 60.0 / (safe_qpm * resolution)
    if not math.isfinite(seconds_per_tick) or seconds_per_tick <= 0.0:
        seconds_per_tick = 60.0 / (120.0 * resolution)

    used_private = False
    if hasattr(pm, "_tick_scales"):
        try:
            pm._tick_scales = [(0, float(seconds_per_tick))]  # type: ignore[attr-defined]
            used_private = True
        except Exception:
            pass

    tick_attrs = ("_PrettyMIDI__tick_to_time", "_tick_to_time")
    for attr in tick_attrs:
        if hasattr(pm, attr):
            try:
                setattr(pm, attr, [0.0])
                used_private = True
            except Exception:
                pass
            break

    end_time = 0.0
    if hasattr(pm, "get_end_time"):
        try:
            end_time = float(pm.get_end_time())
        except Exception:
            end_time = 0.0
    target = max(0.0, end_time)
    try:
        pm.time_to_tick(target)
    except Exception:
        try:
            pm.time_to_tick(0.0)
        except Exception:
            pass

    return used_private


def _ensure_tempo_and_ticks(
    pm: "pretty_midi.PrettyMIDI", seed_bpm: float, ts_changes: Optional[List] = None
) -> None:
    """Ensure tempo/tick tables exist before writing PrettyMIDI outputs."""

    try:
        bpm = float(seed_bpm)
    except Exception:
        bpm = 120.0
    if not math.isfinite(bpm) or bpm <= 0.0:
        bpm = 120.0

    ts_seq = ts_changes
    if ts_seq is None:
        ts_seq = getattr(pm, "time_signature_changes", None)
    if ts_seq and not getattr(pm, "_sparkle_ts_seeded", False):
        seq_list = list(ts_seq)
        first = seq_list[0]
        try:
            first_time = float(getattr(first, "time", 0.0))
        except Exception:
            first_time = 0.0
        num = getattr(first, "numerator", 4)
        den = getattr(first, "denominator", 4)
        if first_time > EPS:
            exists_at_zero = False
            for ts in seq_list:
                try:
                    ts_time = float(getattr(ts, "time", 0.0))
                except Exception:
                    ts_time = 0.0
                if abs(ts_time) <= EPS and getattr(ts, "numerator", num) == num and getattr(
                    ts, "denominator", den
                ) == den:
                    exists_at_zero = True
                    break
            if not exists_at_zero:
                ts_cls = getattr(pretty_midi, "TimeSignature", None)
                if ts_cls is not None:
                    try:
                        pm.time_signature_changes.insert(0, ts_cls(int(num), int(den), 0.0))
                    except Exception:
                        pass
        setattr(pm, "_sparkle_ts_seeded", True)

    try:
        tempo_times, tempi_seq = pm.get_tempo_changes()
    except Exception:
        tempo_times, tempi_seq = [], []

    def _has_items(obj: Any) -> bool:
        if obj is None:
            return False
        length = getattr(obj, "__len__", None)
        if callable(length):
            try:
                return length() > 0
            except Exception:
                pass
        size = getattr(obj, "size", None)
        if size is not None:
            try:
                return int(size) > 0
            except Exception:
                pass
        try:
            return bool(obj)
        except Exception:
            return False

    has_tempi = _has_items(tempi_seq)
    try:
        tempi_list = [float(t) for t in tempi_seq]
    except Exception:
        tempi_list = []
        has_tempi = False
    current_initial = getattr(pm, "initial_tempo", None)
    has_initial = False
    init_candidate = bpm
    if current_initial is not None:
        try:
            init_candidate = float(current_initial)
            has_initial = math.isfinite(init_candidate) and init_candidate > 0.0
        except Exception:
            init_candidate = bpm
            has_initial = False
    first_qpm: Optional[float] = None
    if has_tempi and tempi_list:
        qpm0 = tempi_list[0]
        if not math.isfinite(qpm0) or qpm0 <= 0.0:
            qpm0 = bpm
        first_qpm = max(MIN_QPM, min(qpm0, MAX_QPM))
    else:
        if not has_initial:
            init_candidate = bpm
        if not math.isfinite(init_candidate) or init_candidate <= 0.0:
            init_candidate = bpm
        first_qpm = max(MIN_QPM, min(init_candidate, MAX_QPM))
    pm.initial_tempo = float(first_qpm)

    setattr(pm, "_sparkle_meta_seed_fallback", False)
    used_private = _seed_tick_tables(pm, first_qpm)
    if used_private:
        setattr(pm, "_sparkle_meta_seed_fallback", True)


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    """Return ``value`` rounded to the nearest int and clamped to ``[lo, hi]``."""

    try:
        candidate = int(round(value))
    except Exception:
        return default
    if candidate < lo:
        return lo
    if candidate > hi:
        return hi
    return candidate


def _sanitize_midi_for_mido(pm: "pretty_midi.PrettyMIDI") -> None:
    """Clamp MIDI data bytes so ``pretty_midi.PrettyMIDI.write`` never feeds mido floats."""

    for inst in getattr(pm, "instruments", []) or []:
        for note in getattr(inst, "notes", []) or []:
            note.pitch = _clamp_int(note.pitch, 0, 127, 0)
            note.velocity = _clamp_int(note.velocity, 0, 127, 64)
        for cc in getattr(inst, "control_changes", []) or []:
            cc.number = _clamp_int(cc.number, 0, 127, 0)
            cc.value = _clamp_int(cc.value, 0, 127, 0)
        for pb in getattr(inst, "pitch_bends", []) or []:
            pb.pitch = _clamp_int(pb.pitch, -8192, 8191, 0)
        inst.program = _clamp_int(getattr(inst, "program", 0), 0, 127, 0)
        if hasattr(inst, "midi_channel") and inst.midi_channel is not None:
            inst.midi_channel = _clamp_int(inst.midi_channel, 0, 15, 0)


def _sanitize_tempi(pm: "pretty_midi.PrettyMIDI") -> None:
    """Clamp PrettyMIDI tempi so PrettyMIDI.write() never exceeds mido's MPQN range."""

    def _as_list(seq: Any) -> List[Any]:
        if seq is None:
            return []
        if isinstance(seq, list):
            return list(seq)
        try:
            return list(seq)
        except TypeError:
            tolist = getattr(seq, "tolist", None)
            if callable(tolist):
                try:
                    return list(tolist())
                except Exception:
                    pass
            return [seq]

    try:
        raw_times, raw_tempi = pm.get_tempo_changes()
    except Exception:
        raw_times, raw_tempi = [], []

    raw_time_list: List[float] = []
    for item in _as_list(raw_times):
        try:
            raw_time_list.append(float(item))
        except Exception:
            raw_time_list.append(0.0)

    raw_tempo_list: List[float] = []
    for item in _as_list(raw_tempi):
        try:
            raw_tempo_list.append(float(item))
        except Exception:
            raw_tempo_list.append(float("nan"))

    if not raw_tempo_list:
        try:
            init = float(getattr(pm, "initial_tempo", 120.0))
        except Exception:
            init = 120.0
        if not math.isfinite(init) or init <= 0.0:
            init = 120.0
        safe_init = max(MIN_QPM, min(init, MAX_QPM))
        pm.initial_tempo = safe_init
        _seed_tick_tables(pm, safe_init)
        return

    safe_changes: List[Tuple[float, float]] = []
    changed = False
    last_time = 0.0
    have_time = False

    for idx, raw_qpm in enumerate(raw_tempo_list):
        safe_qpm = raw_qpm
        invalid_qpm = False
        if not math.isfinite(safe_qpm) or safe_qpm <= 0.0:
            safe_qpm = 120.0
            invalid_qpm = True
        safe_qpm = max(MIN_QPM, min(safe_qpm, MAX_QPM))
        if invalid_qpm or not math.isfinite(raw_qpm) or raw_qpm <= 0.0:
            changed = True
        elif abs(safe_qpm - raw_qpm) > EPS:
            changed = True

        try:
            t_val = raw_time_list[idx]
        except IndexError:
            t_val = last_time if have_time else 0.0
        try:
            safe_t = float(t_val)
        except Exception:
            safe_t = last_time if have_time else 0.0
        if safe_t < 0.0:
            safe_t = 0.0
        if have_time and safe_t < last_time - EPS:
            safe_t = last_time
            changed = True

        if have_time and abs(safe_t - last_time) <= EPS:
            prev_time, prev_qpm = safe_changes[-1]
            if abs(prev_qpm - safe_qpm) > EPS:
                safe_changes[-1] = (prev_time, float(safe_qpm))
                changed = True
            elif invalid_qpm:
                safe_changes[-1] = (prev_time, float(safe_qpm))
            continue

        if abs(safe_t - t_val) > EPS:
            changed = True

        safe_changes.append((float(safe_t), float(safe_qpm)))
        last_time = safe_t
        have_time = True

    if not safe_changes:
        safe_init = max(MIN_QPM, min(120.0, MAX_QPM))
        pm.initial_tempo = safe_init
        _seed_tick_tables(pm, safe_init)
        return

    remover = getattr(pm, "remove_tempo_changes", None)
    adder_public = getattr(pm, "add_tempo_change", None)
    applied = False
    if changed and callable(remover) and callable(adder_public):
        try:
            remover()
            for tt, qq in safe_changes:
                adder_public(qq, tt)
            applied = True
        except Exception:
            applied = False
    if not applied:
        adder_private = getattr(pm, "_add_tempo_change", None)
        if callable(adder_private):
            try:
                if callable(remover):
                    try:
                        remover()
                    except Exception:
                        pass
                elif hasattr(pm, "_tempo_changes"):
                    try:
                        pm._tempo_changes = []  # type: ignore[attr-defined]
                    except Exception:
                        pass
                for tt, qq in safe_changes:
                    adder_private(qq, tt)
                applied = True
            except Exception:
                applied = False
    if not applied and hasattr(pm, "_tempo_changes"):
        try:
            tempo_cls = None
            if hasattr(pretty_midi, "TempoChange"):
                tempo_cls = pretty_midi.TempoChange  # type: ignore[attr-defined]
            else:
                containers = getattr(pretty_midi, "containers", None)
                if containers is not None:
                    tempo_cls = getattr(containers, "TempoChange", None)
            if tempo_cls is not None:
                pm._tempo_changes = [  # type: ignore[attr-defined]
                    tempo_cls(qq, tt) for tt, qq in safe_changes
                ]
            else:
                pm._tempo_changes = [  # type: ignore[attr-defined]
                    (qq, tt) for tt, qq in safe_changes
                ]
            applied = True
        except Exception:
            applied = False
    if not applied and changed:
        pm.initial_tempo = safe_changes[0][1]

    try:
        current_initial = float(getattr(pm, "initial_tempo", safe_changes[0][1]))
    except Exception:
        current_initial = safe_changes[0][1]
    if not math.isfinite(current_initial) or current_initial <= 0.0:
        pm.initial_tempo = safe_changes[0][1]
    else:
        clamped_initial = max(MIN_QPM, min(current_initial, MAX_QPM))
        if abs(clamped_initial - current_initial) > EPS:
            pm.initial_tempo = clamped_initial

    _seed_tick_tables(pm, safe_changes[0][1])


def _seed_input_grid(pm: "pretty_midi.PrettyMIDI", bpm: float, end_sec: float) -> None:
    """Seed *pm* with a tempo + silent note so ``get_beats`` succeeds."""

    tempo = float(bpm) if math.isfinite(bpm) and bpm > 0.0 else 120.0
    end = max(1.0, float(end_sec))

    if hasattr(pm, "_add_tempo_change"):
        pm._add_tempo_change(tempo, 0.0)
    else:  # pragma: no cover - legacy pretty_midi fallback
        pm.initial_tempo = tempo

    seed_inst = None
    for inst in pm.instruments:
        if getattr(inst, "name", "") == "__seed_input_grid__":
            seed_inst = inst
            break
    if seed_inst is None:
        seed_inst = pretty_midi.Instrument(program=0, name="__seed_input_grid__")
        pm.instruments.append(seed_inst)

    for note in seed_inst.notes:
        if note.pitch == 0 and note.velocity == 1 and abs(note.start) < 1e-9:
            note.end = max(note.end, end)
            break
    else:
        seed_inst.notes.append(pretty_midi.Note(velocity=1, pitch=0, start=0.0, end=end))


def clip_to_bar(beat_pos: float, bar_start: float, bar_end: float) -> float:
    """Clamp ``beat_pos`` into the inclusive start / exclusive end of a bar."""

    upper = bar_end - EPS
    if beat_pos < bar_start:
        return bar_start
    if beat_pos > upper:
        return upper
    return beat_pos


def resolve_downbeats(
    pm: "pretty_midi.PrettyMIDI",
    meter_map: List[Tuple[float, int, int]],
    beat_times: List[float],
    beat_to_time: Callable[[float], float],
    time_to_beat: Callable[[float], float],
    *,
    allow_meter_mismatch: bool = False,
) -> List[float]:
    """Return sorted downbeat times including the final track end.

    PrettyMIDI sometimes yields downbeats derived from tempo only; when the
    spacing disagrees with the time-signature map we rebuild the list from the
    meter changes so downstream logic sees a consistent, duplicate-free grid.
    """

    if not meter_map:
        raise ValueError("meter_map must contain at least one entry")

    end_t = pm.get_end_time()
    downbeats = list(pm.get_downbeats())
    rebuild = len(downbeats) < 2
    if len(meter_map) > 1 and not rebuild and not allow_meter_mismatch:
        rebuild = True
    if not rebuild and not allow_meter_mismatch:
        # If PrettyMIDI already places downbeats exactly at meter-change boundaries,
        # prefer trusting it even when coarse spacing differs from theoretical
        # bar length (e.g., PM may emit only a single 6/8 bar before a TS change).
        change_times = [mt for mt, _, _ in meter_map[1:]]
        has_all_boundaries = all(
            any(abs(db - mt) <= 1e-6 for db in downbeats) for mt in change_times
        )
        if not has_all_boundaries:
            num0, den0 = meter_map[0][1], meter_map[0][2]
            bar_beats = num0 * (4.0 / den0)
            start_b = time_to_beat(downbeats[0])
            next_b = time_to_beat(downbeats[1])
            actual_beats = next_b - start_b
            if not math.isclose(
                actual_beats, bar_beats, rel_tol=1e-6, abs_tol=1e-9
            ):
                rebuild = True

    if not rebuild and not allow_meter_mismatch:
        meter_times = [mt for mt, _, _ in meter_map]
        for idx in range(len(downbeats) - 1):
            start = downbeats[idx]
            end = downbeats[idx + 1]
            # Ignore spans that contain an intermediate meter change; the
            # subsequent iteration anchored at the boundary will validate that
            # region instead.
            if any(
                mt > start + EPS and mt < end - EPS
                for mt in meter_times[1:]
            ):
                continue
            num, den = get_meter_at(meter_map, start, times=meter_times)
            expected_beats = num * (4.0 / den) if den else 0.0
            if expected_beats <= 0.0:
                continue
            start_b = time_to_beat(start)
            end_b = time_to_beat(end)
            actual_beats = end_b - start_b
            tol = max(1e-6, expected_beats * 1e-6)
            if (
                idx + 1 == len(downbeats) - 1
                and abs(end_t - end) <= 1e-6
                and actual_beats + tol < expected_beats
            ):
                # Final bar may be truncated; accept shorter spans at the tail.
                continue
            if not math.isclose(actual_beats, expected_beats, rel_tol=1e-6, abs_tol=tol):
                rebuild = True
                break

    if rebuild:
        meter_entries = [(time_to_beat(mt), num, den) for mt, num, den in meter_map]
        meter_entries.sort(key=lambda item: item[0])
        if not meter_entries:
            return [0.0, float(end_t)]
        end_beat = time_to_beat(end_t)
        downbeats_beats: List[float] = []
        start_b = meter_entries[0][0]
        downbeats_beats.append(start_b)
        meter_entries.append((end_beat, meter_entries[-1][1], meter_entries[-1][2]))
        for (seg_start_b, num, den), (seg_end_b, next_num, next_den) in zip(
            meter_entries, meter_entries[1:]
        ):
            bar_beats = _beats_per_bar(num, den)
            if bar_beats <= 0:
                continue
            # Treat the meter-change boundary itself as the next bar head without
            # rounding adjustments so the following segment starts exactly at the
            # change point.
            if downbeats_beats[-1] < seg_start_b - EPS:
                downbeats_beats.append(seg_start_b)
            cur = seg_start_b
            while cur < seg_end_b - EPS:
                remaining = seg_end_b - cur
                if remaining + EPS < bar_beats and (num != next_num or den != next_den):
                    break
                if abs(cur - downbeats_beats[-1]) > EPS:
                    downbeats_beats.append(cur)
                cur += bar_beats
            if downbeats_beats[-1] < seg_end_b - EPS:
                downbeats_beats.append(seg_end_b)
            else:
                downbeats_beats[-1] = seg_end_b
        downbeats = []
        for beat_val in downbeats_beats:
            t = beat_to_time(beat_val)
            if not downbeats or abs(t - downbeats[-1]) > EPS:
                downbeats.append(t)
        if not downbeats:
            downbeats = [beat_to_time(meter_entries[0][0]), float(end_t)]

    downbeats.sort()
    unique: List[float] = []
    for t in downbeats:
        ft = float(t)
        if not unique or abs(ft - unique[-1]) > EPS:
            unique.append(ft)
    if not unique:
        unique.append(0.0)
    last = unique[-1]
    if last < end_t - EPS:
        unique.append(float(end_t))
    else:
        unique[-1] = float(end_t)
    assert abs(end_t - unique[-1]) <= 1e-6
    return unique


@lru_cache(None)
def _cached_meter_entry(
    meter_key: Tuple[Tuple[float, int, int], ...],
    idx: int,
) -> Tuple[int, int]:
    """Return ``(numerator, denominator)`` for ``meter_key[idx]`` with clamping."""

    if not meter_key:
        raise ValueError("meter_map must contain at least one entry")
    if idx < 0:
        idx = 0
    elif idx >= len(meter_key):
        idx = len(meter_key) - 1
    _, num, den = meter_key[idx]
    return num, den


def get_meter_at(
    meter_map: List[Tuple[float, int, int]],
    t: float,
    *,
    times: Optional[List[float]] = None,
) -> Tuple[int, int]:
    """Return the meter active at time ``t`` using bisect over change times."""

    if not meter_map:
        raise ValueError("meter_map must contain at least one entry")
    use_times_seq: Union[List[float], Tuple[float, ...]]
    if times is not None:
        use_times_seq = times
    else:
        use_times_seq = [mt for mt, _, _ in meter_map]
    if len(use_times_seq) >= 2:
        assert all(
            earlier <= later + EPS for earlier, later in zip(use_times_seq, use_times_seq[1:])
        ), "meter change times must be non-decreasing"
    idx = bisect.bisect_right(use_times_seq, t + EPS) - 1
    return _cached_meter_entry(tuple(meter_map), idx)


# Lightweight PrettyMIDI stub for tests and dry runs
def _pm_dummy_for_docs(length: float = 6.0):
    """Return a minimal PrettyMIDI-like object for unit tests.

    Provides instruments, tempo/beat helpers, and write().
    """
    try:
        pm_mod = pretty_midi
    except Exception:  # pragma: no cover - pretty_midi should exist
        pm_mod = None  # type: ignore

    class Dummy:
        def __init__(self, length: float) -> None:
            self._length = float(length)
            inst = (
                pm_mod.Instrument(0) if pm_mod else type("I", (), {"notes": [], "is_drum": False})()
            )
            if pm_mod:
                inst.notes.append(pm_mod.Note(velocity=1, pitch=60, start=0.0, end=float(length)))
                inst.is_drum = False
            self.instruments = [inst]
            self.time_signature_changes = []

        def get_beats(self):
            step = 0.5  # 120 bpm
            n = int(self._length / step) + 1
            return [i * step for i in range(n)]

        def get_downbeats(self):
            return self.get_beats()[::4]

        def get_end_time(self):
            return self._length

        def get_tempo_changes(self):
            return [0.0], [120.0]

        def write(self, path: str) -> None:  # pragma: no cover
            Path(path).write_bytes(b"")

    return Dummy(length)


@dataclass
class RuntimeContext:
    """Container for runtime state used during phrase emission.

    Attributes
    ----------
    rng:
        Random number generator for humanization.
    section_lfo:
        Optional low-frequency oscillator scaling velocities per bar.
    humanize_ms:
        Timing jitter range in milliseconds.
    humanize_vel:
        Velocity jitter range.
    beat_to_time, time_to_beat:
        Conversion helpers between beat indices and seconds.
    clip:
        Function clamping note intervals.
    maybe_merge_gap:
        Function deciding merge gaps for adjacent notes.
    append_phrase:
        Callback appending phrase notes to the output instrument.
    vel_factor:
        Function computing per-pulse velocity scaling.
    accent_by_bar, bar_counts, preset_by_bar:
        Immutable per-bar caches.
    accent_scale_by_bar:
        Optional per-bar velocity scaling factors.
    vel_curve:
        Velocity curve mode.
    downbeats:
        List of bar start times in seconds.
    cycle_mode:
        Phrase cycle mode ("bar" or "chord").
    phrase_len_beats:
        Nominal phrase length in beats.
    phrase_inst:
        Destination instrument for phrase notes.
    pick_phrase_note:
        Callback choosing which phrase note to emit.
    release_sec, min_phrase_len_sec:
        Timing constants forwarded to ``append_phrase``.
    phrase_vel:
        Base phrase velocity before scaling.
    duck:
        Function applying vocal ducking to velocities.
    lfo_targets:
        Tuple of streams affected by the section LFO.
    stable_guard:
        Optional guard suppressing rapid chord changes.
    stats:
        Mutable statistics dictionary (updated in place).
    bar_progress:
        Mutable pulse counters per bar.
    pulse_subdiv_beats, swing, swing_unit_beats:
        Timing parameters for pulse scheduling.
    EPS:
        Minimal interval constant.
    """

    rng: random.Random
    section_lfo: Optional[SectionLFO]
    humanize_ms: float
    humanize_vel: float
    beat_to_time: Callable[[float], float]
    time_to_beat: Callable[[float], float]
    clip: Callable[[float, float], Tuple[float, float]]
    maybe_merge_gap: Callable[..., float]
    append_phrase: Callable[..., None]
    vel_factor: Callable[[str, int, int], float]
    accent_by_bar: Dict[int, Optional[List[float]]]
    bar_counts: Dict[int, int]
    preset_by_bar: Dict[int, Dict[str, float]]
    accent_scale_by_bar: Dict[int, float]
    vel_curve: str
    downbeats: List[float]
    cycle_mode: str
    phrase_len_beats: float
    phrase_inst: "pretty_midi.Instrument"
    pick_phrase_note: Callable[[float, int], Optional[int]]
    release_sec: float
    min_phrase_len_sec: float
    phrase_vel: int
    duck: Callable[[int, int], int]
    lfo_targets: Tuple[str, ...]
    stable_guard: Optional[StableChordGuard]
    stats: Optional[Dict]
    bar_progress: Dict[int, int]
    pulse_subdiv_beats: float
    swing: float
    swing_unit_beats: float
    swing_shape: str
    EPS: float = EPS


def _emit_phrases_for_span(span: "ChordSpan", c_idx: int, ctx: RuntimeContext) -> None:
    """Emit phrase triggers for a single chord span.

    Parameters
    ----------
    span:
        Chord span defining start/end times and quality.
    c_idx:
        Index of the span within the chord list.
    ctx:
        RuntimeContext carrying helpers and precomputed state.

    Side Effects
    ------------
    Appends phrase notes to ``ctx.phrase_inst`` and updates ``ctx.stats`` and
    ``ctx.bar_progress``. Returns ``None``.
    """

    sb = ctx.time_to_beat(span.start)
    eb = ctx.time_to_beat(span.end)
    b = sb
    iter_guard = MAX_ITERS
    # Guard to avoid pathological zero-advance loops when phrasing math collapses.
    while b < eb - ctx.EPS:
        t = ctx.beat_to_time(b)
        bar_idx = max(0, bisect.bisect_right(ctx.downbeats, t) - 1)
        preset = ctx.preset_by_bar.get(bar_idx, DENSITY_PRESETS["med"])
        total = ctx.bar_counts.get(bar_idx, 1)
        idx = ctx.bar_progress.get(bar_idx, 0)
        ctx.bar_progress[bar_idx] = idx + 1
        vf = ctx.vel_factor(ctx.vel_curve, idx, total)
        acc_arr = ctx.accent_by_bar.get(bar_idx)
        scale = ctx.accent_scale_by_bar.get(bar_idx, 1.0)
        af = (acc_arr[idx % len(acc_arr)] if acc_arr else 1.0) * preset["accent"] * scale
        base_vel = max(1, min(127, int(round(ctx.phrase_vel * vf * af))))
        base_vel = ctx.duck(bar_idx, base_vel)
        if ctx.section_lfo and "phrase" in ctx.lfo_targets:
            base_vel = max(1, min(127, int(round(base_vel * ctx.section_lfo.vel_scale(bar_idx)))))
            if ctx.stats is not None:
                ctx.stats.setdefault("lfo_pos", {})[bar_idx] = ctx.section_lfo._pos(bar_idx)
        if ctx.humanize_vel > 0:
            base_vel = max(
                1,
                min(
                    127, int(round(base_vel + ctx.rng.uniform(-ctx.humanize_vel, ctx.humanize_vel)))
                ),
            )
        interval = ctx.pulse_subdiv_beats * preset["stride"]
        if ctx.swing > 0.0 and abs(ctx.pulse_subdiv_beats - ctx.swing_unit_beats) < ctx.EPS:
            if ctx.swing_shape == "offbeat":
                interval *= (1 + ctx.swing) if idx % 2 == 0 else (1 - ctx.swing)
            elif ctx.swing_shape == "even":
                half = ctx.swing * 0.5
                if idx % 2 == 0:
                    interval *= max(0.0, 1 - half)
                else:
                    interval *= 1 + half
            else:
                mod = idx % 3
                if mod == 0:
                    interval *= 1 + ctx.swing
                elif mod == 1:
                    interval *= 1 - ctx.swing
        next_b = b + interval
        if ctx.cycle_mode == "bar" and bar_idx + 1 < len(ctx.downbeats):
            next_b = min(next_b, ctx.time_to_beat(ctx.downbeats[bar_idx + 1]))
        next_b = min(next_b, ctx.time_to_beat(span.end))
        beat_inc = next_b - b
        if ctx.stable_guard:
            ctx.stable_guard.step((span.root_pc, span.quality), beat_inc)
            if ctx.stats is not None:
                ctx.stats.setdefault("guard_hold_beats", {})[bar_idx] = ctx.stable_guard.hold
        pn = ctx.pick_phrase_note(t, c_idx)
        if ctx.stable_guard:
            pn = ctx.stable_guard.filter(pn)
        end_b = b + ctx.phrase_len_beats * preset["len"]
        boundary = ctx.beat_to_time(end_b)
        if ctx.cycle_mode == "bar" and bar_idx + 1 < len(ctx.downbeats):
            boundary = min(boundary, ctx.downbeats[bar_idx + 1])
        boundary = min(boundary, span.end)
        if pn is not None:
            if ctx.humanize_ms > 0.0:
                delta_s = ctx.rng.uniform(-ctx.humanize_ms, ctx.humanize_ms) / 1000.0
                delta_e = ctx.rng.uniform(-ctx.humanize_ms, ctx.humanize_ms) / 1000.0
            else:
                delta_s = delta_e = 0.0
            start_t = t + delta_s
            if ctx.cycle_mode == "bar":
                start_t = max(ctx.downbeats[bar_idx], span.start, start_t)
            else:
                start_t = max(span.start, start_t)
            end_t = min(boundary, boundary + delta_e)
            start_t, end_t = ctx.clip(start_t, end_t, eps=ctx.EPS)
            mg = ctx.maybe_merge_gap(
                ctx.phrase_inst,
                pn,
                start_t,
                bar_start=ctx.downbeats[bar_idx],
                chord_start=span.start,
            )
            ctx.append_phrase(
                ctx.phrase_inst,
                pn,
                start_t,
                end_t,
                base_vel,
                mg,
                ctx.release_sec,
                ctx.min_phrase_len_sec,
                ctx.stats,
            )
            if ctx.stats is not None:
                ctx.stats.setdefault("bar_triggers", {}).setdefault(bar_idx, []).append((b, start_t))
            if ctx.stats is not None and bar_idx not in ctx.stats["bar_phrase_notes"]:
                ctx.stats["bar_phrase_notes"][bar_idx] = pn
            if ctx.stats is not None:
                ctx.stats.setdefault("bar_velocities", {}).setdefault(bar_idx, []).append(base_vel)
        if beat_inc <= ctx.EPS:
            logging.warning(
                "emit_phrases_for_span: non-positive beat increment; aborting span"
            )
            break
        b += beat_inc
        iter_guard -= 1
        if iter_guard <= 0:
            logging.warning(
                "emit_phrases_for_span: max iterations reached; aborting span"
            )
            break


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
        letter, accidental, octv = m.groups()
        key = f"{letter.upper()}{accidental}"
        pc = PITCH_CLASS.get(key)
        if pc is None:
            pc = PITCH_CLASS.get(letter.upper())
        if pc is None:
            raise SystemExit(f"Unknown pitch class: {letter}{accidental}")
        val = (int(octv) + 1) * 12 + pc
        if not (0 <= val <= 127):
            raise SystemExit(f"Note out of MIDI range: {token}")
        return val


def parse_int_or(x, default: int) -> int:
    """Parse ``x`` as int, returning ``default`` for ``None`` or invalid values."""

    if x is None:
        return default
    try:
        return int(str(x).strip())
    except Exception:
        return default


def validate_midi_note(v: int) -> int:
    """Validate MIDI note range 0..127."""
    v = int(v)
    if not (0 <= v <= 127):
        raise SystemExit("cycle_phrase_notes must be MIDI 0..127, 'rest', or note names")
    return v


def parse_note_token(
    tok: Union[str, int], *, warn_unknown: bool = False, mapping: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    """Normalize note token to a MIDI int or ``None`` for rests.

    If ``mapping`` is provided, alias dictionaries such as ``phrase_note_map`` and
    ``note_aliases`` are consulted. With ``mapping=None`` only built-in aliases and
    literal MIDI/note-name forms are resolved.
    """
    if isinstance(tok, str):
        t = tok.strip()
        if t.lower() in {"rest", "silence", ""}:
            return None
        alias_map, alias_inv = _current_note_alias_maps()
        alias_val = alias_map.get(t)
        if alias_val is None:
            for midi_val, alias_name in alias_inv.items():
                if alias_name == t:
                    alias_val = midi_val
                    break
        if alias_val is not None:
            resolved_alias = _resolve_pitch_token(alias_val, None)
            if resolved_alias is not None:
                return validate_midi_note(resolved_alias)
        resolved = _resolve_pitch_token(t, mapping)
        if resolved is not None:
            return validate_midi_note(resolved)
        try:
            return validate_midi_note(parse_midi_note(t))
        except SystemExit:
            if warn_unknown:
                logging.warning("unknown note alias: %s", tok)
                return None
            raise
    if tok is None:
        return None
    try:
        resolved = _resolve_pitch_token(tok, mapping)
        if resolved is not None:
            return validate_midi_note(resolved)
        return validate_midi_note(int(tok))
    except Exception as e:  # pragma: no cover - unlikely
        if warn_unknown:
            logging.warning("unknown note alias: %s", tok)
            return None
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

# --- RESOLVED MERGE: keep both damp-arg parser and guide summarization utilities ---


def parse_damp_arg(s: str) -> Tuple[str, Dict[str, Any]]:
    """Parse --damp argument into (mode, kwargs).

    Examples:
        "none"
        "cc:cc=11,channel=0,deadband=2,min_beats=0.25,clip_lo=0,clip_hi=96"
        "follow:cc=1,smooth=4"
    """
    if not s:
        return "none", {}
    if ":" in s:
        mode, rest = s.split(":", 1)
    else:
        mode, rest = s, ""
    mode = mode.strip()
    kw: Dict[str, Any] = {}
    for token in filter(None, (t.strip() for t in rest.split(","))):
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k in {"cc", "channel", "value", "smooth"}:
            try:
                kw[k] = int(v)
            except ValueError:
                raise SystemExit(f"--damp invalid int for {k}: {v}")
        elif k in {"deadband", "min_beats"}:
            try:
                kw[k] = float(v)
            except ValueError:
                raise SystemExit(f"--damp invalid float for {k}: {v}")
        elif k in {"clip_lo", "clip_hi"}:
            try:
                kw[k] = int(v)
            except ValueError:
                raise SystemExit(f"--damp invalid int for {k}: {v}")
        else:
            # best-effort: numeric if possible, else string
            try:
                kw[k] = float(v)
            except ValueError:
                kw[k] = v
    lo = kw.pop("clip_lo", None)
    hi = kw.pop("clip_hi", None)
    if lo is not None or hi is not None:
        kw["clip"] = (int(lo) if lo is not None else 0, int(hi) if hi is not None else 127)
    return mode, kw


# Resolved merge: combine guide summarization utilities with phrase scheduling
# NOTE: Assumes the following are defined elsewhere in the module:
#   - parse_note_token, validate_midi_note, EPS, PHRASE_INST_NAME
#   - pretty_midi, json, math, bisect, collections, random
#   - from typing import Any, Optional, Dict, List, Tuple, Union, Set


def parse_thresholds_arg(s: str) -> Dict[str, Union[int, List[Tuple[int, float]]]]:
    cfg = parse_json_arg("--guide-thresholds", s, GUIDE_THRESHOLDS_SCHEMA)
    for k in ("low", "mid", "high"):
        v = cfg[k]
        if isinstance(v, list):
            items: List[Tuple[int, float]] = []
            for it in v:
                if isinstance(it, list) and len(it) == 2:
                    note = parse_note_token(it[0])
                    weight = float(it[1])
                else:
                    note = parse_note_token(it)
                    weight = 1.0
                if note is None:
                    raise SystemExit("--guide-thresholds cannot use 'rest'")
                items.append((int(note), weight))
            cfg[k] = items
        elif isinstance(v, str):
            p = parse_note_token(v)
            if p is None:
                raise SystemExit("--guide-thresholds cannot use 'rest'")
            cfg[k] = int(p)
        elif isinstance(v, int):
            validate_midi_note(v)
            cfg[k] = int(v)
        else:
            raise SystemExit(
                f"--guide-thresholds {k} must be int, MIDI note token, or list of tokens"
            )
    return cfg


GUIDE_THRESHOLDS_SCHEMA: Dict[str, Any] = {
    "__type__": dict,
    "__description__": '{"low": int|list, "mid": int|list, "high": int|list}',
    "low": {"type": (int, str, list)},
    "mid": {"type": (int, str, list)},
    "high": {"type": (int, str, list)},
}


STYLE_INJECT_SCHEMA: Dict[str, Any] = {
    "__type__": dict,
    "__description__": '{"period": int, "note": int, "duration_beats": >0}',
    "period": {"type": int, "min": 1},
    "note": {"type": int, "min": 0, "max": 127},
    "duration_beats": {"type": (int, float), "min_exclusive": 0.0},
    "min_gap_beats": {"type": (int, float), "min": 0.0, "optional": True},
    "avoid_pitches": {"type": list, "optional": True},
}


SECTIONS_SCHEMA: Dict[str, Any] = {
    "__type__": list,
    "__description__": '[{"start_bar": int, "tag": str, ...}] or label list',
    "__item__": {
        "__type__": (dict, str),
        "__schema__": {
            "start_bar": {"type": int, "min": 0, "optional": True},
            "end_bar": {"type": int, "min": 0, "optional": True},
            "tag": {"type": str, "optional": True},
            "density": {"type": str, "optional": True},
            "phrase_pool": {"type": list, "optional": True},
            "pool": {"type": list, "optional": True},
            "pool_by_quality": {"type": dict, "optional": True},
        },
    },
}


def parse_json_arg(name: str, raw: str, schema: Dict[str, Any]) -> Any:
    label = name if name.startswith("--") else f"--{name}"
    errors: List[str] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        errors.append(f"{exc.msg} at line {exc.lineno} column {exc.colno}")
        normalized = raw.replace("'", '"')
        if normalized != raw:
            try:
                data = json.loads(normalized)
            except json.JSONDecodeError as exc_norm:
                errors.append(
                    f"after quote normalization: {exc_norm.msg} at line {exc_norm.lineno} column {exc_norm.colno}"
                )
                raise SystemExit(f"{label} expects JSON ({'; '.join(errors)})") from exc_norm
        else:
            raise SystemExit(f"{label} expects JSON ({errors[0]})") from exc
    except Exception as exc:
        raise SystemExit(f"{label} expects JSON ({exc})") from exc

    def _type_name(tp: Union[type, Tuple[type, ...]]) -> str:
        if isinstance(tp, tuple):
            return "|".join(sorted(t.__name__ for t in tp))
        return tp.__name__

    def _ensure_type(value: Any, expected: Union[type, Tuple[type, ...]], path: str) -> None:
        types = expected if isinstance(expected, tuple) else (expected,)
        if not isinstance(value, types):
            raise SystemExit(
                f"{label} field '{path}' must be {_type_name(types)}; got {type(value).__name__}"
            )

    def _validate(obj: Any, spec: Dict[str, Any], path: str) -> None:
        description = spec.get("__description__")
        expected_type = spec.get("__type__")
        if expected_type is not None:
            _ensure_type(obj, expected_type, path or "root")
        nested_schema = spec.get("__schema__")
        if nested_schema and isinstance(obj, (dict, list)):
            _validate(obj, nested_schema, path)
        if isinstance(obj, list) and "__item__" in spec:
            item_spec = spec["__item__"]
            for idx, item in enumerate(obj):
                _validate(item, item_spec, f"{path}[{idx}]" if path else f"[{idx}]")
        if not isinstance(obj, dict):
            return
        for key, rule in spec.items():
            if key.startswith("__"):
                continue
            field_path = f"{path}.{key}" if path else key
            if not isinstance(rule, dict):
                expected = rule
                rule = {"type": expected}
            optional = bool(rule.get("optional", False))
            if key not in obj:
                if optional:
                    continue
                desc = description or "see documentation"
                raise SystemExit(
                    f"{label} missing required key '{field_path}' (expects {desc})"
                )
            value = obj[key]
            if "type" in rule and rule["type"] is not None:
                _ensure_type(value, rule["type"], field_path)
            if "choices" in rule and value not in rule["choices"]:
                raise SystemExit(
                    f"{label} field '{field_path}' must be one of {sorted(rule['choices'])}"
                )
            if "min" in rule and value < rule["min"]:
                raise SystemExit(
                    f"{label} field '{field_path}' must be >= {rule['min']}"
                )
            if "min_exclusive" in rule and value <= rule["min_exclusive"]:
                raise SystemExit(
                    f"{label} field '{field_path}' must be > {rule['min_exclusive']}"
                )
            if "max" in rule and value > rule["max"]:
                raise SystemExit(
                    f"{label} field '{field_path}' must be <= {rule['max']}"
                )
            if "max_exclusive" in rule and value >= rule["max_exclusive"]:
                raise SystemExit(
                    f"{label} field '{field_path}' must be < {rule['max_exclusive']}"
                )
            if "schema" in rule and isinstance(value, (dict, list)):
                _validate(value, rule["schema"], field_path)

    _validate(data, schema, "")
    return data


@dataclass
class InlineChordEvent:
    chord: str
    start_beats: Optional[float] = None
    start_time: Optional[float] = None
    bar: Optional[int] = None


def parse_inline_chords(raw: str) -> Optional[List[InlineChordEvent]]:
    if raw is None:
        return None
    normalized = unicodedata.normalize("NFKC", raw)
    normalized = (
        normalized.replace("♯", "#")
        .replace("＃", "#")
        .replace("♭", "b")
        .replace("ｂ", "b")
    )
    trimmed = normalized.strip()
    if not trimmed:
        return []

    if trimmed[0] in "[{":
        data: Any = None
        try:
            data = json.loads(trimmed)
        except Exception:
            if yaml is not None:
                try:
                    data = yaml.safe_load(trimmed)
                except Exception:
                    data = None
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            events: List[InlineChordEvent] = []
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    raise SystemExit(
                        f"--chords inline JSON element at index {idx} must be an object"
                    )
                chord_token = item.get("chord")
                if not isinstance(chord_token, str) or not chord_token.strip():
                    raise SystemExit(
                        f"--chords inline JSON element at index {idx} must provide 'chord' string"
                    )
                start_beats: Optional[float] = None
                start_time: Optional[float] = None
                bar_idx: Optional[int] = None
                if "start_beats" in item:
                    try:
                        start_beats = float(item["start_beats"])
                    except Exception as exc:
                        raise SystemExit(
                            f"--chords inline JSON element at index {idx} has invalid start_beats"
                        ) from exc
                elif "start" in item:
                    try:
                        start_time = float(item["start"])
                    except Exception as exc:
                        raise SystemExit(
                            f"--chords inline JSON element at index {idx} has invalid start"
                        ) from exc
                elif "bar" in item or "start_bar" in item:
                    key = "bar" if "bar" in item else "start_bar"
                    try:
                        bar_idx = int(item[key])
                    except Exception as exc:
                        raise SystemExit(
                            f"--chords inline JSON element at index {idx} has invalid bar index"
                        ) from exc
                    if bar_idx < 0:
                        raise SystemExit(
                            f"--chords inline JSON element at index {idx} must have non-negative bar"
                        )
                else:
                    raise SystemExit(
                        f"--chords inline JSON element at index {idx} requires 'bar', 'start', or 'start_beats'"
                    )
                events.append(
                    InlineChordEvent(
                        chord=chord_token.strip(),
                        start_beats=start_beats,
                        start_time=start_time,
                        bar=bar_idx,
                    )
                )
            return events

    ascii_spec = normalized.replace("，", ",").replace("：", ":")

    def _looks_like_inline_token(token: str) -> bool:
        if ":" not in token:
            return False
        head = token.split(":", 1)[0].strip()
        try:
            float(head)
        except ValueError:
            return False
        return True

    tokens = [tok.strip() for tok in ascii_spec.split(",") if tok.strip()]
    if not tokens:
        return []
    if not any(_looks_like_inline_token(tok) for tok in tokens):
        return None

    events: List[InlineChordEvent] = []
    for idx, raw_token in enumerate(tokens):
        parts = raw_token.split(":", 2)
        if len(parts) == 3:
            start_txt, root_txt, qual_txt = parts
            chord_symbol = f"{root_txt.strip()}:{qual_txt.strip()}"
        elif len(parts) == 2:
            start_txt, chord_txt = parts
            chord_txt = chord_txt.strip()
            if not chord_txt:
                raise SystemExit(
                    f"--chords inline token {idx} ('{raw_token}') missing chord descriptor"
                )
            if ":" in chord_txt:
                chord_symbol = chord_txt
            else:
                body = chord_txt
                quality_guess = "maj"
                lowered = body.lower()
                if lowered.endswith("minor"):
                    body = body[:-5]
                    quality_guess = "min"
                elif lowered.endswith("major"):
                    body = body[:-5]
                    quality_guess = "maj"
                elif lowered.endswith("min"):
                    body = body[:-3]
                    quality_guess = "min"
                elif lowered.endswith("maj"):
                    body = body[:-3]
                    quality_guess = "maj"
                elif lowered.endswith("m"):
                    body = body[:-1]
                    quality_guess = "min"
                body = body.strip()
                if not body:
                    raise SystemExit(
                        f"--chords inline token {idx} ('{raw_token}') cannot infer root from '{chord_txt}'"
                    )
                chord_symbol = f"{body}:{quality_guess}"
        else:
            raise SystemExit(
                f"--chords inline token {idx} ('{raw_token}') must be start:Root[:Quality]"
            )
        try:
            start_beats_val = float(start_txt)
        except ValueError as exc:
            raise SystemExit(
                f"--chords inline token {idx} ('{raw_token}') has invalid beat value"
            ) from exc
        events.append(InlineChordEvent(chord=chord_symbol.strip(), start_beats=start_beats_val))
    return events


def parse_onset_th_arg(s: str) -> Dict[str, int]:
    try:
        cfg = json.loads(s)
    except Exception:
        raise SystemExit("--guide-onset-th must be JSON")
    if not isinstance(cfg, dict):
        raise SystemExit("--guide-onset-th must be JSON object")
    for k in ("mid", "high"):
        v = cfg.get(k)
        if not isinstance(v, int):
            raise SystemExit(f"--guide-onset-th {k} must be int")
    return cfg


def parse_phrase_pool_arg(s: str) -> Dict[str, Any]:
    try:
        data = json.loads(s)
    except Exception:
        raise SystemExit("--phrase-pool must be JSON")
    items: List[Tuple[int, float]] = []
    T = None
    if isinstance(data, dict) and "notes" in data:
        notes = [parse_note_token(n) for n in data["notes"]]
        weights = data.get("weights")
        if weights is None:
            weights = [1.0] * len(notes)
        if len(weights) != len(notes):
            raise SystemExit("--phrase-pool weights length mismatch")
        for n, w in zip(notes, weights):
            if n is not None:
                items.append((int(n), float(w)))
        T = data.get("T")
    else:
        if isinstance(data, dict):
            # take first level if mapping provided
            if data:
                data = next(iter(data.values()))
            else:
                data = []
        if not isinstance(data, list):
            raise SystemExit("--phrase-pool must be list or mapping of lists")
        for it in data:
            if isinstance(it, list) and len(it) == 2:
                note = parse_note_token(it[0])
                weight = float(it[1])
            else:
                note = parse_note_token(it)
                weight = 1.0
            if note is not None:
                items.append((int(note), weight))
    return {"pool": items, "T": T}


class PoolPicker:
    """Utility to pick notes from a pool using various policies."""

    def __init__(
        self,
        pool: List[Tuple[int, float]],
        mode: str = "random",
        T: Optional[List[List[float]]] = None,
        no_repeat_window: int = 1,
        rng: Optional[random.Random] = None,
    ):
        self.pool = pool
        self.mode = mode
        self.T = T
        self.no_repeat_window = max(1, no_repeat_window)
        self.idx = 0
        self.last_idx: Optional[int] = None
        self.recent: collections.deque = collections.deque(maxlen=self.no_repeat_window)
        if rng is None:
            # Honor opt-in deterministic mode while preserving override priority.
            rng = random.Random(0) if _SPARKLE_DETERMINISTIC else random
        self.rng = rng

    def _choose(self, weights: Optional[List[float]] = None) -> int:
        notes = [n for n, _ in self.pool]
        if weights is None:
            weights = [w for _, w in self.pool]
        return self.rng.choices(list(range(len(notes))), weights=weights, k=1)[0]

    def pick(self) -> int:
        if not self.pool:
            raise RuntimeError("empty pool")
        idx: int
        if self.mode == "roundrobin":
            idx = self.idx % len(self.pool)
            self.idx += 1
        elif self.mode == "weighted":
            idx = self._choose()
        elif self.mode == "markov" and self.T:
            if self.last_idx is None:
                idx = 0
            else:
                row = self.T[self.last_idx]
                idx = self._choose(row)
        else:  # random or markov without T
            idx = self._choose()
        note = self.pool[idx][0]
        if note in self.recent and len(self.pool) > len(self.recent):
            candidates = [i for i, (n, _) in enumerate(self.pool) if n not in self.recent]
            if candidates:
                idx = self.rng.choice(candidates)
                note = self.pool[idx][0]
        self.recent.append(note)
        self.last_idx = idx
        return note


def thin_cc_events(
    events: List[Tuple[float, int]],
    *,
    min_interval_beats: float = 0.0,
    deadband: int = 0,
    clip: Optional[Tuple[int, int]] = None,
) -> List[Tuple[float, int]]:
    if not events:
        return events
    out: List[Tuple[float, int]] = []
    last_b = None
    last_v = None
    lo = clip[0] if clip else 0
    hi = clip[1] if clip else 127
    for b, v in events:
        v = max(lo, min(hi, v))
        if last_b is not None:
            if min_interval_beats > 0.0 and (b - last_b) < min_interval_beats - EPS:
                continue
            if deadband > 0 and last_v is not None and abs(v - last_v) <= deadband:
                continue
        out.append((b, v))
        last_b = b
        last_v = v
    return out


def summarize_guide_midi(
    pm: "pretty_midi.PrettyMIDI",
    quant: str,
    thresholds: Dict[str, Union[int, List[Tuple[int, float]]]],
    *,
    rest_silence_th: Optional[float] = None,
    onset_th: Optional[Dict[str, int]] = None,
    note_tokens_allowed: bool = True,
    curve: str = "linear",
    gamma: float = 1.6,
    smooth_sigma: float = 0.0,
    pick_mode: str = "roundrobin",
) -> Tuple[
    Dict[int, int],
    List[Tuple[float, int]],
    List[Tuple[float, float]],
    List[float],
    List[int],
    List[str],
]:
    """Summarize guide MIDI into phrase note map and damping CC values.

    cc_events return pairs of (beat, value). Sections return labels per unit."""
    notes = []
    for inst in pm.instruments:
        if not getattr(inst, "is_drum", False):
            notes.extend(inst.notes)
    notes.sort(key=lambda n: n.start)
    try:
        beats = list(pm.get_beats())
    except Exception:
        beats = []
    if len(beats) == 0:
        end = max(pm.get_end_time(), 1.0)
        n = max(1, int(math.ceil(end)))
        beats = [float(i) for i in range(n + 1)]

    def time_to_beat(t: float) -> float:
        idx = bisect.bisect_right(beats, t) - 1
        if idx < 0:
            return 0.0
        if idx >= len(beats) - 1:
            last = beats[-1] - beats[-2]
            return (len(beats) - 1) + (t - beats[-1]) / last
        span = beats[idx + 1] - beats[idx]
        return idx + (t - beats[idx]) / span

    if quant == "bar":
        try:
            downs = list(pm.get_downbeats())
        except Exception:
            downs = []
        if len(downs) == 0:
            downs = beats[::4]
            if len(downs) == 0:
                downs = beats
    else:
        downs = beats
    units: List[Tuple[float, float]] = []
    for i, s in enumerate(downs):
        e = downs[i + 1] if i + 1 < len(downs) else pm.get_end_time()
        units.append((s, e))
    onset_list: List[int] = []
    rest_list: List[float] = []
    for s, e in units:
        onset = 0
        cov: List[Tuple[float, float]] = []
        for n in notes:
            if n.end <= s or n.start >= e:
                continue
            if s <= n.start < e:
                onset += 1
            cov.append((max(s, n.start), min(e, n.end)))
        cov.sort()
        covered = 0.0
        last = s
        for a, b in cov:
            if b <= last:
                continue
            a = max(a, last)
            covered += b - a
            last = b
        span = e - s if e > s else 1.0
        rest_ratio = 1.0 - covered / span
        onset_list.append(onset)
        rest_list.append(rest_ratio)
    rr = rest_list[:]
    if smooth_sigma > 0.0 and len(rr) > 1:
        radius = max(1, int(smooth_sigma * 3))
        weights = [math.exp(-0.5 * (i / smooth_sigma) ** 2) for i in range(-radius, radius + 1)]
        total = sum(weights)
        weights = [w / total for w in weights]
        smoothed: List[float] = []
        for i in range(len(rr)):
            v = 0.0
            norm = 0.0
            for k, w in enumerate(weights):
                j = i + k - radius
                if 0 <= j < len(rr):
                    v += rr[j] * w
                    norm += w
            if norm > 0:
                v /= norm
            smoothed.append(v)
        rr = smoothed
    cc_events: List[Tuple[float, int]] = []
    for idx, r in enumerate(rr):
        x = max(0.0, min(1.0, r))
        if curve == "exp":
            x = x**gamma
        elif curve == "inv":
            x = 1.0 - x
        val = int(round(x * 127))
        cc_events.append((time_to_beat(units[idx][0]), val))
    t_mid = onset_th.get("mid", 1) if onset_th else 1
    t_high = onset_th.get("high", 3) if onset_th else 3
    low = thresholds.get("low")
    mid = thresholds.get("mid")
    high = thresholds.get("high")
    if note_tokens_allowed:
        if not isinstance(low, list):
            low = parse_note_token(low) if low is not None else None
        if not isinstance(mid, list):
            mid = parse_note_token(mid) if mid is not None else None
        if not isinstance(high, list):
            high = parse_note_token(high) if high is not None else None

    def _norm_pool(v: List) -> List[Tuple[int, float]]:
        items: List[Tuple[int, float]] = []
        for it in v:
            if isinstance(it, (list, tuple)) and len(it) == 2:
                note = parse_note_token(it[0])
                weight = float(it[1])
            else:
                note = parse_note_token(it)
                weight = 1.0
            if note is not None:
                items.append((int(note), weight))
        return items

    pickers: Dict[str, Optional[PoolPicker]] = {
        "low": (
            PoolPicker(_norm_pool(thresholds["low"]), pick_mode)
            if isinstance(thresholds["low"], list)
            else None
        ),
        "mid": (
            PoolPicker(_norm_pool(thresholds["mid"]), pick_mode)
            if isinstance(thresholds["mid"], list)
            else None
        ),
        "high": (
            PoolPicker(_norm_pool(thresholds["high"]), pick_mode)
            if isinstance(thresholds["high"], list)
            else None
        ),
    }
    note_map: Dict[int, int] = {}
    sections = ["verse"] * len(onset_list)
    for idx, onset in enumerate(onset_list):
        if rest_silence_th is not None and rest_list[idx] >= rest_silence_th:
            continue
        if onset >= t_high:
            pool = high
            picker = pickers["high"]
            sec = "chorus"
        elif onset >= t_mid:
            pool = mid
            picker = pickers["mid"]
            sec = "verse"
        else:
            pool = low
            picker = pickers["low"]
            sec = "verse"
        if isinstance(pool, list):
            note = picker.pick() if picker else None
        else:
            note = pool
        if note is not None:
            note_map[idx] = int(note)
            sections[idx] = sec
    # break detection: long rest spans
    i = 0
    while i < len(rest_list):
        if rest_list[i] >= 0.8:
            j = i
            while j < len(rest_list) and rest_list[j] >= 0.8:
                j += 1
            if j - i >= 2:
                for k in range(i, j):
                    sections[k] = "break"
            i = j
        else:
            i += 1
    # simple local maxima for chorus
    dens = onset_list
    for i in range(1, len(dens) - 1):
        if dens[i] > dens[i - 1] and dens[i] > dens[i + 1]:
            sections[i] = "chorus"
    return note_map, cc_events, units, rest_list, onset_list, sections


def normalize_sections(
    raw: Optional[Sequence[Any]],
    *,
    bar_count: Optional[int],
    default_tag: str = "section",
    stats: Optional[Dict] = None,
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Normalize section specs into sorted ranges and per-bar labels."""

    if isinstance(raw, dict):  # pragma: no cover - defensive
        tokens: Sequence[Any] = [raw]
    elif raw is None:
        tokens = []
    else:
        tokens = list(raw)

    unit_count = None
    if bar_count is not None:
        try:
            unit_count = max(0, int(bar_count))
        except (TypeError, ValueError):
            unit_count = None

    has_dict = any(isinstance(tok, dict) for tok in tokens)
    labels_mode = bool(tokens) and not has_dict

    warnings: List[str] = []
    details: List[str] = []

    if labels_mode:
        label_values: List[str] = []
        for tok in tokens:
            if tok is None:
                label = default_tag
            else:
                label = str(tok).strip()
                if not label:
                    label = default_tag
            label_values.append(label)
        if unit_count is not None:
            if len(label_values) < unit_count:
                fill_label = label_values[-1] if label_values else default_tag
                label_values.extend([fill_label] * (unit_count - len(label_values)))
                details.append("extended labels to bar span")
            elif len(label_values) > unit_count:
                label_values = label_values[:unit_count]
                details.append("clamped to bar span")
        labels = label_values
        if not labels and unit_count:
            labels = [default_tag] * unit_count
        normalized: List[Dict[str, Any]] = []
        if labels:
            start = 0
            cur = labels[0]
            for idx in range(1, len(labels) + 1):
                if idx == len(labels) or labels[idx] != cur:
                    normalized.append(
                        {
                            "start_bar": start,
                            "end_bar": idx,
                            "tag": cur,
                            "explicit_end": True,
                        }
                    )
                    if idx < len(labels):
                        start = idx
                        cur = labels[idx]
        if unit_count is not None and len(labels) < unit_count:
            labels.extend([default_tag] * (unit_count - len(labels)))
        final_labels = labels
    else:
        prepared: List[Dict[str, Any]] = []
        did_fill = False
        did_clamp = False
        did_overlap = False
        did_clamp_local = False

        def _coerce_bar(value: Any, fallback: int) -> int:
            nonlocal did_clamp_local
            try:
                bar = int(value)
            except (TypeError, ValueError):
                bar = fallback
                warnings.append("coerced non-integer start/end")
            if bar < 0:
                bar = 0
                did_clamp_local = True
            return bar

        for idx, tok in enumerate(tokens):
            if isinstance(tok, dict):
                start_bar = _coerce_bar(tok.get("start_bar", idx), idx)
                tag_val = tok.get("tag") or tok.get("name") or tok.get("label") or default_tag
                tag = str(tag_val).strip() or default_tag
                end = tok.get("end_bar")
                explicit = end is not None
                end_val = _coerce_bar(end, start_bar + 1) if end is not None else None
            else:
                start_bar = _coerce_bar(idx, idx)
                tag = str(tok).strip() or default_tag
                explicit = True
                end_val = None

            if unit_count is not None and start_bar >= unit_count:
                warnings.append("dropped section starting beyond unit span")
                continue

            prepared.append(
                {
                    "start_bar": start_bar,
                    "end_bar": end_val,
                    "tag": tag,
                    "explicit_end": explicit,
                    "_order": idx,
                }
            )

        if not prepared:
            final_labels = [default_tag] * unit_count if unit_count else []
            normalized = []
        else:
            prepared.sort(key=lambda item: (item["start_bar"], item["_order"]))
            normalized = []
            prev_end = 0
            max_end = 0
            for pos, item in enumerate(prepared):
                start = int(item["start_bar"])
                if start < 0:
                    start = 0
                    did_clamp = True
                if normalized and start < prev_end:
                    start = prev_end
                    did_overlap = True

                if pos + 1 < len(prepared):
                    next_hint: Optional[int] = int(prepared[pos + 1]["start_bar"])
                else:
                    next_hint = unit_count

                end_val = item["end_bar"]
                explicit = bool(item["explicit_end"])
                if end_val is None:
                    candidate = next_hint if next_hint is not None else start + 1
                    end_val = max(start + 1, candidate)
                    if not explicit:
                        did_fill = True
                else:
                    end_val = int(end_val)

                if unit_count is not None and end_val > unit_count:
                    end_val = unit_count
                    did_clamp = True

                if end_val <= start:
                    did_overlap = True
                    prev_end = start
                    continue

                tag = str(item["tag"]).strip() or default_tag
                normalized.append(
                    {
                        "start_bar": start,
                        "end_bar": end_val,
                        "tag": tag,
                        "explicit_end": explicit,
                    }
                )
                prev_end = end_val
                max_end = max(max_end, end_val)

            span = unit_count if unit_count is not None else max_end
            final_labels = [default_tag] * max(0, span)
            if span and normalized:
                for sec in normalized:
                    start = max(0, min(span, int(sec.get("start_bar", 0))))
                    end = max(start, min(span, int(sec.get("end_bar", start + 1))))
                    tag = str(sec.get("tag", default_tag)).strip() or default_tag
                    for idx in range(start, end):
                        final_labels[idx] = tag

            if did_fill:
                details.append("filled missing end bars")
            if did_overlap:
                details.append("resolved overlaps/empties")
            if did_clamp or did_clamp_local:
                details.append("clamped to unit span")

    if warnings:
        details.extend(sorted(set(warnings)))
    if details:
        logging.warning("normalize_sections adjustments: %s", ", ".join(details))
        if stats is not None:
            stats.setdefault("warnings", []).append(
                f"normalize_sections adjustments: {', '.join(details)}"
            )

    if stats is not None:
        stats["sections_norm"] = [dict(sec) for sec in normalized]
        if verbose and normalized:
            logging.info("normalize_sections normalized=%s", normalized)

    return normalized, final_labels


def insert_style_fill(
    pm_out: "pretty_midi.PrettyMIDI",
    mode: str,
    units: List[Tuple[float, float]],
    mapping: Dict,
    *,
    sections: Optional[List[Any]] = None,
    rest_ratio_list: Optional[List[float]] = None,
    rest_th: float = 0.75,
    fill_length_beats: float = 0.25,
    bpm: float = 120.0,
    min_gap_beats: float = 0.0,
    avoid_pitches: Optional[Set[int]] = None,
    filled_bars: Optional[List[int]] = None,
    bar_count: Optional[int] = None,
    section_default: str = "section",
) -> int:
    """Insert style fills based on mode."""
    def _get_phrase_inst(pm_obj: pretty_midi.PrettyMIDI) -> pretty_midi.Instrument:
        """Return the phrase instrument, creating it so fills can be scheduled."""
        for inst in pm_obj.instruments:
            if getattr(inst, "name", "") == PHRASE_INST_NAME:
                return inst
        new_inst = pretty_midi.Instrument(program=0, name=PHRASE_INST_NAME)
        pm_obj.instruments.append(new_inst)
        return new_inst

    phrase_inst = None
    for inst in pm_out.instruments:
        if inst.name == PHRASE_INST_NAME:
            phrase_inst = inst
            break
    if phrase_inst is None:
        phrase_inst = _get_phrase_inst(pm_out)
    if not units:
        return 0
    style_fill_raw = mapping.get("style_fill")

    explicit_avoid: Optional[Iterable[Any]] = None
    if avoid_pitches is not None:
        explicit_avoid = list(avoid_pitches)
    avoid_set = build_avoid_set(mapping, explicit_avoid)

    def _resolve_pitch(candidate: Any) -> Optional[int]:
        if candidate is None:
            return None
        resolved = resolve_phrase_alias(candidate, mapping)
        if resolved is None:
            try:
                resolved = int(round(float(candidate)))
            except Exception:
                return None
        try:
            value = int(round(float(resolved)))
        except Exception:
            return None
        return max(0, min(127, value))

    def _coerce_int(value: Any, default: int) -> int:
        if value is None:
            return default
        try:
            coerced = int(round(float(value)))
        except Exception:
            return default
        return max(0, min(127, coerced))

    base_pitch = _resolve_pitch(mapping.get("phrase_note", 36))
    pitch_min = _coerce_int(mapping.get("phrase_pitch_min"), 0)
    pitch_max = _coerce_int(mapping.get("phrase_pitch_max"), 127)
    if base_pitch is None:
        base_pitch = max(pitch_min, min(pitch_max, 36))
    else:
        base_pitch = max(pitch_min, min(pitch_max, base_pitch))

    def _fallback_candidates(base: int, lo: int, hi: int) -> List[int]:
        up = list(range(base, min(base + 13, hi + 1)))
        down = list(range(base - 1, max(base - 13, lo - 1), -1))
        return up + down

    pitch_value: Optional[int] = None
    pitch_source = "style"
    candidate_sources: List[Tuple[Any, str]] = [
        (style_fill_raw, "style"),
        (mapping.get("style_fill"), "style"),
        (34, "default"),
    ]
    candidate_sources.extend(
        (candidate, "fallback") for candidate in _fallback_candidates(base_pitch, pitch_min, pitch_max)
    )

    for candidate, source in candidate_sources:
        resolved = _resolve_pitch(candidate)
        if resolved is None:
            continue
        if source != "style" and resolved in avoid_set:
            continue
        pitch_value = resolved
        pitch_source = source
        break

    if pitch_value is None:
        logging.warning(
            "insert_style_fill: unable to resolve style note %r; skipping", style_fill_raw
        )
        return 0

    pitch = pitch_value
    pitch = max(0, min(127, pitch))
    if pitch_source != "style" and pitch in avoid_set:
        replacement: Optional[int] = None
        for candidate in range(pitch, 128):
            if candidate not in avoid_set:
                replacement = candidate
                break
        if replacement is None:
            logging.warning(
                "insert_style_fill: all pitches avoided for %r; skipping", style_fill_raw
            )
            return 0
        pitch = replacement
    seed_bpm = float(bpm) if bpm is not None else 120.0
    if not math.isfinite(seed_bpm) or seed_bpm <= 0.0:
        seed_bpm = 120.0
    _sanitize_tempi(pm_out)
    _ensure_tempo_and_ticks(pm_out, seed_bpm, pm_out.time_signature_changes)
    stats_ref = getattr(pm_out, "_sparkle_stats", None)
    beat_times: List[float]
    cached_beats: Optional[List[float]] = None
    if stats_ref:
        raw = stats_ref.get("beat_times")
        if raw:
            cached_beats = [float(bt) for bt in raw]
    if cached_beats:
        beat_times = cached_beats
    else:
        try:
            beat_times = pm_out.get_beats()
        except (AttributeError, IndexError, ValueError):
            step = 60.0 / seed_bpm
            end = units[-1][1] if units else step
            n = int(math.ceil(end / step)) + 1
            beat_times = [i * step for i in range(n)]
    if len(beat_times) < 2:
        step = 60.0 / seed_bpm
        beat_times = [0.0, step]

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

    if avoid_pitches is None and avoid_set and pitch not in avoid_set:
        avoid_overlap = {pitch}
    else:
        avoid_overlap = set(avoid_set)
        avoid_overlap.add(pitch)

    count = 0
    used: Set[int] = set()
    filled_tracker: Set[int] = set(filled_bars or [])
    inserted_spans: List[Tuple[float, float, float]] = []

    max_bar_idx = len(units) - 1
    unit_starts = [start for start, _ in units]
    bar_note_index: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    for note_obj in phrase_inst.notes:
        try:
            note_start = float(getattr(note_obj, "start", 0.0))
            note_end = float(getattr(note_obj, "end", note_start))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(note_start):
            continue
        if not math.isfinite(note_end):
            note_end = note_start
        note_pitch = _resolve_pitch_token(note_obj, mapping)
        if note_pitch is None:
            continue
        note_start_b = time_to_beat(note_start)
        note_end_b = time_to_beat(note_end)
        data = (note_pitch, note_start, note_end, note_start_b, note_end_b)
        if max_bar_idx < 0:
            continue
        start_idx = bisect.bisect_right(unit_starts, note_start) - 1
        if start_idx < 0:
            start_idx = 0
        for bar in range(start_idx, max_bar_idx + 1):
            unit_start, unit_end = units[bar]
            if note_end <= unit_start - EPS:
                break
            if note_start < unit_end + EPS and note_end > unit_start - EPS:
                bar_note_index.setdefault(bar, []).append(data)
            if note_end <= unit_end + EPS:
                break

    def iter_notes_for_bar(bar_idx: int) -> Iterable[Tuple[int, float, float, float, float]]:
        if not bar_note_index:
            return []  # type: ignore[return-value]
        seen: Set[Tuple[int, float, float]] = set()
        collected: List[Tuple[int, float, float, float, float]] = []
        for neighbor in (bar_idx - 1, bar_idx, bar_idx + 1):
            if neighbor < 0:
                continue
            if max_bar_idx >= 0 and neighbor > max_bar_idx:
                continue
            for entry in bar_note_index.get(neighbor, []):
                key = (entry[0], entry[1], entry[2])
                if key in seen:
                    continue
                seen.add(key)
                collected.append(entry)
        return collected
    section_unit_count = bar_count if bar_count is not None else len(units)
    section_layout: List[Dict[str, Any]] = []
    if sections:
        raw_sections = list(sections) if not isinstance(sections, list) else sections
        stats_target = stats_ref if isinstance(stats_ref, dict) else None
        layout, _labels = normalize_sections(
            raw_sections,
            bar_count=section_unit_count,
            default_tag=section_default,
            stats=stats_target,
        )
        if any(not isinstance(sec, dict) for sec in raw_sections):
            logging.info("sections(label) was provided; normalized to ranges")
        sections = layout
        section_layout = layout

    def _norm_sections(sec: Iterable[Any], bars: int) -> List[str]:
        items = list(sec)
        if not items:
            return []
        if isinstance(items[0], str):
            return [str(label) for label in items[:bars]]
        labels: List[Optional[str]] = [None] * bars
        for item in items:
            if isinstance(item, dict):
                sb = int(item.get("start_bar", 0) or 0)
                eb = int(item.get("end_bar", sb + 1) or (sb + 1))
                tag_val = item.get("tag") or item.get("label") or item.get("section")
            else:
                sb = int(getattr(item, "start_bar", 0) or 0)
                eb = int(getattr(item, "end_bar", sb + 1) or (sb + 1))
                tag_val = (
                    getattr(item, "tag", None)
                    or getattr(item, "label", None)
                    or getattr(item, "section", None)
                )
            tag = str(tag_val) if tag_val else "A"
            for bar in range(max(0, sb), min(bars, eb)):
                labels[bar] = tag
        current = labels[0] or "A"
        for idx in range(bars):
            if labels[idx] is None:
                labels[idx] = current
            else:
                current = labels[idx]
        return [label or "A" for label in labels]

    section_tags: Set[str] = set(
        str(sec.get("tag", section_default))
        for sec in section_layout
        if isinstance(sec, dict)
    )

    if (
        mode == "section_end"
        and section_layout
        and len(section_layout) > 1
        and len(section_tags) > 1
        and units
        and bpm is not None
    ):
        bars_total = int(bar_count) if bar_count is not None else len(units)
        labels = _norm_sections(section_layout, bars_total)
        if labels:
            end_bars: List[int] = []
            max_bar = min(bars_total, len(units))
            span = min(len(labels), max_bar)
            for idx in range(max_bar):
                if idx >= span:
                    if idx == bars_total - 1:
                        end_bars.append(idx)
                    continue
                if idx == bars_total - 1 or idx == span - 1:
                    end_bars.append(idx)
                elif labels[idx] != labels[idx + 1]:
                    end_bars.append(idx)
            candidate_pitches = [mapping.get("style_fill"), 34, base_pitch]
            fill_pitch: Optional[int] = None
            for candidate in candidate_pitches:
                resolved = _resolve_pitch(candidate)
                if resolved is None:
                    continue
                value = int(round(resolved))
                value = max(pitch_min, min(pitch_max, value))
                if value in avoid_set:
                    continue
                fill_pitch = value
                break
            if fill_pitch is None:
                fallback_seq = _fallback_candidates(base_pitch, pitch_min, pitch_max)
                fill_pitch = next((cand for cand in fallback_seq if cand not in avoid_set), None)
            if fill_pitch is None:
                logging.warning(
                    "insert_style_fill: unable to find non-avoided pitch for %r",
                    mapping.get("style_fill"),
                )
                return count
            velocity = _coerce_int(mapping.get("phrase_velocity"), 96)
            try:
                dur_beats = float(mapping.get("phrase_length_beats", 0.5))
            except Exception:
                dur_beats = 0.5
            try:
                seed_bpm_local = float(bpm)
            except Exception:
                seed_bpm_local = 120.0
            if not math.isfinite(seed_bpm_local) or seed_bpm_local <= 0.0:
                seed_bpm_local = 120.0
            seconds_per_beat = 60.0 / seed_bpm_local
            duration = max(1e-4, dur_beats * seconds_per_beat)
            inst_obj = _get_phrase_inst(pm_out)
            for bar_idx in end_bars:
                start_time = float(units[bar_idx][0])
                inst_obj.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=fill_pitch,
                        start=start_time,
                        end=start_time + duration,
                    )
                )
                count += 1
            return count

    if mode == "section_end":
        if len(section_layout) <= 1:
            return 0
        for sec in section_layout:
            if not isinstance(sec, dict):
                # Gracefully ignore malformed section entries that slipped through
                # normalisation (e.g. raw label strings from legacy stats dumps).
                continue
            idx = int(sec.get("end_bar", 0)) - 1
            if idx < 0 or idx >= len(units):
                continue
            if idx in used or idx in filled_tracker:
                continue
            start = units[idx][0]
            start_b = time_to_beat(start)
            length = beat_to_time(start_b + fill_length_beats) - start
            if length <= 0.0:
                continue
            candidate_pitches: List[int] = []
            for delta in (0, 1, -1):
                cand = pitch + delta
                if delta != 0 and cand == pitch:
                    continue
                if cand in candidate_pitches:
                    continue
                if cand < 0 or cand > 127:
                    continue
                if cand in avoid_set and (delta != 0 or avoid_pitches is not None):
                    continue
                candidate_pitches.append(cand)
            chosen_pitch: Optional[int] = None
            for cand in candidate_pitches:

                conflict = False
                for (
                    note_pitch,
                    note_start,
                    note_end,
                    note_start_b,
                    note_end_b,
                ) in iter_notes_for_bar(idx):
                    overlaps = start < note_end + EPS and note_start < start + length - EPS
                    if overlaps and note_pitch == cand:
                        conflict = True
                        break
                    if min_gap_beats > 0.0 and start >= note_start:
                        gap = start_b - note_end_b
                        if gap < min_gap_beats and cand in avoid_overlap:
                            conflict = True
                            break
                if conflict:
                    continue
                for st_t, en_t, st_b in inserted_spans:
                    if start < en_t + EPS and st_t < start + length - EPS:
                        conflict = True
                        break
                    if min_gap_beats > 0.0 and start_b - st_b < min_gap_beats:
                        conflict = True
                        break
                if conflict:
                    continue
                chosen_pitch = cand
                break
            if chosen_pitch is None:
                continue
            phrase_inst.notes.append(
                pretty_midi.Note(
                    velocity=int(mapping.get("phrase_velocity", 96)),
                    pitch=int(chosen_pitch),
                    start=start,
                    end=start + length,
                )
            )
            used.add(idx)
            filled_tracker.add(idx)
            inserted_spans.append((start, start + length, start_b))
            count += 1
            if filled_bars is not None:
                filled_bars.append(idx)
            if stats_ref is not None:
                fills = stats_ref.setdefault("fills", [])  # type: ignore[assignment]
                fills.append(
                    {"bar": idx, "pitch": int(chosen_pitch), "len_beats": float(fill_length_beats)}
                )
    elif mode == "long_rest" and rest_ratio_list:
        i = 0
        n = len(rest_ratio_list)
        while i < n:
            if rest_ratio_list[i] >= rest_th:
                idx = i - 1 if i > 0 else 0
                if idx not in used and idx not in filled_tracker:
                    start = units[idx][0]
                    start_b = time_to_beat(start)
                    length = beat_to_time(start_b + fill_length_beats) - start
                    conflict = False
                    for note_obj in phrase_inst.notes:
                        note_pitch = _resolve_pitch_token(note_obj, mapping)
                        if note_pitch is None:
                            continue
                        if note_pitch in avoid_overlap:
                            if start < note_obj.end + EPS and note_obj.start < start + length - EPS:
                                conflict = True
                                break
                            if min_gap_beats > 0.0:
                                gap = start_b - time_to_beat(note_obj.end)
                                if gap < min_gap_beats and start >= note_obj.start:
                                    conflict = True
                                    break
                    if not conflict:
                        for st_t, en_t, st_b in inserted_spans:
                            if start < en_t + EPS and st_t < start + length - EPS:
                                conflict = True
                                break
                            if min_gap_beats > 0.0 and start_b - st_b < min_gap_beats:
                                conflict = True
                                break
                    if conflict:
                        pass
                    else:
                        phrase_inst.notes.append(
                            pretty_midi.Note(
                                velocity=int(mapping.get("phrase_velocity", 96)),
                                pitch=pitch,
                                start=start,
                                end=start + length,
                            )
                        )
                        used.add(idx)
                        filled_tracker.add(idx)
                        inserted_spans.append((start, start + length, start_b))
                        count += 1
                        if filled_bars is not None:
                            filled_bars.append(idx)
                        if stats_ref is not None:
                            fills = stats_ref.setdefault("fills", [])  # type: ignore[assignment]
                            fills.append(
                                {
                                    "bar": idx,
                                    "pitch": int(pitch),
                                    "len_beats": float(fill_length_beats),
                                }
                            )
                while i < n and rest_ratio_list[i] >= rest_th:
                    i += 1
                continue
            i += 1
    return count


def _normalize_sections_ranges(
    sections: Optional[Sequence[Any]],
    num_bars: Optional[int],
    *,
    source: str = "cli",
) -> List[Dict[str, Any]]:
    """Normalize section specs into sorted ranges."""

    layout, _ = normalize_sections(
        sections,
        bar_count=num_bars,
        default_tag="section",
    )
    for sec in layout:
        sec["source"] = source
    return layout


def _sections_to_labels_infer(
    sections: List[Dict[str, Any]],
    num_bars: Optional[int],
    section_default: str,
) -> List[str]:
    target_units = num_bars
    if target_units is None:
        target_units = max((int(sec.get("end_bar", 0)) for sec in sections), default=0)
    _, labels = normalize_sections(
        sections,
        bar_count=target_units,
        default_tag=section_default,
    )
    if num_bars is not None and len(labels) < num_bars:
        labels.extend([section_default] * (num_bars - len(labels)))
    if num_bars is not None and len(labels) > num_bars:
        return labels[:num_bars]
    return labels


def _labels_to_sections(
    labels: List[str],
    sources: List[str],
) -> List[Dict[str, Any]]:
    if not labels:
        return []
    out: List[Dict[str, Any]] = []
    start = 0
    cur_label = str(labels[0]) if labels[0] is not None else "sec0"
    cur_source = sources[0] if sources else "auto"
    for idx in range(1, len(labels) + 1):
        if idx == len(labels) or labels[idx] != cur_label or sources[idx] != cur_source:
            out.append(
                {
                    "start_bar": start,
                    "end_bar": idx,
                    "tag": str(cur_label),
                    "source": cur_source,
                    "explicit_end": True,
                }
            )
            if idx < len(labels):
                start = idx
                cur_label = str(labels[idx]) if labels[idx] is not None else f"sec{idx}"
                cur_source = sources[idx]
    return out


def _merge_sections(
    cli_sections: Sequence[Any],
    guide_sections: Sequence[Any],
    num_bars: Optional[int],
) -> List[Dict[str, Any]]:
    cli_norm = _normalize_sections_ranges(cli_sections, num_bars, source="cli")
    guide_norm = _normalize_sections_ranges(guide_sections, num_bars, source="guide")
    if not cli_norm and not guide_norm:
        return []

    max_end = num_bars or 0
    for sec in cli_norm + guide_norm:
        if sec.get("end_bar") is not None:
            max_end = max(max_end, int(sec["end_bar"]))
    if max_end <= 0:
        return []

    guide_boundaries = sorted({sec["start_bar"] for sec in guide_norm})
    cli_boundaries = sorted({sec["start_bar"] for sec in cli_norm})

    cli_labels: List[Optional[str]] = [None] * max_end
    cli_sources: List[Optional[str]] = [None] * max_end
    for sec in cli_norm:
        start = max(0, min(max_end, int(sec["start_bar"])))
        if sec.get("explicit_end"):
            end = int(sec["end_bar"])
        else:
            next_candidates = [b for b in cli_boundaries if b > start]
            next_candidates.extend(b for b in guide_boundaries if b > start)
            candidate = min(next_candidates) if next_candidates else max_end
            end = min(candidate, int(sec.get("end_bar", max_end)))
        end = max(start + 1, min(max_end, end))
        for b in range(start, end):
            cli_labels[b] = str(sec.get("tag"))
            cli_sources[b] = "cli"

    guide_labels: List[Optional[str]] = [None] * max_end
    for sec in guide_norm:
        start = max(0, min(max_end, int(sec["start_bar"])))
        end = max(start + 1, min(max_end, int(sec.get("end_bar", max_end))))
        for b in range(start, end):
            guide_labels[b] = str(sec.get("tag"))

    last_tag: Optional[str] = None
    for b in range(max_end):
        if guide_labels[b] is not None:
            last_tag = guide_labels[b]
        elif last_tag is not None:
            guide_labels[b] = last_tag

    labels: List[str] = []
    sources: List[str] = []
    for b in range(max_end):
        if cli_labels[b] is not None:
            labels.append(str(cli_labels[b]))
            sources.append(cli_sources[b] or "cli")
        elif guide_labels[b] is not None:
            labels.append(str(guide_labels[b]))
            sources.append("guide")
        else:
            labels.append(f"sec{b}")
            sources.append("auto")

    return _labels_to_sections(labels, sources)


def _format_sections_for_log(sections: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for sec in sections:
        start = int(sec.get("start_bar", 0))
        end = int(sec.get("end_bar", start))
        tag = str(sec.get("tag", ""))
        parts.append(f"[{start:02d}\u2013{end:02d}) {tag}")
    return " | ".join(parts)


def insert_style_layer(
    pm_out: "pretty_midi.PrettyMIDI",
    mode: str,
    units: List[Tuple[float, float]],
    picker: Optional[PoolPicker],
    *,
    sections: Optional[List[str]] = None,
    every: int = 4,
    length_beats: float = 0.5,
    bpm: float = 120.0,
    mapping: Optional[Dict[str, Any]] = None,
) -> int:
    if mode == "off" or picker is None or not units:
        return 0
    phrase_inst = None
    for inst in pm_out.instruments:
        if inst.name == PHRASE_INST_NAME:
            phrase_inst = inst
            break
    if phrase_inst is None:
        phrase_inst = pretty_midi.Instrument(program=0, name=PHRASE_INST_NAME)
        pm_out.instruments.append(phrase_inst)
    seed_bpm = float(bpm) if bpm is not None else 120.0
    if not math.isfinite(seed_bpm) or seed_bpm <= 0.0:
        seed_bpm = 120.0
    _sanitize_tempi(pm_out)
    _ensure_tempo_and_ticks(pm_out, seed_bpm, pm_out.time_signature_changes)
    stats_ref = getattr(pm_out, "_sparkle_stats", None)
    beat_times: List[float]
    cached_beats: Optional[List[float]] = None
    if stats_ref:
        raw = stats_ref.get("beat_times")
        if raw:
            cached_beats = [float(bt) for bt in raw]
    if cached_beats:
        beat_times = cached_beats
    else:
        try:
            beat_times = pm_out.get_beats()
        except (AttributeError, IndexError, ValueError):
            step = 60.0 / seed_bpm
            end = units[-1][1]
            n = int(math.ceil(end / step)) + 1
            beat_times = [i * step for i in range(n)]

    def beat_to_time(b: float) -> float:
        idx = int(math.floor(b))
        frac = b - idx
        if idx >= len(beat_times) - 1:
            last = beat_times[-1] - beat_times[-2]
            return beat_times[-1] + (b - (len(beat_times) - 1)) * last
        return beat_times[idx] + frac * (beat_times[idx + 1] - beat_times[idx])

    bars: List[int]
    if mode == "every":
        bars = list(range(0, len(units), max(1, every)))
    else:  # transitions
        bars = []
        if sections:
            prev = sections[0]
            for i, sec in enumerate(sections[1:], 1):
                if sec != prev:
                    bars.append(i)
                prev = sec
    count = 0
    for b_idx in bars:
        if b_idx >= len(units):
            continue
        start = units[b_idx][0]
        start_b = b_idx  # approximate
        length = beat_to_time(start_b + length_beats) - start
        raw_pitch = picker.pick()
        pitch_val = resolve_phrase_alias(raw_pitch, mapping)
        if pitch_val is None:
            logging.warning(
                "insert_style_layer: skipping bar %d unresolved pitch %r", b_idx, raw_pitch
            )
            continue
        try:
            pitch = int(round(float(pitch_val)))
        except Exception:
            logging.warning(
                "insert_style_layer: skipping bar %d invalid pitch %r", b_idx, raw_pitch
            )
            continue
        pitch = max(0, min(127, pitch))
        velocity = 100
        phrase_inst.notes.append(
            pretty_midi.Note(
                velocity=int(velocity),
                pitch=int(pitch),
                start=float(start),
                end=float(start + length),
            )
        )
        count += 1
    return count


def finalize_phrase_track(
    out_pm: "pretty_midi.PrettyMIDI",
    args: Optional[argparse.Namespace],
    stats: Optional[Dict],
    mapping: Dict,
    *,
    section_lfo: Optional[SectionLFO] = None,
    lfo_targets: Tuple[str, ...] = (),
    downbeats: Optional[List[float]] = None,
    guide_units: Optional[List[Tuple[float, float]]] = None,
    guide_units_time: Optional[List[Tuple[float, float]]] = None,
    guide_notes: Optional[Dict[int, int]] = None,
    rest_ratios: Optional[List[float]] = None,
    onset_counts: Optional[List[int]] = None,
    chord_inst: Optional["pretty_midi.Instrument"] = None,
    phrase_inst: Optional["pretty_midi.Instrument"] = None,
    beat_to_time: Optional[Callable[[float], float]] = None,
    time_to_beat: Optional[Callable[[float], float]] = None,
    pulse_subdiv_beats: float = 1.0,
    phrase_vel: int = 0,
    phrase_merge_gap: float = 0.0,
    release_sec: float = 0.0,
    min_phrase_len_sec: float = 0.0,
    stop_min_gap_beats: float = 0.0,
    stop_velocity: int = 64,
    damp_dst: Optional[str] = None,
    damp_cc_num: int = 11,
    guide_cc: Optional[List[Tuple[float, int]]] = None,
    bpm: Optional[float] = None,
    section_overrides: Optional[List[Dict]] = None,
    fill_map: Optional[Dict[int, Tuple[int, float, float]]] = None,
    rest_silence_send_stop: bool = False,
    quantize_strength: Union[float, List[float]] = 0.0,
    write_markers: bool = False,
    marker_encoding: str = "raw",
    section_labels: Optional[List[str]] = None,
    section_default: str = "verse",
    chord_merge_gap: float = 0.01,
    clone_meta_only: bool = False,
    meta_src: str = "input",
    chords: Optional[List[ChordSpan]] = None,
) -> Dict[str, Any]:
    """Finalize phrase output by applying fills, STOPs, quantization, and reports.

    ``marker_encoding`` controls how section marker labels are sanitised before they
    are written to the MIDI file.
    """

    stats_enabled = stats is not None
    fills: List[Dict[str, Any]] = []

    # --- (A) stats の最低限初期化（codex側の要点を先に適用） ---
    if stats_enabled:
        stats.setdefault("bar_phrase_notes", {})
        stats.setdefault("bar_velocities", {})
        if downbeats and "downbeats" not in stats:
            stats["downbeats"] = list(downbeats)
        fills = stats.setdefault("fills", [])

    # --- (B) セクション入力の正規化・マージ（main側のロジックを続けて適用） ---
    downbeat_ref = (
        downbeats
        if downbeats is not None
        else (stats.get("downbeats") if stats_enabled else None)
    )
    num_bars = len(downbeat_ref) - 1 if downbeat_ref else None

    guide_section_input = section_labels
    cli_sections = section_overrides or []
    merged_sections: List[Dict[str, Any]] = []
    if cli_sections and guide_section_input:
        merged_sections = _merge_sections(cli_sections, guide_section_input, num_bars)
    elif cli_sections:
        merged_sections = _normalize_sections_ranges(cli_sections, num_bars, source="cli")
    elif guide_section_input:
        merged_sections = _normalize_sections_ranges(guide_section_input, num_bars, source="guide")

    if num_bars is None and merged_sections:
        num_bars = max((sec["end_bar"] for sec in merged_sections), default=0)

    if merged_sections:
        section_labels = _sections_to_labels_infer(merged_sections, num_bars, section_default)
    elif isinstance(guide_section_input, list):
        section_labels = [
            str(tag) if tag is not None else section_default for tag in guide_section_input
        ]

    if stats_enabled:
        stats["sections"] = section_labels or []

    if args and getattr(args, "section_verbose", False) and merged_sections:
        logging.info("section table: %s", _format_sections_for_log(merged_sections))

    # --- (以下、元の後続処理) ---
    fill_count = stats.get("fill_count", 0) if stats_enabled else 0
    fill_count += _apply_fills(
        phrase_inst,
        fill_map,
        beat_to_time,
        time_to_beat,
        downbeats,
        pulse_subdiv_beats,
        phrase_vel,
        phrase_merge_gap,
        release_sec,
        min_phrase_len_sec,
        section_lfo,
        lfo_targets,
        stats,
    )
    _inject_stops(
        phrase_inst,
        rest_silence_send_stop,
        guide_units,
        guide_notes,
        beat_to_time,
        mapping,
        pulse_subdiv_beats,
        stop_min_gap_beats,
        stop_velocity,
    )

    _apply_quantize_safe(
        out_pm,
        phrase_inst,
        quantize_strength,
        beat_to_time,
        time_to_beat,
        pulse_subdiv_beats,
        downbeats,
        chords,
    )

    bar_count = max(0, len(downbeats) - 1) if downbeats else 0
    normalized_sections: List[Dict[str, Any]] = []
    sections_candidate: Optional[Sequence[Any]] = None
    for candidate in (
        section_overrides,
        mapping.get("sections"),
        stats.get("sections_layout") if stats_enabled else None,
        stats.get("section_labels") if stats_enabled else None,
        stats.get("sections") if stats_enabled else None,
        section_labels,
    ):
        if isinstance(candidate, list):
            sections_candidate = candidate
            break
    raw_sections = sections_candidate or []
    if stats_enabled and stats.get("_section_verbose") and raw_sections:
        logging.info("section normalize input=%s", raw_sections)
    normalized_sections, section_labels_by_bar = normalize_sections(
        raw_sections,
        bar_count=bar_count,
        default_tag=section_default,
        stats=stats if stats_enabled else None,
        verbose=bool(stats_enabled and stats.get("_section_verbose")),
    )
    if stats_enabled and stats.get("_section_verbose") and normalized_sections:
        logging.info("section normalize output=%s", normalized_sections)
    if stats_enabled:
        stats["sections_layout"] = [dict(sec) for sec in normalized_sections]
        stats["section_labels"] = list(section_labels_by_bar)
        # Maintain legacy key for downstream consumers while transitioning.
        stats["sections"] = list(section_labels_by_bar)
        stats["bar_count"] = max(bar_count, int(stats.get("bar_count", bar_count)))

    _write_markers(
        out_pm,
        write_markers,
        normalized_sections,
        section_default,
        downbeats,
        marker_encoding,
    )

    auto_fill_mode = getattr(args, "auto_fill", "off") if args else "off"
    section_fill_added = 0
    if auto_fill_mode == "section_end":
        if (
            normalized_sections
            and phrase_inst is not None
            and downbeats
            and len(downbeats) >= 2
        ):
            fill_beats = 0.25
            if args is not None and getattr(args, "fill_length_beats", None) is not None:
                try:
                    fill_beats = float(args.fill_length_beats)
                except Exception:
                    fill_beats = 0.25
            if not math.isfinite(fill_beats) or fill_beats <= 0.0:
                fill_beats = 0.25
            fill_beats = max(0.25, fill_beats)
            filled_bars_stat = stats.setdefault("fill_bars", []) if stats_enabled else None
            seen_bars: Set[int] = set(filled_bars_stat or []) if filled_bars_stat else set()
            for sec in normalized_sections:
                if not isinstance(sec, dict):
                    continue
                try:
                    end_bar = int(sec.get("end_bar", 0))
                except Exception:
                    continue
                bar_idx = end_bar - 1
                if bar_idx < 0 or end_bar >= len(downbeats):
                    continue
                if bar_idx in seen_bars:
                    continue
                t1 = float(downbeats[end_bar])
                t0 = float(downbeats[end_bar - 1])
                if not math.isfinite(t0) or not math.isfinite(t1) or t1 <= t0 + EPS:
                    continue
                if beat_to_time is not None and time_to_beat is not None:
                    bar_start_b = time_to_beat(t0)
                    bar_end_b = time_to_beat(t1)
                    span_beats = max(0.0, bar_end_b - bar_start_b)
                    head_beats = min(fill_beats, span_beats * 0.5)
                    onset_beats = max(bar_start_b, bar_end_b - head_beats)
                    onset = beat_to_time(onset_beats)
                else:
                    bar_duration = max(0.0, t1 - t0)
                    if bpm is not None and math.isfinite(bpm) and bpm > 0.0:
                        seconds_per_beat = 60.0 / float(bpm)
                        headroom = min(fill_beats * seconds_per_beat, bar_duration * 0.5)
                    else:
                        headroom = bar_duration * 0.5
                    onset = max(t0, t1 - headroom)
                if not math.isfinite(onset) or onset >= t1 - EPS:
                    continue
                inserted = _insert_fill_note(
                    phrase_inst,
                    mapping,
                    onset,
                    t1,
                    velocity=phrase_vel if phrase_vel else None,
                    phrase_merge_gap=phrase_merge_gap,
                    release_sec=release_sec,
                    min_phrase_len_sec=min_phrase_len_sec,
                    stats=stats if stats_enabled else None,
                )
                if inserted:
                    section_fill_added += 1
                    if stats_enabled:
                        fills.append(
                            {"start": float(onset), "end": float(t1), "bar": int(bar_idx)}
                        )
                        if filled_bars_stat is not None and bar_idx not in filled_bars_stat:
                            filled_bars_stat.append(bar_idx)
                            seen_bars.add(bar_idx)
                    else:
                        seen_bars.add(bar_idx)

    fill_count += section_fill_added
    if auto_fill_mode != "off" and guide_units_time:
        avoid: Optional[Set[int]] = None
        if getattr(args, "fill_avoid_pitches", None):
            toks = [t.strip() for t in args.fill_avoid_pitches.split(",") if t.strip()]
            avoid = set()
            for tok in toks:
                val = parse_note_token(tok, mapping=mapping)
                if val is None:
                    raise SystemExit("fill-avoid-pitches cannot include 'rest'")
                avoid.add(val)

        filled_bars = stats.setdefault("fill_bars", []) if stats_enabled else None

        inserted = insert_style_fill(
            out_pm,
            auto_fill_mode,
            guide_units_time,
            mapping,
            sections=normalized_sections,          # ← ここだけを使う
            rest_ratio_list=rest_ratios,
            rest_th=getattr(args, "guide_rest_silence_th", None) or 0.75,
            fill_length_beats=getattr(args, "fill_length_beats", 0.25),
            bpm=bpm if bpm is not None else 120.0,
            min_gap_beats=getattr(args, "fill_min_gap_beats", 0.0),
            avoid_pitches=avoid,
            filled_bars=filled_bars,
            bar_count=bar_count,
            section_default=section_default,
        )

        fill_count += inserted

    cc_stats = _emit_damp_cc(out_pm, guide_cc, damp_dst, damp_cc_num, chord_inst, phrase_inst)

    if stats_enabled:
        stats["fill_count"] = fill_count

    if stats_enabled and guide_notes is not None:
        stats["guide_keys"] = [guide_notes.get(i) for i in sorted(guide_notes.keys())]
        if rest_ratios is not None and onset_counts is not None and guide_cc:
            sample: List[Dict[str, Any]] = []
            for i in range(min(4, len(guide_cc))):
                sample.append(
                    {
                        "bar": i,
                        "onset": onset_counts[i] if i < len(onset_counts) else 0,
                        "rest": rest_ratios[i] if i < len(rest_ratios) else 0.0,
                        "cc": guide_cc[i][1],
                    }
                )
            stats["guide_sample"] = sample
            th = getattr(args, "guide_rest_silence_th", None) if args else None
            if th is not None:
                stats["rest_silence"] = sum(1 for r in rest_ratios if r >= th)
        if cc_stats:
            stats["damp_stats"] = cc_stats
        stats["auto_fill"] = {
            "mode": auto_fill_mode,
            "count": fill_count,
            "length_beats": getattr(args, "fill_length_beats", 0.25) if args else 0.25,
        }
        density_hist: Dict[str, int] = {}
        for label in (stats.get("bar_density") or {}).values():
            if label:
                density_hist[str(label)] = density_hist.get(str(label), 0) + 1
        section_sample: List[str] = []
        section_stat = stats.get("section_labels") or stats.get("sections")
        if isinstance(section_stat, list):
            section_sample = [str(s) for s in section_stat[:4]]
        sections_norm_sample: List[Dict[str, Any]] = []
        norm_stat = stats.get("sections_norm")
        if isinstance(norm_stat, list):
            for sec in norm_stat[:2]:
                if isinstance(sec, dict):
                    sections_norm_sample.append(
                        {
                            "start": int(sec.get("start_bar", 0)),
                            "end": int(sec.get("end_bar", 0)),
                            "tag": str(sec.get("tag", "")),
                        }
                    )
        bar_total = int(stats.get("bar_count", len(downbeats or [])))
        quicklook = {
            "bpm": float(bpm) if bpm is not None and math.isfinite(bpm) else None,
            "bar_count": bar_total,
            "fill_count": fill_count,
            "density_hist": density_hist,
            "sections": section_sample,
            "sections_norm": sections_norm_sample,
        }
        stats["quicklook"] = quicklook

    if (
        stats_enabled
        and args is not None
        and getattr(args, "debug_csv", None)
        and rest_ratios is not None
        and onset_counts is not None
    ):
        with open(args.debug_csv, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                ["bar", "onset_count", "rest_ratio", "phrase_note", "cc_value", "section"]
            )
            cc_map = {i: v for i, (_, v) in enumerate(guide_cc)} if guide_cc else {}
            sect = stats.get("section_labels") or stats.get("sections") or []
            for i in range(len(onset_counts)):
                pn = stats.get("bar_phrase_notes", {}).get(i)
                writer.writerow(
                    [
                        i,
                        onset_counts[i],
                        rest_ratios[i],
                        pn if pn is not None else "",
                        cc_map.get(i, ""),
                        sect[i] if i < len(sect) else getattr(args, "section_default", section_default),
                    ]
                )

    if (
        stats_enabled
        and args is not None
        and getattr(args, "bar_summary", None)
        and rest_ratios is not None
        and onset_counts is not None
    ):
        with open(args.bar_summary, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "bar",
                    "section",
                    "phrase_note",
                    "pulses_emitted",
                    "onsets",
                    "rest_ratio",
                    "avg_vel",
                    "fill_flag",
                    "cc_value",
                ]
            )
            sect = stats.get("section_labels") or stats.get("sections") or []
            bar_vel = stats.get("bar_velocities", {})
            fill_bars = set(stats.get("fill_bars", []))
            cc_map = {i: v for i, (_, v) in enumerate(guide_cc)} if guide_cc else {}
            for i in range(len(onset_counts)):
                pn = stats.get("bar_phrase_notes", {}).get(i)
                pulses = len(stats.get("bar_pulses", {}).get(i, []))
                vel_list = bar_vel.get(i, [])
                avg_vel = sum(vel_list) / len(vel_list) if vel_list else ""
                writer.writerow(
                    [
                        i,
                        sect[i] if i < len(sect) else getattr(args, "section_default", section_default),
                        pn if pn is not None else "",
                        pulses,
                        onset_counts[i],
                        rest_ratios[i],
                        avg_vel,
                        1 if i in fill_bars else 0,
                        cc_map.get(i, ""),
                    ]
                )

    if stats_enabled and args is not None and getattr(args, "report_json", None):
        p = Path(args.report_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(stats, indent=2, sort_keys=True))

    return {"fill_count": fill_count, "cc_stats": cc_stats}


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


def validate_style_inject_cfg(cfg: Dict, mapping: Optional[Dict[str, Any]] = None) -> Dict:
    if not isinstance(cfg, dict):
        raise SystemExit("style_inject must be object")
    if int(cfg.get("period", 0)) < 1:
        raise SystemExit("style_inject.period must be >=1")
    note = cfg.get("note")
    if note is None:
        raise SystemExit("style_inject.note 0..127 required")
    resolved_note = _resolve_pitch_token(note, mapping)
    if resolved_note is None:
        parsed = parse_note_token(note, mapping=mapping, warn_unknown=True)
        if parsed is None:
            raise SystemExit("style_inject.note must be MIDI note token")
        resolved_note = parsed
    cfg["note"] = validate_midi_note(int(resolved_note))
    if float(cfg.get("duration_beats", 0)) <= 0:
        raise SystemExit("style_inject.duration_beats must be >0")
    if "min_gap_beats" in cfg and float(cfg["min_gap_beats"]) < 0:
        raise SystemExit("style_inject.min_gap_beats must be >=0")
    if "avoid_pitches" in cfg:
        if not isinstance(cfg["avoid_pitches"], list):
            raise SystemExit("style_inject.avoid_pitches must be list")
        resolved: List[int] = []
        for n in cfg["avoid_pitches"]:
            resolved_note = _resolve_pitch_token(n, mapping)
            if resolved_note is None:
                parsed = parse_note_token(n, mapping=mapping, warn_unknown=True)
                if parsed is None:
                    raise SystemExit(
                        "style_inject.avoid_pitches entries must be MIDI note tokens"
                    )
                resolved_note = parsed
            resolved.append(validate_midi_note(int(resolved_note)))
        cfg["avoid_pitches"] = resolved
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


def _apply_fills(
    phrase_inst: Optional["pretty_midi.Instrument"],
    fill_map: Optional[Dict[int, Tuple[int, float, float]]],
    beat_to_time: Optional[Callable[[float], float]],
    time_to_beat: Optional[Callable[[float], float]],
    downbeats: Optional[List[float]],
    pulse_subdiv_beats: float,
    phrase_vel: int,
    phrase_merge_gap: float,
    release_sec: float,
    min_phrase_len_sec: float,
    section_lfo: Optional[SectionLFO],
    lfo_targets: Tuple[str, ...],
    stats: Optional[Dict[str, Any]],
) -> int:
    """Insert configured fill hits at section ends and return the added count."""

    if (
        not fill_map
        or phrase_inst is None
        or beat_to_time is None
        or time_to_beat is None
        or downbeats is None
    ):
        return 0

    stats_enabled = stats is not None
    fill_count = 0
    for bar_idx, pitch in sorted(fill_map.items()):
        if pitch is None or bar_idx + 1 >= len(downbeats):
            continue
        dur_beats = pulse_subdiv_beats
        vscale = 1.0
        note = pitch
        if isinstance(pitch, tuple):
            note = pitch[0]
            if len(pitch) > 1 and pitch[1] is not None:
                dur_beats = float(pitch[1])
            if len(pitch) > 2 and pitch[2] is not None:
                vscale = float(pitch[2])
        start_b = time_to_beat(downbeats[bar_idx + 1]) - dur_beats
        end_b = time_to_beat(downbeats[bar_idx + 1])
        start_t = beat_to_time(start_b)
        end_t = beat_to_time(end_b)
        vel = phrase_vel
        if section_lfo and "fill" in lfo_targets:
            try:
                vel = max(1, min(127, int(round(vel * section_lfo.vel_scale(bar_idx)))))
                if stats_enabled:
                    stats.setdefault("lfo_pos", {})[bar_idx] = section_lfo._pos(bar_idx)
            except Exception:
                pass
        vel = max(1, min(127, int(round(vel * vscale))))
        _append_phrase(
            phrase_inst,
            int(note),
            start_t,
            end_t,
            vel,
            phrase_merge_gap,
            release_sec,
            min_phrase_len_sec,
        )
        fill_count += 1
    return fill_count


def _inject_stops(
    phrase_inst: Optional["pretty_midi.Instrument"],
    rest_silence_send_stop: bool,
    guide_units: Optional[List[Tuple[float, float]]],
    guide_notes: Optional[Dict[int, int]],
    beat_to_time: Optional[Callable[[float], float]],
    mapping: Dict,
    pulse_subdiv_beats: float,
    stop_min_gap_beats: float,
    stop_velocity: int,
) -> None:
    """Inject STOP key hits after long guide rests when enabled."""

    if (
        not rest_silence_send_stop
        or not guide_units
        or guide_notes is None
        or beat_to_time is None
        or phrase_inst is None
    ):
        return

    stop_pitch = mapping.get("style_stop") if isinstance(mapping, dict) else None
    if stop_pitch is None:
        return

    last_b = -1e9
    for idx, (sb, _) in enumerate(guide_units):
        if guide_notes.get(idx) is None and (sb - last_b) >= stop_min_gap_beats - EPS:
            st = beat_to_time(sb)
            en = beat_to_time(sb + min(0.1, pulse_subdiv_beats))
            phrase_inst.notes.append(
                pretty_midi.Note(
                    velocity=int(stop_velocity),
                    pitch=int(stop_pitch),
                    start=st,
                    end=en,
                )
            )
            last_b = sb


def _apply_quantize_safe(
    out_pm: "pretty_midi.PrettyMIDI",
    phrase_inst: Optional["pretty_midi.Instrument"],
    quantize_strength: Union[float, List[float]],
    beat_to_time: Optional[Callable[[float], float]],
    time_to_beat: Optional[Callable[[float], float]],
    pulse_subdiv_beats: float,
    downbeats: Optional[List[float]],
    chords: Optional[List[ChordSpan]],
) -> None:
    """Apply quantisation and clip notes back to bar/chord bounds."""

    if (
        phrase_inst is None
        or beat_to_time is None
        or time_to_beat is None
        or downbeats is None
        or not phrase_inst.notes
    ):
        return

    qs_list = quantize_strength if isinstance(quantize_strength, list) else None
    qs_val = float(quantize_strength) if not isinstance(quantize_strength, list) else 0.0
    do_quant = (qs_list and any(x > 0 for x in qs_list)) or (
        qs_list is None and qs_val > 0.0
    )
    if do_quant:
        for idx, note in enumerate(phrase_inst.notes):
            sb = time_to_beat(note.start)
            eb = time_to_beat(note.end)
            grid_s = round(sb / pulse_subdiv_beats) * pulse_subdiv_beats
            grid_e = round(eb / pulse_subdiv_beats) * pulse_subdiv_beats
            if qs_list:
                if pulse_subdiv_beats > 0:
                    start_idx = int(round(sb / pulse_subdiv_beats))
                    end_idx = int(round(eb / pulse_subdiv_beats))
                else:
                    start_idx = end_idx = 0
                strength_s = qs_list[start_idx % len(qs_list)]
                strength_e = qs_list[end_idx % len(qs_list)]
            else:
                strength_s = strength_e = qs_val
            sb = sb + (grid_s - sb) * strength_s
            eb = eb + (grid_e - eb) * strength_e
            note.start = beat_to_time(sb)
            note.end = beat_to_time(eb)

    chord_spans = chords or []
    chord_starts = [span.start for span in chord_spans]
    try:
        end_time = float(out_pm.get_end_time())
    except Exception:
        end_time = downbeats[-1] if downbeats else 0.0
    for note in phrase_inst.notes:
        bar_idx = max(0, bisect.bisect_right(downbeats, note.start) - 1)
        bar_start = downbeats[bar_idx] if bar_idx < len(downbeats) else 0.0
        if bar_idx + 1 < len(downbeats):
            bar_end = downbeats[bar_idx + 1]
        else:
            bar_end = end_time
        clip_start = bar_start
        clip_end = bar_end
        if chord_starts:
            c_idx = bisect.bisect_right(chord_starts, note.start) - 1
            if 0 <= c_idx < len(chord_spans):
                span = chord_spans[c_idx]
                clip_start = max(clip_start, span.start)
                clip_end = min(clip_end, span.end)
        note.start = max(note.start, clip_start)
        note.end = min(note.end, clip_end)
        if note.end <= note.start + EPS:
            note.end = min(clip_end, max(note.start + EPS, clip_start + EPS))


def _encode_marker_label(label: str, mode: str) -> str:
    if mode == "raw":
        return label
    normalized = unicodedata.normalize("NFC", label)
    label_up = normalized.upper()
    if mode == "ascii":
        encoded_chars: List[str] = []
        for ch in label_up:
            code = ord(ch)
            if 32 <= code < 127:
                encoded_chars.append(ch)
            else:
                encoded_chars.append("?")
        return "".join(encoded_chars)
    if mode == "escape":
        encoded: List[str] = []
        for ch in label_up:
            code = ord(ch)
            if 32 <= code < 127:
                encoded.append(ch)
            else:
                encoded.append(f"\\u{code:04x}")
        return "".join(encoded)
    return label_up


def _write_markers(
    out_pm: "pretty_midi.PrettyMIDI",
    write_markers: bool,
    sections: Optional[List[Dict[str, Any]]],
    section_default: str,
    downbeats: Optional[List[float]],
    marker_encoding: str,
) -> None:
    """Emit section markers when requested via CLI flags."""

    if not write_markers or downbeats is None:
        return
    mode = (marker_encoding or "raw").strip().lower()
    if mode not in {"raw", "ascii", "escape"}:
        raise SystemExit(f"unknown marker-encoding: {marker_encoding}")
    bar_count = max(0, len(downbeats) - 1)
    _, labels = normalize_sections(
        sections or [],
        bar_count=bar_count,
        default_tag=section_default,
    )
    for i, t in enumerate(downbeats):
        if labels:
            label = labels[i] if i < len(labels) else labels[-1]
        else:
            label = section_default
        encoded = _encode_marker_label(label, mode)
        try:
            out_pm.markers.append(pretty_midi.Marker(encoded, t))
        except Exception:
            pass


def _emit_damp_cc(
    out_pm: "pretty_midi.PrettyMIDI",
    guide_cc: Optional[List[Tuple[float, int]]],
    damp_dst: Optional[str],
    damp_cc_num: int,
    chord_inst: Optional["pretty_midi.Instrument"],
    phrase_inst: Optional["pretty_midi.Instrument"],
) -> Optional[Dict[str, Any]]:
    """Emit damping CC automation and return simple statistics."""

    if not guide_cc or damp_dst is None:
        return None

    if damp_dst == "phrase":
        inst = next((i for i in out_pm.instruments if i.name == PHRASE_INST_NAME), None)
        if inst is None:
            inst = pretty_midi.Instrument(program=0, name=PHRASE_INST_NAME)
            out_pm.instruments.append(inst)
    elif damp_dst == "chord":
        inst = next((i for i in out_pm.instruments if i.name == CHORD_INST_NAME), None)
        if inst is None:
            inst = chord_inst
            if inst is not None and inst not in out_pm.instruments:
                out_pm.instruments.append(inst)
    else:
        inst = pretty_midi.Instrument(program=0, name=DAMP_INST_NAME)
        out_pm.instruments.append(inst)

    if inst is None:
        return None

    for t, v in guide_cc:
        inst.control_changes.append(
            pretty_midi.ControlChange(number=int(damp_cc_num), value=int(v), time=t)
        )
    vals = [v for _, v in guide_cc]
    if not vals:
        return None
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
        "count": len(vals),
    }


def parse_chord_symbol(symbol: str) -> Tuple[int, str]:
    """Parse a chord symbol like ``G:maj`` -> ``(7, "maj")``.

    Normalises common quality aliases (maj/M/major, min/m/minor) and supports
    mixed-width accidentals for the root token.
    """

    if not symbol:
        raise ValueError("Chord symbol cannot be empty")
    parts = symbol.split(":")
    if len(parts) != 2:
        raise ValueError(f"Chord symbol '{symbol}' must be in Root:Quality format")
    raw_root, raw_quality = parts
    root = (
        raw_root.strip()
        .replace("＃", "#")
        .replace("♯", "#")
        .replace("♭", "b")
        .replace("ｂ", "b")
    )
    quality_raw = raw_quality.strip()
    if not root:
        raise ValueError(f"Chord symbol '{symbol}' missing root")
    if not quality_raw:
        raise ValueError(f"Chord symbol '{symbol}' missing quality")

    # Normalise casing while preserving accidentals (e.g., Bb, C#)
    if len(root) >= 1:
        leading = root[0].upper()
        trailing = root[1:]
        if trailing:
            trailing = trailing.replace("＃", "#").replace("♯", "#").replace("♭", "b").replace("ｂ", "b")
            trailing = trailing.lower()
        root = leading + trailing

    canonical_quality = _canonicalize_quality(quality_raw, error_type=ValueError)
    quality_base = canonical_quality.split("/", 1)[0]

    # Accept shorthand like "Em" (root token includes trailing 'm').
    if root not in PITCH_CLASS and quality_base == "min" and root.endswith("m"):
        candidate = root[:-1]
        if candidate:
            candidate = candidate[0].upper() + candidate[1:]
        if candidate in PITCH_CLASS:
            root = candidate

    pc = PITCH_CLASS.get(root)
    if pc is None:
        raise ValueError(f"Unknown chord root '{root}' in symbol '{symbol}'")
    return pc, canonical_quality


def parse_time_sig(default_num=4, default_den=4) -> Tuple[int, int]:
    # pretty_midi doesn't store TS per track reliably; keep configurable if needed
    return default_num, default_den


def parse_chords_ts_arg(spec: Optional[str]) -> List[Tuple[int, int, int]]:
    if not spec:
        return []
    hint_map: Dict[int, Tuple[int, int]] = {}
    tokens: List[str] = []
    text = spec.strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except Exception as exc:
            logging.warning("--chords-ts JSON parse failed: %s", exc)
            return []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str):
                    tokens.append(item.strip())
                else:
                    logging.warning("--chords-ts ignoring non-string entry %r", item)
        else:
            logging.warning("--chords-ts JSON must be an array of strings")
            return []
    else:
        tokens = [tok.strip() for tok in text.split(",") if tok.strip()]

    for idx, token in enumerate(tokens):
        if not token:
            continue
        meter_txt, bar_txt = token, "0"
        if "@" in token:
            meter_txt, bar_txt = token.split("@", 1)
        meter_txt = meter_txt.strip()
        bar_txt = bar_txt.strip()
        if not meter_txt or "/" not in meter_txt:
            logging.warning(
                "--chords-ts entry %d ('%s') must be NUM/DEN@bar; skipping", idx, token
            )
            continue
        num_txt, den_txt = meter_txt.split("/", 1)
        try:
            num = int(num_txt)
            den = int(den_txt)
        except Exception as exc:
            logging.warning(
                "--chords-ts entry %d ('%s') has non-integer meter (%s); skipping",
                idx,
                token,
                exc,
            )
            continue
        if num <= 0 or den <= 0:
            logging.warning(
                "--chords-ts entry %d ('%s') requires positive meter; skipping", idx, token
            )
            continue
        try:
            bar = int(bar_txt) if bar_txt else 0
        except Exception as exc:
            logging.warning(
                "--chords-ts entry %d ('%s') has invalid bar index (%s); skipping",
                idx,
                token,
                exc,
            )
            continue
        if bar < 0:
            logging.warning(
                "--chords-ts entry %d ('%s') must use non-negative bar index; skipping",
                idx,
                token,
            )
            continue
        hint_map[bar] = (num, den)
    return [(bar, vals[0], vals[1]) for bar, vals in sorted(hint_map.items())]


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
            guard = MAX_ITERS
            while p < lo:
                if guard == 0:
                    logging.warning(
                        "place_in_range: max iterations while raising closed voicing note"
                    )
                    break
                guard -= 1
                p += 12
            guard = MAX_ITERS
            while p > hi:
                if guard == 0:
                    logging.warning(
                        "place_in_range: max iterations while lowering closed voicing note"
                    )
                    break
                guard -= 1
                p -= 12
            res.append(p)
        res.sort()
        changed = True
        change_guard = MAX_ITERS
        while changed:
            if change_guard == 0:
                logging.warning(
                    "place_in_range: max iterations while normalizing closed voicing"
                )
                break
            change_guard -= 1
            changed = False
            res.sort()
            for i in range(1, len(res)):
                gap_guard = MAX_ITERS
                while res[i] - res[i - 1] > 12 and res[i] - 12 >= lo:
                    if gap_guard == 0:
                        logging.warning(
                            "place_in_range: max iterations while tightening closed gaps"
                        )
                        break
                    gap_guard -= 1
                    res[i] -= 12
                    changed = True
        for i in range(len(res)):
            lower_guard = MAX_ITERS
            while res[i] > hi and res[i] - 12 >= lo:
                if lower_guard == 0:
                    logging.warning(
                        "place_in_range: max iterations while lowering closed note into range"
                    )
                    break
                lower_guard -= 1
                res[i] -= 12
        res.sort()
        return res

    for p in pitches:
        guard = MAX_ITERS
        while p < lo:
            if guard == 0:
                logging.warning(
                    "place_in_range: max iterations while raising stacked voicing note"
                )
                break
            guard -= 1
            p += 12
        guard = MAX_ITERS
        while p > hi:
            if guard == 0:
                logging.warning(
                    "place_in_range: max iterations while lowering stacked voicing note"
                )
                break
            guard -= 1
            p -= 12
        if prev is not None:
            order_guard = MAX_ITERS
            while p <= prev:
                if order_guard == 0:
                    logging.warning(
                        "place_in_range: max iterations while enforcing ascending order"
                    )
                    break
                order_guard -= 1
                p += 12
        prev = p
        res.append(p)
    if res and res[-1] > hi:
        adjust_guard = MAX_ITERS
        while any(p > hi for p in res) and all(p - 12 >= lo for p in res):
            if adjust_guard == 0:
                logging.warning(
                    "place_in_range: max iterations while lowering stacked chord into range"
                )
                break
            adjust_guard -= 1
            res = [p - 12 for p in res]
        if any(p > hi for p in res):
            logging.warning("place_in_range: notes fall outside range %s-%s", lo, hi)
    return res


def smooth_triad(prev: Optional[List[int]], curr: List[int], lo: int, hi: int) -> List[int]:
    if not prev:
        return curr
    best = curr
    prev_sorted = sorted(prev)
    combos = []
    for offs in itertools.product([-12, 0, 12], repeat=len(curr)):
        cand = [p + o for p, o in zip(curr, offs)]
        if all(lo <= n <= hi for n in cand):
            combos.append(cand)
    if not combos:
        return curr

    def cost(c: List[int]) -> int:
        return sum(abs(a - b) for a, b in zip(sorted(c), prev_sorted))

    best = min(combos, key=cost)
    return best


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
            dedent(
                """
            phrase_note: 36
            phrase_velocity: 96
            phrase_length_beats: 0.25
            phrase_hold: off  # off, bar, chord
            phrase_merge_gap: 0.02  # seconds
            chord_merge_gap: 0.01  # seconds
            chord_octave: 4
            chord_velocity: 90
            triad_intervals:
              maj: [0,4,7]
              min: [0,3,7]
            cycle_phrase_notes: []  # e.g., [24, rest, 26] to alternate per bar (小節ごとに切替)
            cycle_start_bar: 0
            cycle_mode: bar  # or 'chord'
            cycle_stride: 1  # number of bars/chords before advancing cycle
            merge_reset_at: none  # none, bar, chord
            voicing_mode: stacked  # or 'closed'
            top_note_max: null  # e.g., 72 to cap highest chord tone
            phrase_channel: null  # MIDI channel for phrase notes
            chord_channel: null  # MIDI channel for chord notes
            accent: []  # velocity multipliers per pulse
            skip_phrase_in_rests: false
            clone_meta_only: false
            silent_qualities: []
            swing: 0.0  # 0..1 swing feel
            swing_unit: "1/8"  # subdivision for swing
            chord_input_range: {lo: 48, hi: 72}
        """
            )
            .lstrip()
            .rstrip()
            + "\n"
        )
    else:
        return (
            dedent(
                """
            phrase_note: 36
            cycle_phrase_notes: []  # e.g., [24, rest, 26] to alternate per bar (小節ごとに切替)
            phrase_hold: off
            phrase_merge_gap: 0.02
            chord_merge_gap: 0.01
            cycle_start_bar: 0
            cycle_mode: bar  # or 'chord'
            cycle_stride: 1
            merge_reset_at: none
            voicing_mode: stacked  # or 'closed'
            top_note_max: null
            phrase_channel: null
            chord_channel: null
            accent: []
            skip_phrase_in_rests: false
            clone_meta_only: false
            silent_qualities: []
            swing: 0.0
            swing_unit: "1/8"
            chord_input_range: {lo: 48, hi: 72}
        """
            )
            .lstrip()
            .rstrip()
            + "\n"
        )

class ChordCsvError(SystemExit, ValueError):
    """Raised when a chord CSV cannot be parsed."""


@dataclass
class _CompactChordRow:
    line_no: int
    start_bar: int
    symbol: str
    start_beat: Fraction = Fraction(0)
    end_bar: Optional[int] = None
    raw_symbol: str = ""
    length_beats: Optional[Fraction] = None


_QUALITY_TOKEN_MAP: Dict[str, str] = {
    "maj": "maj",
    "major": "maj",
    "m": "min",
    "min": "min",
    "minor": "min",
    "maj7": "maj7",
    "m7": "m7",
    "min7": "m7",
    "m7b5": "m7b5",
    "half-diminished": "m7b5",
    "halfdim": "m7b5",
    "ø": "m7b5",
    "ø7": "m7b5",
    "dim": "dim",
    "dim7": "dim7",
    "diminished": "dim",
    "diminished7": "dim7",
    "aug": "aug",
    "augmented": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "7sus4": "7sus4",
    "add9": "add9",
    "7": "7",
}

_QUALITY_SUFFIX_MAP: Dict[str, str] = {
    "m7b5": "m7b5",
    "dim7": "dim7",
    "maj7": "maj7",
    "min7": "m7",
    "m7": "m7",
    "7sus4": "7sus4",
    "sus4": "sus4",
    "sus2": "sus2",
    "add9": "add9",
    "dim": "dim",
    "aug": "aug",
    "minor": "min",
    "major": "maj",
    "min": "min",
    "maj": "maj",
    "7": "7",
    "m": "min",
}

_QUALITY_SUFFIXES = tuple(sorted(_QUALITY_SUFFIX_MAP.keys(), key=len, reverse=True))

_ACCIDENTAL_CHARS = {"#", "♯", "＃", "b", "♭", "ｂ"}


def _canonicalize_quality(raw: str, *, error_type: Type[Exception]) -> str:
    text = raw.strip()
    if not text:
        raise error_type("quality cannot be empty")
    if "/" in text:
        base, slash_part = text.split("/", 1)
        slash = "/" + slash_part.strip()
    else:
        base, slash = text, ""
    base = base.strip()
    if not base:
        raise error_type("quality cannot be empty")
    if base == "M":
        canonical = "maj"
    elif base == "m":
        canonical = "min"
    elif base in {"+", "aug+"}:
        canonical = "aug"
    elif base == "-":
        canonical = "min"
    elif base in {"ø", "Ø", "ø7", "Ø7"}:
        canonical = "m7b5"
    elif base in {"°", "o"}:
        canonical = "dim"
    else:
        base_lower = base.lower()
        canonical = _QUALITY_TOKEN_MAP.get(base_lower, base)
    return f"{canonical}{slash}"


def _guess_root_display(token: str, fallback: str) -> str:
    stripped = token.strip()
    if not stripped:
        return fallback
    if ":" in stripped:
        head = stripped.split(":", 1)[0].strip()
        return head or fallback
    first = stripped[0]
    if first.upper() not in "ABCDEFG":
        return fallback
    root = first
    if len(stripped) >= 2 and stripped[1] in _ACCIDENTAL_CHARS:
        root += stripped[1]
    return root


def _normalize_compact_chord(token: str) -> str:
    """Return a ``Root:Quality`` symbol for a compact chord token."""

    if not token:
        raise ChordCsvError("empty chord token")
    normalized = (
        token.strip()
        .replace("＃", "#")
        .replace("♯", "#")
        .replace("♭", "b")
        .replace("ｂ", "b")
    )
    if not normalized:
        raise ChordCsvError("empty chord token")
    normalized = normalized.lstrip("\ufeff")
    if ":" in normalized:
        root_txt, qual_txt = normalized.split(":", 1)
        root_txt = root_txt.strip()
        qual_txt = qual_txt.strip()
        if not root_txt or not qual_txt:
            raise ChordCsvError(f"invalid chord token '{token}'")
        quality = _canonicalize_quality(qual_txt, error_type=ChordCsvError)
        return f"{root_txt}:{quality}"

    slash_suffix = ""
    if "/" in normalized:
        base, bass = normalized.split("/", 1)
        normalized = base
        slash_suffix = f"/{bass.strip()}" if bass.strip() else ""

    body = normalized
    quality = "maj"
    if body.endswith("M"):
        body = body[:-1]
        quality = "maj"
    elif body.endswith("+"):
        body = body[:-1]
        quality = "aug"
    elif body.endswith("-"):
        body = body[:-1]
        quality = "min"
    else:
        lowered = body.lower()
        for suffix in _QUALITY_SUFFIXES:
            if lowered.endswith(suffix):
                body = body[: -len(suffix)] if suffix else body
                quality = _QUALITY_SUFFIX_MAP[suffix]
                break
    body = body.strip()
    if not body:
        raise ChordCsvError(f"invalid chord token '{token}'")
    return f"{body}:{quality}{slash_suffix}" if slash_suffix else f"{body}:{quality}"


def _parse_fraction_token(text: str, *, path: Path, line_no: int) -> Fraction:
    cleaned = text.strip()
    if not cleaned:
        raise ChordCsvError(f"{path} line {line_no}: beat value cannot be empty")
    cleaned = cleaned.replace(" ", "")
    try:
        frac = Fraction(cleaned)
    except ValueError as exc:
        raise ChordCsvError(
            f"{path} line {line_no}: beat '{text}' must be integer, float, or fraction"
        ) from exc
    if frac < 0:
        raise ChordCsvError(f"{path} line {line_no}: beat must be >= 0 (got {text})")
    return frac


def _parse_bar_chord_rows(
    rows: Iterable[Sequence[str]],
    *,
    start_line: int,
    path: Path,
    header: Optional[Sequence[str]] = None,
) -> List[_CompactChordRow]:
    """Parse compact chord rows into structured entries."""

    allowed = {"bar", "chord", "bar_start", "bar_end", "beat", "beats"}
    if header:
        header = [cell.strip().lower() for cell in header]
        if header:
            header[0] = header[0].lstrip("\ufeff")
        unexpected = [col for col in header if col and col not in allowed]
        if unexpected:
            raise ChordCsvError(
                f"{path}: unsupported column(s) {unexpected}; expected subset of "
                f"{sorted(allowed)}"
            )
        columns = list(header)
    else:
        columns = ["bar", "chord"]
        for raw_row in rows:
            cells = [cell.strip() for cell in raw_row]
            if not any(cells):
                continue
            if len(cells) >= 3 and cells[2]:
                columns = ["bar", "chord", "beats"]
            break

    if not columns or "chord" not in columns:
        raise ChordCsvError(f"{path}: chord CSV must include a 'chord' column")

    parsed: List[_CompactChordRow] = []
    last_pos: Optional[Tuple[int, Fraction]] = None
    for offset, raw_row in enumerate(rows):
        line_no = start_line + offset
        if not raw_row:
            continue
        cells = [cell.strip() for cell in raw_row]
        if not any(cells):
            continue
        cells[0] = cells[0].lstrip("\ufeff")
        if len(cells) < len(columns):
            cells = cells + [""] * (len(columns) - len(cells))
        if len(cells) > len(columns):
            extras = cells[len(columns) :]
            if any(extras):
                logging.warning(
                    "%s line %d: skipping row with unexpected extra columns %s",
                    path,
                    line_no,
                    extras,
                )
                continue
            cells = cells[: len(columns)]

        row_map = dict(zip(columns, cells))
        chord_txt = row_map.get("chord", "")
        if not chord_txt:
            raise ChordCsvError(f"{path} line {line_no}: missing chord symbol")
        symbol = _normalize_compact_chord(chord_txt)

        bar_token = row_map.get("bar_start") or row_map.get("bar")
        if not bar_token:
            raise ChordCsvError(f"{path} line {line_no}: missing bar index")
        try:
            bar_idx = int(bar_token)
        except ValueError as exc:
            raise ChordCsvError(
                f"{path} line {line_no}: bar '{bar_token}' must be an integer"
            ) from exc
        if bar_idx < 0:
            raise ChordCsvError(
                f"{path} line {line_no}: bar must be >= 0 (got {bar_idx})"
            )

        beat_token = row_map.get("beat")
        start_beat = Fraction(0)
        if beat_token:
            start_beat = _parse_fraction_token(beat_token, path=path, line_no=line_no)

        end_token = row_map.get("bar_end")
        end_bar: Optional[int] = None
        if end_token:
            try:
                end_bar = int(end_token)
            except ValueError as exc:
                raise ChordCsvError(
                    f"{path} line {line_no}: bar_end '{end_token}' must be an integer"
                ) from exc
            if end_bar < 0:
                raise ChordCsvError(
                    f"{path} line {line_no}: bar_end must be >= 0 (got {end_bar})"
                )
            if end_bar <= bar_idx:
                raise ChordCsvError(
                    f"{path} line {line_no}: bar_end {end_bar} must be greater than bar {bar_idx}"
                )

        pos = (bar_idx, start_beat)
        if last_pos is not None:
            if pos == last_pos:
                raise ChordCsvError(
                    f"{path} line {line_no}: duplicate bar/beat position {bar_idx}"
                )
            if pos < last_pos:
                raise ChordCsvError(
                    f"{path} line {line_no}: bars must be ascending (previous {last_pos[0]})"
                )
        last_pos = pos

        beats_len_token = row_map.get("beats")
        length_beats: Optional[Fraction] = None
        if beats_len_token:
            try:
                length_beats = _parse_fraction_token(
                    beats_len_token, path=path, line_no=line_no
                )
            except ChordCsvError as exc:
                logging.warning(str(exc))
                continue

        parsed.append(
            _CompactChordRow(
                line_no=line_no,
                start_bar=bar_idx,
                symbol=symbol,
                start_beat=start_beat,
                end_bar=end_bar,
                raw_symbol=chord_txt.strip(),
                length_beats=length_beats,
            )
        )

    return parsed


class _BarTimeline:
    """Helper generating bar boundary times from mixed timing inputs."""

    def __init__(
        self,
        *,
        path: Path,
        bar_times: Optional[List[float]],
        beat_times: Optional[List[float]],
        meter_map: Optional[List[Tuple[float, int, int]]],
        meter_hints: Optional[List[Tuple[int, int, int]]],
        bpm_hint: Optional[float],
        default_meter: Tuple[int, int],
    ) -> None:
        self._path = path
        num, den = default_meter
        self._default_meter = (max(1, int(num)), max(1, int(den)))
        bpm_value = bpm_hint if bpm_hint and math.isfinite(bpm_hint) and bpm_hint > 0 else 120.0
        self._seconds_per_quarter = 60.0 / bpm_value

        self._beat_times: List[float] = []
        if beat_times:
            seq = [float(t) for t in beat_times if math.isfinite(float(t))]
            seq.sort()
            filtered: List[float] = []
            for t in seq:
                if not filtered or t > filtered[-1] + EPS:
                    filtered.append(t)
            if len(filtered) >= 2:
                self._beat_times = filtered
        self._beat_len = len(self._beat_times)

        self._meter_seq: List[Tuple[float, int, int]] = []
        self._meter_times: List[float] = []
        if meter_map:
            cleaned: List[Tuple[float, int, int]] = []
            for entry in meter_map:
                mt, mnum, mden = entry
                if mden == 0:
                    raise ChordCsvError(f"{path}: meter denominator cannot be zero")
                cleaned.append((float(mt), int(mnum), int(mden)))
            cleaned.sort(key=lambda item: item[0])
            self._meter_seq = cleaned
            self._meter_times = [mt for mt, _, _ in cleaned]

        self._meter_hints: List[Tuple[int, int, int]] = []
        self._meter_hint_bars: List[int] = []
        if meter_hints:
            hints: List[Tuple[int, int, int]] = []
            for bar, mnum, mden in meter_hints:
                if mden == 0:
                    raise ChordCsvError(f"{path}: meter hint denominator cannot be zero")
                if mnum <= 0:
                    raise ChordCsvError(f"{path}: meter hint numerator must be >0 (bar {bar})")
                hints.append((int(bar), int(mnum), int(mden)))
            hints.sort(key=lambda item: item[0])
            self._meter_hints = hints
            self._meter_hint_bars = [bar for bar, _, _ in hints]

        initial: List[float] = []
        if bar_times:
            for idx, t in enumerate(bar_times):
                try:
                    ft = float(t)
                except Exception as exc:
                    raise ChordCsvError(
                        f"{path}: bar_times entry {idx} ('{t}') must be numeric"
                    ) from exc
                if not initial or ft > initial[-1] + EPS:
                    initial.append(ft)
        self._explicit_boundaries = bool(initial)
        if not initial:
            initial = [0.0]
        if not bar_times:
            self._explicit_boundaries = False
        self._boundaries = initial

    def _beat_to_time(self, beat_pos: float) -> float:
        if self._beat_len < 2:
            return beat_pos * self._seconds_per_quarter
        idx = int(math.floor(beat_pos))
        if idx < 0:
            return self._beat_times[0]
        if idx >= self._beat_len - 1:
            last = self._beat_times[-1] - self._beat_times[-2]
            if last <= 0.0:
                last = self._seconds_per_quarter
            return self._beat_times[-1] + (beat_pos - (self._beat_len - 1)) * last
        frac = beat_pos - idx
        return self._beat_times[idx] + frac * (self._beat_times[idx + 1] - self._beat_times[idx])

    def _time_to_beat(self, time_pos: float) -> float:
        if self._beat_len < 2:
            if self._seconds_per_quarter <= 0.0:
                return 0.0
            return time_pos / self._seconds_per_quarter
        idx = bisect.bisect_right(self._beat_times, time_pos) - 1
        if idx < 0:
            return 0.0
        if idx >= self._beat_len - 1:
            last = self._beat_times[-1] - self._beat_times[-2]
            if last <= 0.0:
                last = self._seconds_per_quarter
            return (self._beat_len - 1) + (time_pos - self._beat_times[-1]) / last
        span = self._beat_times[idx + 1] - self._beat_times[idx]
        if span <= 0.0:
            return float(idx)
        return idx + (time_pos - self._beat_times[idx]) / span

    def _meter_for(self, bar_index: int, start_time: float) -> Tuple[int, int]:
        if self._meter_seq:
            num, den = get_meter_at(self._meter_seq, start_time, times=self._meter_times)
            return int(num), int(den)
        if self._meter_hints:
            idx = bisect.bisect_right(self._meter_hint_bars, bar_index) - 1
            if idx < 0:
                idx = 0
            _, num, den = self._meter_hints[idx]
            return num, den
        return self._default_meter

    def _bar_duration(self, bar_index: int, start_time: float, num: int, den: int) -> float:
        if den <= 0:
            raise ChordCsvError(f"{self._path}: meter denominator must be >0 (bar {bar_index})")
        if num <= 0:
            raise ChordCsvError(f"{self._path}: meter numerator must be >0 (bar {bar_index})")
        beats_per_bar = num * (4.0 / den)
        if beats_per_bar <= 0.0:
            raise ChordCsvError(f"{self._path}: meter {num}/{den} yields non-positive bar")
        if self._beat_len >= 2:
            start_beats = self._time_to_beat(start_time)
            end_time = self._beat_to_time(start_beats + beats_per_bar)
            duration = end_time - start_time
            if duration <= 0.0:
                duration = beats_per_bar * self._seconds_per_quarter
        else:
            duration = beats_per_bar * self._seconds_per_quarter
        return duration

    def ensure(self, target: int) -> None:
        if target < len(self._boundaries):
            return
        while len(self._boundaries) <= target:
            bar_index = len(self._boundaries) - 1
            start_time = self._boundaries[-1]
            num, den = self._meter_for(bar_index, start_time)
            bar_seconds = self._bar_duration(bar_index, start_time, num, den)
            if bar_seconds <= 0.0:
                raise ChordCsvError(
                    f"{self._path}: non-positive bar duration at bar {bar_index}"
                )
            self._boundaries.append(start_time + bar_seconds)

    def start_time(self, bar_index: int) -> float:
        if bar_index < 0:
            raise ChordCsvError(f"{self._path}: bar index must be >=0 (got {bar_index})")
        self.ensure(bar_index + 1)
        return self._boundaries[bar_index]

    def end_time(self, bar_index: int) -> float:
        if bar_index < 0:
            raise ChordCsvError(f"{self._path}: bar index must be >=0 (got {bar_index})")
        self.ensure(bar_index + 1)
        return self._boundaries[bar_index + 1]

    def meter_for(self, bar_index: int, start_time: float) -> Tuple[int, int]:
        return self._meter_for(bar_index, start_time)

    def uses_explicit_bar_times(self) -> bool:
        return self._explicit_boundaries

def read_chords_csv(
    path: Path,
    *,
    bar_times: Optional[List[float]] = None,
    beat_times: Optional[List[float]] = None,
    meter_map: Optional[List[Tuple[float, int, int]]] = None,
    meter_hints: Optional[List[Tuple[int, int, int]]] = None,
    bpm_hint: Optional[float] = None,
    default_meter: Tuple[int, int] = (4, 4),
    strict: bool = False,
    sections: Optional[Sequence[Mapping[str, Any]]] = None,
    song_end_bar: Optional[int] = None,
) -> List["ChordSpan"]:
    """Load chord spans from CSV supporting explicit and compact layouts.

    Supported schemas:

    * ``start,end,root,quality`` (or ``start,end,chord``) with explicit second offsets.
    * ``start,chord`` or ``start,root,quality`` rows (header optional) with implicit ends.
    * ``bar,chord`` rows (header optional) for per-bar chords.
    * ``bar_start,bar_end,chord`` and ``bar,beat,chord`` compact rows.

    Compact timelines prioritise provided ``bar_times`` (downbeats), then supplied
    ``beat_times`` (PrettyMIDI beat grid), followed by meter hints or the MIDI
    ``meter_map``. As a final fallback the ``bpm_hint`` with ``default_meter`` seeds a
    uniform grid. Errors raise :class:`ChordCsvError` with actionable messages. Set
    ``strict=True`` to require explicit spans to be strictly ascending.
    """

    raw_text = path.read_text(encoding="utf-8")
    if not raw_text.strip():
        return []
    raw_text = raw_text.replace("，", ",")
    raw_text = raw_text.lstrip("\ufeff")

    def _guard(
        func: Callable[..., List["ChordSpan"]], *args, **kwargs
    ) -> List["ChordSpan"]:
        try:
            return func(*args, **kwargs)
        except ChordCsvError:
            raise

    def _sections_end_cap(src: Optional[Sequence[Mapping[str, Any]]]) -> Optional[int]:
        if not src:
            return None
        best: Optional[int] = None
        for item in src:
            if not isinstance(item, Mapping):
                continue
            end_bar = item.get("end_bar")
            if end_bar is None:
                continue
            try:
                idx = int(end_bar)
            except Exception:
                continue
            if idx < 0:
                continue
            best = idx if best is None else max(best, idx)
        return best

    section_cap: Optional[int] = None
    sec_from_sections = _sections_end_cap(sections)
    if sec_from_sections is not None:
        section_cap = sec_from_sections
    if song_end_bar is not None:
        try:
            song_end_bar_int = int(song_end_bar)
        except Exception:
            song_end_bar_int = None
        else:
            if song_end_bar_int is not None and song_end_bar_int >= 0:
                section_cap = (
                    song_end_bar_int
                    if section_cap is None
                    else max(section_cap, song_end_bar_int)
                )

    reader = csv.reader(io.StringIO(raw_text))
    rows_with_line: List[Tuple[int, List[str]]] = [
        (line_no, list(row)) for line_no, row in enumerate(reader, start=1)
    ]
    if not rows_with_line:
        return []

    recognised = {
        "start",
        "end",
        "root",
        "quality",
        "chord",
        "bar",
        "beat",
        "bar_start",
        "bar_end",
    }

    header_cells: Optional[List[str]] = None
    header_lower: Optional[List[str]] = None
    header_line: Optional[int] = None
    data_index = 0
    for idx, (line_no, raw_row) in enumerate(rows_with_line):
        cells = [cell.strip() for cell in raw_row]
        if not any(cells):
            continue
        lower = [cell.strip().lower() for cell in raw_row]
        if lower:
            lower[0] = lower[0].lstrip("\ufeff")
        if any(token in recognised for token in lower if token):
            header_cells = cells
            header_lower = lower
            header_line = line_no
            data_index = idx + 1
            break
        data_index = idx
        break

    if data_index >= len(rows_with_line):
        return []

    def _safe_float(token: str, *, line: int, field: str) -> float:
        try:
            value = float(token)
        except Exception as exc:
            raise ChordCsvError(
                f"{path} line {line}: {field} '{token}' must be a number"
            ) from exc
        if not math.isfinite(value):
            raise ChordCsvError(f"{path} line {line}: {field} '{token}' is not finite")
        return value

    def _fallback_bar_seconds() -> float:
        num, den = default_meter
        den = max(1, int(den))
        num = max(1, int(num))
        beats_per_bar = num * (4.0 / den)
        bpm_val = bpm_hint if bpm_hint and math.isfinite(bpm_hint) and bpm_hint > 0 else 120.0
        seconds = beats_per_bar * (60.0 / bpm_val)
        if not math.isfinite(seconds) or seconds <= 0.0:
            return 4.0 * (60.0 / 120.0)
        return seconds

    def _parse_explicit(
        header_map: Dict[str, int],
        data_rows: Sequence[Tuple[int, List[str]]],
    ) -> List[ChordSpan]:
        start_idx = header_map["start"]
        end_idx = header_map["end"]
        chord_idx = header_map.get("chord")
        root_idx = header_map.get("root")
        qual_idx = header_map.get("quality")
        max_idx = max(idx for idx in (start_idx, end_idx, chord_idx or 0, root_idx or 0, qual_idx or 0))
        spans: List[ChordSpan] = []
        prev_end: Optional[float] = None
        for line_no, raw_row in data_rows:
            cells = [cell.strip() for cell in raw_row]
            if not any(cells):
                continue
            if len(cells) <= max_idx:
                cells.extend([""] * (max_idx + 1 - len(cells)))
            start_txt = cells[start_idx].lstrip("\ufeff")
            end_txt = cells[end_idx]
            start_val = _safe_float(start_txt, line=line_no, field="start")
            end_val = _safe_float(end_txt, line=line_no, field="end")
            if end_val <= start_val + EPS:
                raise ChordCsvError(
                    f"{path} line {line_no}: chord duration must be positive"
                )
            if strict and prev_end is not None and start_val < prev_end - EPS:
                raise ChordCsvError(
                    f"{path} line {line_no}: start {start_val} must be >= previous end {prev_end}"
                )
            prev_end = end_val

            parse_token: Optional[str] = None
            if chord_idx is not None and chord_idx < len(cells):
                raw_symbol = cells[chord_idx]
            else:
                if root_idx is None or qual_idx is None:
                    raise ChordCsvError(
                        f"{path} line {line_no}: chord columns missing (need 'chord' or 'root'+'quality')"
                    )
                root_txt = cells[root_idx]
                qual_txt = cells[qual_idx]
                if not root_txt or not qual_txt:
                    raise ChordCsvError(
                        f"{path} line {line_no}: root/quality cannot be empty"
                    )
                raw_symbol = f"{root_txt}:{qual_txt}"

            symbol_attr = raw_symbol.strip()
            parse_token = symbol_attr
            try:
                root_pc, quality = parse_chord_symbol(parse_token)
            except ValueError as exc:
                if ":" not in symbol_attr:
                    try:
                        parse_token = _normalize_compact_chord(symbol_attr)
                        root_pc, quality = parse_chord_symbol(parse_token)
                    except (ChordCsvError, ValueError) as inner_exc:
                        raise ChordCsvError(f"{path} line {line_no}: {inner_exc}") from inner_exc
                else:
                    raise ChordCsvError(f"{path} line {line_no}: {exc}") from exc

            span = ChordSpan(start_val, end_val, root_pc, quality)
            setattr(span, "symbol", symbol_attr)
            setattr(span, "root_name", _guess_root_display(symbol_attr, symbol_attr.split(":", 1)[0]))
            spans.append(span)
        return spans

    def _parse_start_rows(
        start_idx: int,
        symbol_fn: Callable[[List[str]], str],
        data_rows: Sequence[Tuple[int, List[str]]],
    ) -> List[ChordSpan]:
        entries: List[Tuple[int, float, str]] = []
        for line_no, raw_row in data_rows:
            cells = [cell.strip() for cell in raw_row]
            if not any(cells):
                continue
            if len(cells) <= start_idx:
                cells.extend([""] * (start_idx + 1 - len(cells)))
            start_txt = cells[start_idx].lstrip("\ufeff")
            start_val = _safe_float(start_txt, line=line_no, field="start")
            raw_symbol = symbol_fn(cells)
            if not raw_symbol:
                raise ChordCsvError(f"{path} line {line_no}: missing chord symbol")
            entries.append((line_no, start_val, raw_symbol))
        if not entries:
            return []
        entries.sort(key=lambda item: item[1])
        spans: List[ChordSpan] = []
        fallback = _fallback_bar_seconds()
        for idx, (line_no, start_val, raw_symbol) in enumerate(entries):
            end_val = entries[idx + 1][1] if idx + 1 < len(entries) else start_val + fallback
            if end_val <= start_val + EPS:
                raise ChordCsvError(
                    f"{path} line {line_no}: start times must be strictly increasing"
                )
            symbol_attr = raw_symbol.strip()
            parse_token = symbol_attr
            try:
                root_pc, quality = parse_chord_symbol(parse_token)
            except ValueError as exc:
                if ":" not in symbol_attr:
                    try:
                        parse_token = _normalize_compact_chord(symbol_attr)
                        root_pc, quality = parse_chord_symbol(parse_token)
                    except (ChordCsvError, ValueError) as inner_exc:
                        raise SystemExit(f"{path} line {line_no}: {inner_exc}") from inner_exc
                else:
                    raise SystemExit(f"{path} line {line_no}: {exc}") from exc
            span = ChordSpan(start_val, end_val, root_pc, quality)
            setattr(span, "symbol", symbol_attr)
            setattr(span, "root_name", _guess_root_display(symbol_attr, symbol_attr.split(":", 1)[0]))
            spans.append(span)
        return spans

    def _compact_to_spans(
        rows: List[_CompactChordRow], section_end_bar: Optional[int] = None
    ) -> List[ChordSpan]:
        if not rows:
            return []
        timeline = _BarTimeline(
            path=path,
            bar_times=bar_times,
            beat_times=beat_times,
            meter_map=meter_map,
            meter_hints=meter_hints,
            bpm_hint=bpm_hint,
            default_meter=default_meter,
        )

        starts: List[float] = []
        for row in rows:
            start_time = timeline.start_time(row.start_bar)
            num, den = timeline.meter_for(row.start_bar, start_time)
            bar_duration = timeline.end_time(row.start_bar) - start_time
            if row.start_beat:
                beat_val = float(row.start_beat)
                if beat_val < 0:
                    raise ChordCsvError(
                        f"{path} line {row.line_no}: beat must be >= 0 (got {beat_val})"
                    )
                if beat_val >= num:
                    raise ChordCsvError(
                        f"{path} line {row.line_no}: beat {beat_val} must be < meter numerator {num}"
                    )
                if bar_duration <= 0.0:
                    raise ChordCsvError(
                        f"{path} line {row.line_no}: bar duration is non-positive"
                    )
                start_time += (beat_val / float(num)) * bar_duration
            starts.append(start_time)

        spans: List[ChordSpan] = []
        cap_time: Optional[float] = None
        if section_end_bar is not None:
            try:
                cap_time = timeline.start_time(int(section_end_bar))
            except (ValueError, ChordCsvError) as exc:
                logging.warning("%s: invalid section end bar %s (%s)", path, section_end_bar, exc)
                cap_time = None
        for idx, row in enumerate(rows):
            start_time = starts[idx]
            end_time: float
            bar_start_abs = timeline.start_time(row.start_bar)
            bar_duration = timeline.end_time(row.start_bar) - bar_start_abs
            if row.end_bar is not None:
                end_time = timeline.start_time(row.end_bar)
            elif idx + 1 < len(starts):
                end_time = starts[idx + 1]
            else:
                end_time = timeline.end_time(row.start_bar)
                if row.start_beat is not None and timeline.uses_explicit_bar_times():
                    bar_start = timeline.start_time(row.start_bar)
                    offset = max(0.0, start_time - bar_start)
                    next_bar_start = timeline.start_time(row.start_bar + 1)
                    end_time = next_bar_start + offset
            if row.length_beats is not None and row.end_bar is None:
                try:
                    num, den = timeline.meter_for(row.start_bar, start_time)
                except ChordCsvError:
                    num, den = 4, 4
                beats_per_bar = float(num) if num else 1.0
                if beats_per_bar <= 0:
                    beats_per_bar = 1.0
                if bar_duration <= 0.0:
                    duration_candidate = 0.0
                else:
                    duration_candidate = float(row.length_beats) * (bar_duration / beats_per_bar)
                candidate_end = start_time + duration_candidate
                if candidate_end > start_time + EPS:
                    end_time = min(end_time, candidate_end)
                else:
                    logging.warning(
                        "%s line %d: beats duration produced non-positive span; skipping", path, row.line_no
                    )
                    continue
            if end_time <= start_time + EPS:
                raise ChordCsvError(
                    f"{path} line {row.line_no}: chord duration must be positive"
                )
            if cap_time is not None and end_time > cap_time + EPS:
                if start_time >= cap_time - EPS:
                    logging.warning(
                        "%s line %d: chord start beyond section end; skipping", path, row.line_no
                    )
                    continue
                logging.warning(
                    "%s line %d: chord truncated at section end", path, row.line_no
                )
                end_time = cap_time
            try:
                root_pc, quality = parse_chord_symbol(row.symbol)
            except ValueError as exc:
                raise ChordCsvError(
                    f"{path} line {row.line_no}: token {idx} ({row.symbol!r}) {exc}"
                ) from exc
            span = ChordSpan(start_time, end_time, root_pc, quality)
            display = row.raw_symbol or row.symbol
            setattr(span, "symbol", display)
            setattr(span, "root_name", _guess_root_display(display, display.split(":", 1)[0]))
            spans.append(span)
        return spans

    if header_lower:
        header_map = {name: idx for idx, name in enumerate(header_lower) if name}
        data_rows = rows_with_line[data_index:]
        if "start" in header_map and "end" in header_map:
            return _parse_explicit(header_map, data_rows)
        if "start" in header_map and ("chord" in header_map or {"root", "quality"} <= header_map.keys()):
            def _symbol_from_row(cells: List[str]) -> str:
                if "chord" in header_map:
                    return cells[header_map["chord"]]
                root_idx = header_map["root"]
                qual_idx = header_map["quality"]
                if root_idx >= len(cells) or qual_idx >= len(cells):
                    return ""
                root_txt = cells[root_idx]
                qual_txt = cells[qual_idx]
                return f"{root_txt}:{qual_txt}"

            return _parse_start_rows(header_map["start"], _symbol_from_row, data_rows)
        if "bar" in header_map or "bar_start" in header_map:
            compact_rows = _parse_bar_chord_rows(
                [row for _, row in data_rows],
                start_line=(header_line or 0) + 1,
                path=path,
                header=header_cells,
            )
            return _guard(_compact_to_spans, compact_rows, section_cap)
        raise ChordCsvError(
            f"{path}: unsupported chord CSV header {header_cells}"
        )

    first_line_no, first_row = rows_with_line[data_index]
    first_cells = [cell.strip() for cell in first_row]
    if not first_cells or len(first_cells) < 2:
        raise ChordCsvError(f"{path} line {first_line_no}: expected at least two columns")
    if re.fullmatch(r"^[+-]?\d+$", first_cells[0].lstrip("\ufeff")):
        compact_rows = _parse_bar_chord_rows(
            [row for _, row in rows_with_line[data_index:]],
            start_line=first_line_no,
            path=path,
        )
        return _guard(_compact_to_spans, compact_rows, section_cap)

    def _symbol_plain(cells: List[str]) -> str:
        if len(cells) < 2:
            return ""
        return cells[1]

    return _parse_start_rows(0, _symbol_plain, rows_with_line[data_index:])


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
    accent_map: Optional[Dict[str, List[float]]] = None,
    skip_phrase_in_rests: bool = False,
    silent_qualities: Optional[List[str]] = None,
    clone_meta_only: bool = False,
    stats: Optional[Dict] = None,
    merge_reset_at: str = "none",
    # extras from codex branch
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
    # extras from main branch
    guide_notes: Optional[Dict[int, int]] = None,
    guide_quant: str = "bar",
    guide_units: Optional[List[Tuple[float, float]]] = None,
    rest_silence_hold_off: bool = False,
    phrase_change_lead_beats: float = 0.0,
    phrase_pool: Optional[Dict[str, Any]] = None,
    phrase_pick: str = "roundrobin",
    no_repeat_window: int = 1,
    rest_silence_send_stop: bool = False,
    stop_min_gap_beats: float = 0.0,
    stop_velocity: int = 64,
    section_profiles: Optional[Dict[str, Dict]] = None,
    sections: Optional[Sequence[Any]] = None,
    section_default: str = "verse",
    section_verbose: bool = False,
    style_layer_mode: str = "off",
    style_layer_every: int = 4,
    style_layer_len_beats: float = 0.5,
    style_phrase_pool: Optional[Dict[str, Any]] = None,
    trend_window: int = 0,
    trend_th: float = 0.0,
    quantize_strength: Union[float, List[float]] = 0.0,
    rng_pool: Optional[random.Random] = None,
    rng_human: Optional[random.Random] = None,
    write_markers: bool = False,
    marker_encoding: str = "raw",
    onset_list: Optional[List[int]] = None,
    rest_list: Optional[List[float]] = None,
    density_rules: Optional[List[Dict[str, Any]]] = None,
    swing_shape: str = "offbeat",
) -> "pretty_midi.PrettyMIDI":
    """Render Sparkle-compatible MIDI with optional statistics payload.

    When ``stats`` is supplied a schema version of ``1.1`` is recorded alongside
    ``bar_pulse_grid`` (meter-derived reference grid), ``bar_pulses`` (legacy
    alias containing the same ``[(rel_beat, time), ...]`` tuples), and
    ``bar_triggers`` (actual emitted trigger pulses, also mirrored to
    ``bar_trigger_pulses``/``bar_trigger_pulses_compat``).
    """
    rng = rng_human or rng
    if rng is None:
        rng = random.Random(0) if _SPARKLE_DETERMINISTIC else random.Random()

    def _merge_end_hint(
        current: Optional[float], candidate: Optional[float]
    ) -> Optional[float]:
        if candidate is None:
            return current
        if current is None:
            return candidate
        return max(current, candidate)

    def _trim_downbeat_grid(grid: List[float], end_time: Optional[float]) -> List[float]:
        if not grid or end_time is None or not math.isfinite(end_time):
            return grid
        target = max(end_time, grid[0])
        trimmed: List[float] = []
        for t in grid:
            if t <= target + EPS:
                trimmed.append(t)
            else:
                break
        if not trimmed:
            trimmed = [grid[0]]
        if trimmed[-1] < target - EPS:
            trimmed.append(target)
        else:
            trimmed[-1] = target
        return trimmed

    song_end_hint: Optional[float] = None
    try:
        pm_input_end = float(pm_in.get_end_time())
    except Exception:
        pm_input_end = None
    if chords:
        chord_end = max((c.end for c in chords), default=None)
        song_end_hint = _merge_end_hint(song_end_hint, chord_end)

    def _duck(bar_idx: int, vel: int) -> int:
        if vocal_ducking > 0 and vocal_adapt and vocal_adapt.dense_phrase is not None:
            if vocal_adapt.phrase_for_bar(bar_idx) == vocal_adapt.dense_phrase:
                return max(1, int(round(vel * (1.0 - vocal_ducking))))
        return vel

    def _copy_time_signatures_meta(
        src_pm: "pretty_midi.PrettyMIDI", dest_pm: "pretty_midi.PrettyMIDI"
    ) -> None:
        ts_src = getattr(src_pm, "time_signature_changes", []) or []
        dest_pm.time_signature_changes = []
        ts_cls = getattr(pretty_midi, "TimeSignature", None)
        for ts in ts_src:
            clone = None
            if ts_cls is not None:
                try:
                    clone = ts_cls(ts.numerator, ts.denominator, ts.time)
                except Exception:
                    clone = None
            if clone is None:
                try:
                    clone = ts.__class__(ts.numerator, ts.denominator, ts.time)
                except Exception:
                    clone = types.SimpleNamespace(
                        numerator=getattr(ts, "numerator", 4),
                        denominator=getattr(ts, "denominator", 4),
                        time=getattr(ts, "time", 0.0),
                    )
            dest_pm.time_signature_changes.append(clone)

    def _copy_tempi_meta(
        src_pm: "pretty_midi.PrettyMIDI", dest_pm: "pretty_midi.PrettyMIDI"
    ) -> str:
        used_private = False
        try:
            times, tempos = src_pm.get_tempo_changes()
        except Exception:
            times, tempos = [], []

        def _safe_list(seq: Any) -> List[Any]:
            if seq is None:
                return []
            if isinstance(seq, list):
                return list(seq)
            tolist = getattr(seq, "tolist", None)
            if callable(tolist):
                try:
                    return list(tolist())
                except Exception:
                    pass
            try:
                return list(seq)
            except Exception:
                return [seq]

        time_list = _safe_list(times)
        tempo_list = _safe_list(tempos)
        if hasattr(dest_pm, "_tempo_changes"):
            dest_pm._tempo_changes = []  # type: ignore[attr-defined]
        if hasattr(dest_pm, "_tick_scales"):
            dest_pm._tick_scales = []  # type: ignore[attr-defined]
        if hasattr(dest_pm, "_tick_to_time"):
            dest_pm._tick_to_time = []  # type: ignore[attr-defined]
        if hasattr(dest_pm, "_add_tempo_change"):
            for t, tempo in zip(time_list, tempo_list):
                try:
                    dest_pm._add_tempo_change(tempo, t)  # type: ignore[attr-defined]
                except Exception:
                    pass
        elif tempo_list:
            dest_pm.initial_tempo = tempo_list[0]

        tempo_changes = getattr(src_pm, "_tempo_changes", None)
        if tempo_changes is not None and hasattr(dest_pm, "_tempo_changes"):
            used_private = True
            normalized: List[Tuple[float, float]] = []
            for tc in tempo_changes:
                tempo_val: Optional[float] = None
                time_val: Optional[float] = None
                if hasattr(tc, "tempo") and hasattr(tc, "time"):
                    tempo_val = getattr(tc, "tempo", None)
                    time_val = getattr(tc, "time", None)
                elif isinstance(tc, (list, tuple)) and len(tc) >= 2:
                    tempo_val = tc[0]
                    time_val = tc[1]
                if tempo_val is None or time_val is None:
                    continue
                try:
                    tempo_val = float(tempo_val)
                except Exception:
                    tempo_val = float("nan")
                try:
                    time_val = float(time_val)
                except Exception:
                    time_val = 0.0
                normalized.append((tempo_val, time_val))
            try:
                tempo_cls = pretty_midi.containers.TempoChange  # type: ignore[attr-defined]
            except Exception:
                tempo_cls = None
            if tempo_cls is not None:
                dest_pm._tempo_changes = [  # type: ignore[attr-defined]
                    tempo_cls(tempo, time) for tempo, time in normalized
                ]
            else:
                dest_pm._tempo_changes = [  # type: ignore[attr-defined]
                    (tempo, time) for tempo, time in normalized
                ]

        tick_scales = getattr(src_pm, "_tick_scales", None)
        if tick_scales is not None and hasattr(dest_pm, "_tick_scales"):
            used_private = True
            dest_pm._tick_scales = list(tick_scales)  # type: ignore[attr-defined]

        tick_to_time = getattr(src_pm, "_tick_to_time", None)
        if tick_to_time is not None and hasattr(dest_pm, "_tick_to_time"):
            used_private = True
            dest_pm._tick_to_time = list(tick_to_time)  # type: ignore[attr-defined]

        if used_private:
            # Normalise tempo caches after copying private metadata so later tick
            # reseeding works from a consistent baseline.
            _sanitize_tempi(dest_pm)

        return "private" if used_private else "public"

    def _new_pretty_midi_with_meta(
        src_pm: "pretty_midi.PrettyMIDI",
    ) -> Tuple["pretty_midi.PrettyMIDI", str]:
        try:
            dest_pm = pretty_midi.PrettyMIDI()
        except TypeError:
            dest_pm = pretty_midi.PrettyMIDI(None)
        _copy_time_signatures_meta(src_pm, dest_pm)
        # _copy_tempi_meta sanitises tempo caches before we reseed tick tables with
        # _ensure_tempo_and_ticks, keeping legacy PrettyMIDI internals in sync.
        meta_kind = _copy_tempi_meta(src_pm, dest_pm)
        return dest_pm, meta_kind

    if clone_meta_only:
        # Deep copies of PrettyMIDI were removed for memory savings; rebuild metadata instead.
        out, meta_src = _new_pretty_midi_with_meta(pm_in)
    else:
        out, meta_src = _new_pretty_midi_with_meta(pm_in)

    seed_bpm = float(bpm) if bpm is not None else 120.0
    if not math.isfinite(seed_bpm) or seed_bpm <= 0.0:
        seed_bpm = 120.0
    _ensure_tempo_and_ticks(out, seed_bpm, out.time_signature_changes)
    meta_private = getattr(out, "_sparkle_meta_seed_fallback", False)
    if stats is not None:
        setattr(out, "_sparkle_stats", stats)
        if meta_private:
            stats["meta_seeded"] = "private_fallback"

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
    if merge_reset_at == "none" and phrase_hold in ("bar", "chord"):
        merge_reset_at = phrase_hold
    chord_range = mapping.get("chord_input_range")
    voicing_mode = mapping.get("voicing_mode", "stacked")
    top_note_max = mapping.get("top_note_max")
    strict = bool(mapping.get("strict", False))

    # --- Beat grid with robust fallbacks ---
    try:
        beat_times = list(pm_in.get_beats())
    except Exception:
        beat_times = []
    if len(beat_times) < 2:
        # try beats from the output object where we already seeded tempo/ticks
        try:
            beat_times = list(out.get_beats())
        except Exception:
            beat_times = []
    if len(beat_times) < 2:
        # final fallback: seed a constant grid from BPM up to best end hint
        target_end = (
            song_end_hint
            if (song_end_hint is not None and math.isfinite(song_end_hint))
            else pm_input_end
        )
        if (
            target_end is None
            or not math.isfinite(target_end)
            or target_end <= 0.0
        ):
            # default to 4 bars of 4/4 to stay conservative
            target_end = 16 * (60.0 / seed_bpm)
        step = 60.0 / seed_bpm
        t = 0.0
        beat_times = [0.0]
        while t + step <= target_end + EPS:
            t += step
            beat_times.append(t)
        logging.warning(
            "build_sparkle_midi: no beats found; seeded constant grid (bpm=%s, end=%.3fs)",
            seed_bpm,
            target_end,
        )
    if len(beat_times) < 2:
        raise SystemExit("Could not determine beats from MIDI")
    if stats is not None:
        stats["beat_times"] = [float(bt) for bt in beat_times]

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

    if guide_units:
        guide_end = max((beat_to_time(end) for _, end in guide_units), default=None)
        song_end_hint = _merge_end_hint(song_end_hint, guide_end)
    if song_end_hint is None:
        song_end_hint = pm_input_end

    unit_starts: List[float] = [u[0] for u in guide_units] if guide_units else []

    def maybe_merge_gap(inst, pitch, start_t, *, bar_start=None, chord_start=None):
        """Return merge gap or -1.0 to force new note at reset boundary."""
        mg = (
            phrase_merge_gap
            if phrase_hold != "off" or merge_reset_at != "none"
            else -1.0
        )
        if mg >= 0 and inst.notes and inst.notes[-1].pitch == pitch:
            gap = start_t - inst.notes[-1].end
            if (
                merge_reset_at != "none"
                and gap <= phrase_merge_gap + EPS
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
    try:
        raw_downbeats = list(pm_in.get_downbeats())
    except Exception:
        raw_downbeats = []
    meter_map: List[Tuple[float, int, int]] = []
    estimated_4_4 = False
    if ts_changes:
        for ts in ts_changes:
            meter_map.append((ts.time, ts.numerator, ts.denominator))
    else:
        meter_map.append((0.0, 4, 4))
        estimated_4_4 = True
    if len(meter_map) > 1:
        meter_map.sort(key=lambda x: x[0])
    downbeats = resolve_downbeats(
        pm_in,
        meter_map,
        beat_times,
        beat_to_time,
        time_to_beat,
        allow_meter_mismatch=estimated_4_4,
    )
    downbeats = _trim_downbeat_grid(downbeats, song_end_hint)
    meter_times = [mt for mt, _, _ in meter_map]
    if cycle_notes and (len(downbeats) - 1) < 2 and cycle_mode == "bar":
        logging.info("cycle disabled; using fixed phrase_note=%d", phrase_note)
        cycle_notes = []
        if stats is not None:
            stats["cycle_disabled"] = True

    if stats is not None:
        stats.setdefault("warnings", [])
        stats["schema_version"] = "1.1"
        stats["schema"] = "1.1"
        stats["downbeats"] = list(downbeats)
        if raw_downbeats:
            stats["_legacy_downbeats_raw"] = [float(db) for db in raw_downbeats]
        if song_end_hint is not None and math.isfinite(song_end_hint):
            stats["song_end_hint"] = float(song_end_hint)
        if downbeats:
            stats["downbeats_last"] = float(downbeats[-1])
        # Dict[bar_index, List[(bar_relative_beats, absolute_time_seconds)]]
        # capturing the meter-derived reference grid per bar. Stored as floats to
        # ease JSON export without further casting. ``bar_triggers`` records the
        # actual trigger placements when phrases are emitted so analytics can
        # distinguish between the theoretical grid and realised pulses.

        bar_triggers_obj = stats.get("bar_triggers")
        if isinstance(bar_triggers_obj, dict):
            bar_triggers_obj.clear()
        else:
            bar_triggers_obj = {}
        stats["bar_triggers"] = bar_triggers_obj
        stats["bar_trigger_pulses"] = bar_triggers_obj
        stats["bar_trigger_pulses_compat"] = bar_triggers_obj

        bar_pulse_grid_obj = stats.get("bar_pulse_grid")
        if isinstance(bar_pulse_grid_obj, dict):
            bar_pulse_grid_obj.clear()
        else:
            bar_pulse_grid_obj = {}
        stats["bar_pulse_grid"] = bar_pulse_grid_obj

        bar_pulses_obj = stats.get("bar_pulses")
        if isinstance(bar_pulses_obj, dict):
            bar_pulses_obj.clear()
        else:
            bar_pulses_obj = {}
        stats["bar_pulses"] = bar_pulses_obj
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
    if any(den == 8 and num % 3 == 0 for _, num, den in meter_map):
        if not math.isclose(swing_unit_beats, 1 / 12, abs_tol=EPS):
            logging.info("suggest --swing-unit 1/12 for ternary feel")

    num_bars = len(downbeats) - 1
    section_labels_override: Optional[List[str]] = None
    if sections:
        normalized_sections_cli, labels_override = normalize_sections(
            sections,
            bar_count=num_bars,
            default_tag=section_default,
            stats=stats,
        )
        sections = normalized_sections_cli
        section_labels_override = labels_override
    density_map: Optional[Dict[int, str]] = {} if stats is not None else None
    sections_map = mapping.get("sections")
    if sections_map:
        tag_map: Optional[Dict[int, str]] = {} if stats is not None else None
        for sec in sections_map:
            dens = sec.get("density")
            start = int(sec.get("start_bar", 0))
            end = int(sec.get("end_bar", num_bars))
            if dens in ("low", "med", "high"):
                for b in range(max(0, start), min(num_bars, end)):
                    if density_map is not None:
                        density_map[b] = dens
            tag = sec.get("tag")
            if tag:
                for b in range(max(0, start), min(num_bars, end)):
                    if tag_map is not None:
                        tag_map[b] = tag
        if stats is not None and tag_map is not None:
            stats["section_tags"] = tag_map
    if stats is not None:
        stats["bar_density"] = density_map or {}
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
        if bi in accent_by_bar:
            return accent_by_bar[bi]
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

    # From add-guide-midi-phrase-selection-and-damping branch
    accent_by_bar: Dict[int, List[float]] = {}
    accent_scale_by_bar: Dict[int, float] = {}
    if accent_map:
        for i, t in enumerate(downbeats):
            num, den = get_meter_at(meter_map, t, times=meter_times)
            key = f"{num}/{den}"
            lst = accent_map.get(key)
            if lst:
                accent_by_bar[i] = lst

    damp_scale_by_bar: Dict[int, Tuple[int, int]] = {}
    bar_pool_pickers: Dict[int, PoolPicker] = {}
    section_labels: List[str] = []
    if sections:
        section_labels = section_labels_override or [section_default] * len(downbeats)
        if section_labels:
            section_labels = section_labels[: len(downbeats)] + [
                section_default
            ] * max(0, len(downbeats) - len(section_labels))
        else:
            section_labels = [section_default] * len(downbeats)
    elif stats is not None:
        labels_from_stats = stats.get("section_labels") or stats.get("sections")
        if isinstance(labels_from_stats, list):
            section_labels = list(labels_from_stats)
        else:
            section_labels = [section_default] * len(downbeats)
    else:
        section_labels = [section_default] * len(downbeats)
    if section_profiles:
        for i, tag in enumerate(section_labels):
            prof = section_profiles.get(tag)
            if not prof:
                continue
            if "accent" in prof:
                accent_by_bar[i] = prof["accent"]
            if "accent_scale" in prof:
                try:
                    accent_scale_by_bar[i] = float(prof["accent_scale"])
                except Exception:
                    pass
            if "damp_scale" in prof:
                ds = prof["damp_scale"]
                if isinstance(ds, list) and len(ds) == 2:
                    damp_scale_by_bar[i] = (int(ds[0]), int(ds[1]))
            if "phrase_pool" in prof:
                notes = prof.get("phrase_pool", {}).get("notes", [])
                weights = prof.get("phrase_pool", {}).get("weights", [1] * len(notes))
                pool = []
                for n, w in zip(notes, weights):
                    nt = parse_note_token(n)
                    if nt is not None:
                        pool.append((nt, float(w)))
                if pool:
                    bar_pool_pickers[i] = PoolPicker(pool, phrase_pick, rng=rng_pool)
            if "phrase_pick" in prof:
                bar_pool_pickers[i] = PoolPicker(
                    bar_pool_pickers[i].pool if i in bar_pool_pickers else [],
                    prof["phrase_pick"],
                    rng=rng_pool,
                )
            if prof.get("no_immediate_repeat"):
                no_repeat_window = max(no_repeat_window, 1)
    density_override: Dict[int, int] = {}
    if density_rules is None:
        density_rules = [
            {"rest_ratio": 0.5, "note": 24},
            {"onset_count": 3, "note": 36},
        ]
    if rest_list is not None and onset_list is not None:
        for i, (r, o) in enumerate(zip(rest_list, onset_list)):
            for rule in density_rules:
                note = None
                if "rest_ratio" in rule and r >= rule["rest_ratio"]:
                    note = parse_note_token(rule["note"])
                elif "onset_count" in rule and o >= rule["onset_count"]:
                    note = parse_note_token(rule["note"])
                if note is not None:
                    density_override[i] = note
                    break
    if section_verbose and section_labels:
        logging.info("sections: %s", section_labels)

    # From main branch: phrase scheduling support
    # Build phrase plan and fill map (supports both 2- and 3-value returns)
    phrase_plan: List[Optional[int]] = []
    fill_map: Dict[int, int] = {}
    fill_src: Dict[int, str] = {}
    plan_active = bool(cycle_notes)
    if cycle_mode == "bar":
        sec_list = mapping.get("sections")
        # ensure num_bars covers any chord tail that may extend last bar
        if chords:
            last_idx = max(max(0, bisect.bisect_right(downbeats, c.end - EPS) - 1) for c in chords)
            num_bars = max(num_bars, last_idx + 1)
            if len(bar_qualities) < num_bars:
                bar_qualities.extend([None] * (num_bars - len(bar_qualities)))
        style_inject = mapping.get("style_inject")
        if not (fill_policy == "style" or (lfo_targets and "fill" in lfo_targets)):
            style_inject = None
        res = schedule_phrase_keys(
            num_bars,
            cycle_notes,
            sec_list,
            mapping.get("style_fill"),
            cycle_start_bar=cycle_start_bar,
            cycle_stride=cycle_stride,
            lfo=section_lfo,
            style_inject=style_inject,
            fill_policy=fill_policy,
            pulse_subdiv=pulse_subdiv_beats,
            markov=mapping.get("markov"),
            bar_qualities=bar_qualities,
            section_pool_weights=section_pool_weights,
            rng=rng,
            stats=stats,
        )
        if isinstance(res, tuple) and len(res) == 3:
            phrase_plan, fill_map, fill_src = res  # type: ignore
        else:
            phrase_plan, fill_map = res  # type: ignore
            fill_src = {}
        if sec_list or mapping.get("style_fill") is not None or mapping.get("markov") is not None:
            if any(p is not None for p in phrase_plan):
                plan_active = True
        bar_sources: Optional[List[str]] = ["cycle"] * len(phrase_plan) if stats is not None else None
        if stats is not None:
            stats["bar_phrase_notes_list"] = list(phrase_plan)
            stats["fill_bars"] = list(fill_map.keys())
        if sec_list and bar_sources is not None:
            for sec in sec_list:
                pool = sec.get("pool")
                if pool:
                    start = int(sec.get("start_bar", 0))
                    end = int(sec.get("end_bar", num_bars))
                    for b in range(max(0, start), min(num_bars, end)):
                        bar_sources[b] = "section"
        if vocal_adapt and phrase_plan and bar_sources is not None:
            for i in range(len(phrase_plan)):
                alt = vocal_adapt.phrase_for_bar(i)
                if alt is not None:
                    phrase_plan[i] = alt
                    bar_sources[i] = (
                        f"{bar_sources[i]}+vocal" if bar_sources[i] != "vocal" else "vocal"
                    )
        if guide_onsets and guide_style_note is not None:
            for idx, cnt in enumerate(guide_onsets):
                if cnt >= guide_onset_th:
                    tgt = idx - 1
                    if tgt >= 0 and tgt not in fill_map:
                        fill_map[tgt] = (guide_style_note, pulse_subdiv_beats, 1.0)  # type: ignore
                        fill_src[tgt] = "style"
                        if stats is not None:
                            stats["fill_bars"].append(tgt)
                    break
        if stats is not None and bar_sources is not None:
            labels_for_reason = section_labels if section_labels else []
            for i, pn in enumerate(phrase_plan):
                src = bar_sources[i]
                if labels_for_reason:
                    tag = labels_for_reason[i] if i < len(labels_for_reason) else labels_for_reason[-1]
                    if tag:
                        prefix = f"section:{tag}"
                        src = f"{prefix}|{src}" if src else prefix
                stats["bar_reason"][i] = {"source": src, "note": pn}
            stats["fill_sources"].update(fill_src)

    # Precompute pulse timestamps per bar (meter-derived grid; swing shifts timing only)
    if stats is not None:
        bar_grid = stats.setdefault("bar_pulse_grid", {})
        bar_triggers = stats.setdefault("bar_triggers", {})
        stats["bar_trigger_pulses"] = bar_triggers
        stats["bar_trigger_pulses_compat"] = bar_triggers
        bar_pulses_dict = stats.setdefault("bar_pulses", {})
        bar_grid.clear()
        bar_triggers.clear()
        bar_pulses_dict.clear()
        # ``bar_pulses`` intentionally mirrors ``bar_pulse_grid`` so legacy
        # callers that still read "pulses" receive the metre-derived grid
        # rather than phrase note pitches.
        for i, (start, next_start) in enumerate(zip(downbeats[:-1], downbeats[1:])):
            num, den = get_meter_at(meter_map, start, times=meter_times)
            sb = time_to_beat(start)
            eb = time_to_beat(next_start)
            bar_len_beats = _beats_per_bar(num, den) if den else 0.0
            wants_legacy = bool(stats and stats.get("_legacy_bar_pulses_grid"))
            legacy_total = _legacy_pulses_per_bar(num, den, stats) if wants_legacy else None
            legacy_override = bool(estimated_4_4 and legacy_total and den == 4)
            if legacy_override:
                beats_est = float(eb - sb)
                inferred = int(round(beats_est * 4.0)) if math.isfinite(beats_est) else 0
                total = max(1, inferred if inferred > 0 else int(legacy_total))
            else:
                step_beats = pulse_subdiv_beats if pulse_subdiv_beats > 0.0 else (bar_len_beats or 1.0)
                total = int(round(bar_len_beats / step_beats)) if step_beats > 0.0 else 1
                total = max(1, total)
            if bar_len_beats <= 0.0 or total <= 0 or eb <= sb:
                pulse = (0.0, float(start))
                grid_list = [pulse]
                bar_grid[i] = grid_list
                bar_pulses_dict[i] = list(grid_list)
                continue
            if legacy_override:
                duration = max(EPS, float(next_start - start))
                rel_step = bar_len_beats / total if bar_len_beats > 0.0 else 0.0
                grid_list: List[Tuple[float, float]] = []
                if rel_step <= 0.0:
                    grid_list = [(0.0, float(start))]
                else:
                    for k in range(total):
                        rel_beat = k * rel_step
                        frac = rel_beat / bar_len_beats if bar_len_beats else 0.0
                        # 等分を優先し、最終インパルスは常にバー終端に配置する
                        if k == total - 1:
                            rel_beat = bar_len_beats
                            time_val = float(next_start)
                        else:
                            time_val = float(start + frac * duration)
                        grid_list.append((float(rel_beat), time_val))
                seen_rel: set[float] = set()
                deduped: List[Tuple[float, float]] = []
                for rel, t_val in grid_list:
                    rel_key = round(rel, 9)
                    if rel_key in seen_rel:
                        continue
                    seen_rel.add(rel_key)
                    deduped.append((rel, t_val))
                while len(deduped) < total:
                    rel = deduped[-1][0] + rel_step
                    deduped.append((rel, float(start + (rel / bar_len_beats) * duration)))
                grid_list = deduped[:total]
                if (
                    swing > 0.0
                    and swing_shape in {"offbeat", "even"}
                    and math.isclose(pulse_subdiv_beats, swing_unit_beats, abs_tol=EPS)
                ):
                    half = swing * pulse_subdiv_beats * 0.5
                    adjusted: List[Tuple[float, float]] = []
                    for idx, (rel, time_val) in enumerate(grid_list):
                        beat_val = rel
                        if swing_shape == "offbeat":
                            if idx % 2 == 1:
                                beat_val = min(rel + swing * pulse_subdiv_beats, bar_len_beats - EPS)
                        else:  # even
                            if idx % 2 == 1:
                                beat_val = min(rel + half, bar_len_beats - EPS)
                            elif idx > 0:
                                beat_val = max(rel - half, 0.0)
                                if adjusted and beat_to_time(sb + beat_val) <= adjusted[-1][1] + EPS:
                                    beat_val = max(
                                        0.0,
                                        time_to_beat(adjusted[-1][1] + EPS) - sb,
                                    )
                        beat_val_abs = clip_to_bar(sb + beat_val, sb, sb + bar_len_beats)
                        beat_val = beat_val_abs - sb
                        adjusted.append((beat_val, beat_to_time(beat_val_abs)))
                    grid_list = adjusted
                bar_grid[i] = list(grid_list)
                bar_pulses_dict[i] = list(grid_list)
                continue

            grid: List[Tuple[float, float]] = []
            beat_val = sb
            for k in range(total):
                if k > 0:
                    interval = step_beats
                    if swing > 0.0 and math.isclose(pulse_subdiv_beats, swing_unit_beats, abs_tol=EPS):
                        if swing_shape == "offbeat":
                            interval *= (1 + swing) if k % 2 == 0 else (1 - swing)
                        elif swing_shape == "even":
                            interval *= (1 - swing) if k % 2 == 0 else (1 + swing)
                        else:
                            mod = k % 3
                            if mod == 0:
                                interval *= 1 + swing
                            elif mod == 1:
                                interval *= 1 - swing
                    interval = max(interval, EPS)
                    beat_val += interval
                if beat_val > eb + EPS:
                    break
                if beat_val >= sb + bar_len_beats - EPS:
                    beat_val = sb + bar_len_beats - EPS
                time_val = beat_to_time(beat_val)
                if time_val >= next_start - EPS:
                    break
                rel_beat = float(beat_val - sb)
                grid.append((rel_beat, float(time_val)))
            if not grid:
                grid = [(0.0, float(start))]
            grid.sort(key=lambda item: item[1])
            grid_list = list(grid)
            if (
                swing > 0.0
                and swing_shape == "even"
                and math.isclose(pulse_subdiv_beats, swing_unit_beats, abs_tol=EPS)
                and len(grid_list) >= 2
            ):
                half = swing * pulse_subdiv_beats * 0.5
                adjusted: List[Tuple[float, float]] = []
                for idx, (rel, time_val) in enumerate(grid_list):
                    if idx % 2 == 1:
                        time_val = beat_to_time(sb + rel + half)
                    elif idx > 0:
                        time_val = beat_to_time(max(sb, rel - half))
                        if adjusted and time_val <= adjusted[-1][1] + EPS:
                            time_val = adjusted[-1][1] + EPS
                    adjusted.append((rel, float(time_val)))
                grid_list = adjusted
            if (
                not legacy_override
                and legacy_total
                and estimated_4_4
                and legacy_total > len(grid_list)
            ):
                # legacy mode disabled but caller asked for compatibility; fall back to
                # the previous interpolation logic to avoid dropping pulses.
                extras_sorted: List[Tuple[float, float]] = []
                base = list(grid_list)
                extras_needed = legacy_total - len(base)
                extras: List[Tuple[float, float]] = []
                if len(base) == 1:
                    span_rel = max(0.0, time_to_beat(next_start) - sb)
                    if extras_needed > 0 and span_rel > 0.0:
                        for j in range(extras_needed):
                            frac = (j + 1) / (extras_needed + 1)
                            rel_beat = base[0][0] + frac * span_rel
                            time_val = base[0][1] + frac * (next_start - base[0][1])
                            extras.append((float(rel_beat), float(time_val)))
                else:
                    intervals = len(base) - 1
                    if intervals == 1:
                        per = extras_needed
                        rem = 0
                    else:
                        slots = max(1, intervals - 1)
                        per = extras_needed // slots
                        rem = extras_needed % slots
                    for idx in range(intervals):
                        cur = base[idx]
                        nxt = base[idx + 1]
                        extras_here = 0
                        if intervals == 1:
                            extras_here = per
                        else:
                            if idx == 0:
                                extras_here = 0
                            else:
                                extras_here = per
                                if idx - 1 < rem:
                                    extras_here += 1
                        if extras_here > 0:
                            rel_start, time_start = cur
                            rel_end, time_end = nxt
                            for j in range(1, extras_here + 1):
                                frac = j / (extras_here + 1)
                                rel_beat = rel_start + frac * (rel_end - rel_start)
                                time_val = time_start + frac * (time_end - time_start)
                                extras.append((float(rel_beat), float(time_val)))
                seen = {(b[0], b[1]) for b in base}
                for item in sorted(extras, key=lambda it: it[1]):
                    key = (item[0], item[1])
                    if key in seen:
                        continue
                    extras_sorted.append(item)
                    seen.add(key)
                    if len(extras_sorted) >= legacy_total - len(base):
                        break
                grid_list = base + extras_sorted
                if __debug__ and len(extras_sorted) == 0 and len(grid_list) > 1:
                    assert all(
                        grid_list[idx][1] <= grid_list[idx + 1][1]
                        for idx in range(len(grid_list) - 1)
                    ), "bar pulse times must be monotonic"
            elif __debug__ and len(grid_list) > 1:
                assert all(
                    grid_list[idx][1] <= grid_list[idx + 1][1]
                    for idx in range(len(grid_list) - 1)
                ), "bar pulse times must be monotonic"
            bar_grid[i] = grid_list
            bar_pulses_dict[i] = list(grid_list)

    # Velocity curve helper
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

    # --- Unified phrase note picking (guide → density → cycle/plan → pool) ---
    last_guided: Optional[int] = None
    prev_hold: Optional[int] = None

    pool_picker: Optional[PoolPicker] = None
    if phrase_pool:
        if isinstance(phrase_pool, list):
            phrase_pool = {"pool": phrase_pool}
        if phrase_pool.get("pool"):
            pool_picker = PoolPicker(
                phrase_pool["pool"],
                phrase_pick,
                T=phrase_pool.get("T"),
                no_repeat_window=no_repeat_window,
                rng=rng_pool,
            )

    trend_labels: List[int] = []
    if onset_list is not None:
        trend_labels = [0] * len(onset_list)
        if trend_window > 0 and len(onset_list) > trend_window:
            for i in range(trend_window, len(onset_list)):
                prev = sum(onset_list[i - trend_window : i]) / trend_window
                curr = sum(onset_list[i - trend_window + 1 : i + 1]) / trend_window
                slope = curr - prev
                if slope > trend_th:
                    trend_labels[i] = 1
                elif slope < -trend_th:
                    trend_labels[i] = -1

    def pulses_in_range(start_t: float, end_t: float) -> List[Tuple[int, int]]:
        indices: List[Tuple[int, int]] = []
        b = time_to_beat(start_t)
        eb = time_to_beat(end_t)
        iter_guard = MAX_ITERS
        # Guard to avoid pathological zero-interval loops when swing math stalls.
        while b < eb - EPS:
            t = beat_to_time(b)
            bar_idx = max(0, bisect.bisect_right(downbeats, t) - 1)
            bar_start_b = time_to_beat(downbeats[bar_idx])
            idx = int(math.floor((b - bar_start_b + EPS) / pulse_subdiv_beats))
            indices.append((bar_idx, idx))
            interval = pulse_subdiv_beats
            # Trigger swing adjusts stride spacing for emitted notes, complementing the
            # earlier grid swing so both analytics and playback share the same feel.
            if swing > 0.0 and math.isclose(pulse_subdiv_beats, swing_unit_beats, abs_tol=EPS):
                if swing_shape == "offbeat":
                    interval *= (1 + swing) if idx % 2 == 0 else (1 - swing)
                elif swing_shape == "even":
                    interval *= (1 - swing) if idx % 2 == 0 else (1 + swing)
                else:
                    mod = idx % 3
                    if mod == 0:
                        interval *= 1 + swing
                    elif mod == 1:
                        interval *= 1 - swing
            if interval <= EPS:
                logging.warning(
                    "pulses_in_range: non-positive interval; aborting pulse walk"
                )
                break
            b += interval
            iter_guard -= 1
            if iter_guard <= 0:
                logging.warning(
                    "pulses_in_range: max iterations reached; aborting pulse walk"
                )
                break
        return indices

    # placeholder removed; phrase emission handled below
    # --- merged: accents / section profiles / density overrides / phrase plan ---
    # Accents from meter and explicit accent_map
    accent_by_bar: Dict[int, List[float]] = {}
    accent_scale_by_bar: Dict[int, float] = {}
    if accent_map:

        def meter_at(t: float) -> Tuple[int, int]:
            idx = 0
            for j, (mt, num, den) in enumerate(meter_map):
                if mt <= t:
                    idx = j
                else:
                    break
            return meter_map[idx][1], meter_map[idx][2]

        for i, t in enumerate(downbeats):
            num, den = meter_at(t)
            key = f"{num}/{den}"
            lst = accent_map.get(key)
            if lst:
                accent_by_bar[i] = lst

    # Per-bar damp scaling and per-bar phrase pool pickers (from section profiles)
    damp_scale_by_bar: Dict[int, Tuple[int, int]] = {}
    bar_pool_pickers: Dict[int, PoolPicker] = {}

    # Determine section labels per bar
    section_labels: List[str] = []
    if sections:
        section_labels = section_labels_override or [section_default] * len(downbeats)
        if section_labels:
            section_labels = section_labels[: len(downbeats)] + [
                section_default
            ] * max(0, len(downbeats) - len(section_labels))
        else:
            section_labels = [section_default] * len(downbeats)
    elif stats is not None:
        labels_from_stats = stats.get("section_labels") or stats.get("sections")
        if isinstance(labels_from_stats, list):
            section_labels = list(labels_from_stats)
        else:
            section_labels = [section_default] * len(downbeats)
    else:
        section_labels = [section_default] * len(downbeats)

    # Apply section_profiles overrides
    if section_profiles:
        for i, tag in enumerate(section_labels):
            prof = section_profiles.get(tag)
            if not prof:
                continue
            if "accent" in prof:
                accent_by_bar[i] = prof["accent"]
            if "accent_scale" in prof:
                try:
                    accent_scale_by_bar[i] = float(prof["accent_scale"])
                except Exception:
                    pass
            if "damp_scale" in prof:
                ds = prof["damp_scale"]
                if isinstance(ds, list) and len(ds) == 2:
                    damp_scale_by_bar[i] = (int(ds[0]), int(ds[1]))
            # Phrase pool / picker policy
            if "phrase_pool" in prof:
                notes = prof.get("phrase_pool", {}).get("notes", [])
                weights = prof.get("phrase_pool", {}).get("weights", [1] * len(notes))
                pool: List[Tuple[int, float]] = []
                for n, w in zip(notes, weights):
                    nt = parse_note_token(n)
                    if nt is not None:
                        pool.append((int(nt), float(w)))
                if pool:
                    bar_pool_pickers[i] = PoolPicker(pool, phrase_pick, rng=rng_pool)
            if "phrase_pick" in prof:
                # Rebuild picker with requested mode, reuse pool if available
                existing_pool = bar_pool_pickers[i].pool if i in bar_pool_pickers else []
                bar_pool_pickers[i] = PoolPicker(existing_pool, prof["phrase_pick"], rng=rng_pool)
            if prof.get("no_immediate_repeat"):
                no_repeat_window = max(no_repeat_window, 1)

    # Density-driven overrides (e.g., force phrase keys on busy/silent bars)
    density_override: Dict[int, int] = {}
    if density_rules is None:
        density_rules = [
            {"rest_ratio": 0.5, "note": 24},
            {"onset_count": 3, "note": 36},
        ]
    if rest_list is not None and onset_list is not None:
        for i, (r, o) in enumerate(zip(rest_list, onset_list)):
            for rule in density_rules:
                note = None
                if "rest_ratio" in rule and r >= rule["rest_ratio"]:
                    note = parse_note_token(rule["note"])
                elif "onset_count" in rule and o >= rule["onset_count"]:
                    note = parse_note_token(rule["note"])
                if note is not None:
                    density_override[i] = int(note)
                    break

    if section_verbose and section_labels:
        logging.info("sections: %s", section_labels)
    # phrase_plan and fill_map have already been computed earlier via
    # schedule_phrase_keys (which also handles style inject and LFO fills).
    # Avoid resetting them here so later features can override as expected.

    # --- merged: phrase_pool setup + trend + unified pick_phrase_note ---
    # State for guide/density and pool picking
    last_guided: Optional[int] = None
    prev_hold: Optional[int] = None

    # Optional global pool picker constructed from phrase_pool arg
    pool_picker: Optional[PoolPicker] = None
    if phrase_pool:
        if isinstance(phrase_pool, list):
            phrase_pool = {"pool": phrase_pool}
        if phrase_pool.get("pool"):
            pool_picker = PoolPicker(
                phrase_pool["pool"],
                phrase_pick,
                T=phrase_pool.get("T"),
                no_repeat_window=no_repeat_window,
                rng=rng_pool,
            )

    # Simple trend labels over onset density
    trend_labels: List[int] = []
    if onset_list is not None:
        trend_labels = [0] * len(onset_list)
        if trend_window > 0 and len(onset_list) > trend_window:
            for i in range(trend_window, len(onset_list)):
                prev = sum(onset_list[i - trend_window : i]) / trend_window
                curr = sum(onset_list[i - trend_window + 1 : i + 1]) / trend_window
                slope = curr - prev
                if slope > trend_th:
                    trend_labels[i] = 1
                elif slope < -trend_th:
                    trend_labels[i] = -1

    # State for cycle/plan-based picking (from main branch)
    last_bar_idx = -1
    last_bar_note: Optional[int] = None

    def pick_phrase_note(t: float, chord_idx: int) -> Optional[int]:
        nonlocal last_guided, last_bar_idx, last_bar_note
        bar_idx = max(0, bisect.bisect_right(downbeats, t) - 1)
        pn: Optional[int] = None
        decided = False
        plan_tried = False

        if guide_notes is not None:
            if guide_quant == "bar":
                base = bar_idx
            else:
                base = max(0, bisect.bisect_right(beat_times, t) - 1)
            if base in guide_notes:
                note = guide_notes.get(base)
                decided = True
                if note is not None and note != last_guided:
                    pn = note
                    last_guided = note

        if pn is None and bar_idx in density_override:
            pn = density_override[bar_idx]
            decided = True

        if pn is None and vocal_adapt is not None:
            alt = vocal_adapt.phrase_for_bar(bar_idx)
            if alt is not None:
                pn = alt
                decided = True

        if pn is None:
            picker = bar_pool_pickers.get(bar_idx)
            if picker is not None:
                decided = True
                if trend_labels and bar_idx < len(trend_labels) and trend_labels[bar_idx] != 0:
                    notes = [n for n, _ in picker.pool]
                    pn = max(notes) if trend_labels[bar_idx] > 0 else min(notes)
                else:
                    pn = picker.pick()
            elif pool_picker is not None:
                decided = True
                if trend_labels and bar_idx < len(trend_labels) and trend_labels[bar_idx] != 0:
                    notes = [n for n, _ in pool_picker.pool]
                    pn = max(notes) if trend_labels[bar_idx] > 0 else min(notes)
                else:
                    pn = pool_picker.pick()

        if (
            pn is None
            and plan_active
            and phrase_plan is not None
            and (phrase_plan or cycle_notes)
        ):
            plan_tried = True
            if bar_idx < len(phrase_plan):
                pn = phrase_plan[bar_idx]
            else:
                cand = None
                if cycle_notes:
                    idx = ((bar_idx + cycle_start_bar) // max(1, cycle_stride)) % len(cycle_notes)
                    cand = cycle_notes[idx]
                phrase_plan.append(cand)
                pn = cand

        if pn is None and not (decided or plan_tried):
            pn = phrase_note

        if bar_idx != last_bar_idx:
            last_bar_idx = bar_idx
            if pn is None:
                last_bar_note = pn
                return None
            if plan_active and plan_tried and pn == last_bar_note and cycle_stride <= 1:
                last_bar_note = pn
                return None
            last_bar_note = pn
        return pn

    silent_qualities = set(silent_qualities or [])

    def _bar_preset_for(index: int) -> Dict[str, float]:
        base = (
            DENSITY_PRESETS.get(stats.get("bar_density", {}).get(index, "med"), DENSITY_PRESETS["med"])
            if stats
            else DENSITY_PRESETS["med"]
        )
        preset = dict(base)
        stride_val = preset.get("stride")
        default_stride = 1
        if stride_val is None:
            stride_int = default_stride
        else:
            try:
                stride_int = int(round(float(stride_val)))
            except Exception:
                stride_int = default_stride
        preset["stride"] = max(1, stride_int)
        return preset

    bar_presets = {i: _bar_preset_for(i) for i in range(len(downbeats))}
    precomputed_accents = {i: accent_for_bar(i) for i in range(len(downbeats))}
    vf0_by_bar = {i: vel_factor(vel_curve, 0, bar_counts.get(i, 1)) for i in range(len(downbeats))}

    ctx = RuntimeContext(
        rng=rng,
        section_lfo=section_lfo,
        humanize_ms=humanize_ms,
        humanize_vel=humanize_vel,
        beat_to_time=beat_to_time,
        time_to_beat=time_to_beat,
        clip=clip_note_interval,
        maybe_merge_gap=maybe_merge_gap,
        append_phrase=_append_phrase,
        vel_factor=vel_factor,
        accent_by_bar=precomputed_accents,
        bar_counts=bar_counts,
        preset_by_bar=bar_presets,
        accent_scale_by_bar=accent_scale_by_bar,
        vel_curve=vel_curve,
        downbeats=downbeats,
        cycle_mode=cycle_mode,
        phrase_len_beats=phrase_len_beats,
        phrase_inst=phrase_inst,
        pick_phrase_note=pick_phrase_note,
        release_sec=release_sec,
        min_phrase_len_sec=min_phrase_len_sec,
        phrase_vel=phrase_vel,
        duck=_duck,
        lfo_targets=lfo_targets,
        stable_guard=stable_guard,
        stats=stats,
        bar_progress=bar_progress,
        pulse_subdiv_beats=pulse_subdiv_beats,
        swing=swing,
        swing_unit_beats=swing_unit_beats,
        swing_shape=swing_shape,
    )

    prev_triad_voicing: Optional[List[int]] = None
    for c_idx, span in enumerate(chords):
        is_silent = span.quality in silent_qualities or span.quality == "rest"
        triad: List[int] = []
        if not is_silent:
            triad = triad_pitches(span.root_pc, span.quality, chord_oct, mapping)
            if chord_range:
                mode = voicing_mode if voicing_mode != "smooth" else "stacked"
                triad = place_in_range(
                    triad, chord_range["lo"], chord_range["hi"], voicing_mode=mode
                )
                if voicing_mode == "smooth":
                    triad = smooth_triad(prev_triad_voicing, triad, chord_range["lo"], chord_range["hi"])
                    prev_triad_voicing = triad
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
                vf = vf0_by_bar[bar_idx]
                pulse_idx = pulses_in_range(span.start, span.end)
                preset = bar_presets[bar_idx]
                acc_arr = precomputed_accents.get(bar_idx)
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
                        1,
                        min(127, int(round(base_vel + rng.uniform(-humanize_vel, humanize_vel)))),
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
                if rest_silence_hold_off and guide_units and guide_notes is not None:
                    sb = time_to_beat(start_t)
                    uidx = bisect.bisect_right(unit_starts, sb) - 1
                    if 0 <= uidx < len(unit_starts) - 1 and guide_notes.get(uidx + 1) is None:
                        end_t = min(end_t, beat_to_time(unit_starts[uidx + 1]))
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
                    vf = vf0_by_bar[bar_idx]
                    pulse_idx = pulses_in_range(start, end)
                    preset = bar_presets[bar_idx]
                    acc_arr = precomputed_accents.get(bar_idx)
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
                                127,
                                int(round(base_vel + rng.uniform(-humanize_vel, humanize_vel))),
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
                    if rest_silence_hold_off and guide_units and guide_notes is not None:
                        sb = time_to_beat(start_t)
                        uidx = bisect.bisect_right(unit_starts, sb) - 1
                        if 0 <= uidx < len(unit_starts) - 1 and guide_notes.get(uidx + 1) is None:
                            end_t = min(end_t, beat_to_time(unit_starts[uidx + 1]))
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
            _emit_phrases_for_span(span, c_idx, ctx)

    if phrase_change_lead_beats > 0 and phrase_plan:
        lead_len = min(pulse_subdiv_beats, phrase_change_lead_beats)
        for i in range(1, len(phrase_plan)):
            cur = phrase_plan[i]
            prev = phrase_plan[i - 1]
            if cur is not None and prev is not None and cur != prev:
                start_b = time_to_beat(downbeats[i]) - lead_len
                start_t = beat_to_time(start_b)
                end_t = downbeats[i]
                _append_phrase(
                    phrase_inst,
                    cur,
                    start_t,
                    end_t,
                    phrase_vel,
                    -1.0,
                    release_sec,
                    min_phrase_len_sec,
                )
    finalize_phrase_track(
        out,
        None,
        stats,
        mapping,
        section_lfo=section_lfo,
        lfo_targets=lfo_targets,
        downbeats=downbeats,
        guide_units=guide_units,
        guide_units_time=None,
        guide_notes=guide_notes,
        rest_ratios=None,
        onset_counts=None,
        chord_inst=chord_inst,
        phrase_inst=phrase_inst,
        beat_to_time=beat_to_time,
        time_to_beat=time_to_beat,
        pulse_subdiv_beats=pulse_subdiv_beats,
        phrase_vel=phrase_vel,
        phrase_merge_gap=phrase_merge_gap,
        release_sec=release_sec,
        min_phrase_len_sec=min_phrase_len_sec,
        stop_min_gap_beats=stop_min_gap_beats,
        stop_velocity=stop_velocity,
        damp_dst=None,
        damp_cc_num=0,
        guide_cc=None,
        bpm=bpm,
        section_overrides=sections,
        fill_map=fill_map,
        rest_silence_send_stop=rest_silence_send_stop,
        quantize_strength=quantize_strength,
        write_markers=write_markers,
        marker_encoding=marker_encoding,
        section_labels=section_labels,
        section_default=section_default,
        chord_merge_gap=chord_merge_gap,
        clone_meta_only=clone_meta_only,
        meta_src=meta_src,
        chords=chords,
    )

    _legato_merge_chords(chord_inst, chord_merge_gap)
    out.instruments.append(chord_inst)
    out.instruments.append(phrase_inst)
    sections_for_clamp = locals().get("normalized_sections")
    fallback_end = song_end_hint if song_end_hint is not None else pm_input_end
    song_end_time, notes_clipped = finalize_song_length(
        out, downbeats, sections_for_clamp, fallback_end
    )
    if clone_meta_only and logging.getLogger().isEnabledFor(logging.INFO):
        logging.info("clone_meta_only tempo/time-signature via %s API", meta_src)
    if stats is not None:
        stats["pulse_count"] = len(phrase_inst.notes)
        stats["bar_count"] = max(0, len(downbeats) - 1)
        if song_end_hint is not None and math.isfinite(song_end_hint):
            stats["song_end_hint"] = float(song_end_hint)
        if downbeats:
            stats["downbeats_last"] = float(downbeats[-1])
        if song_end_time is not None and math.isfinite(song_end_time):
            stats["song_end_time"] = float(song_end_time)
        else:
            stats["song_end_time"] = None
        stats["notes_clipped"] = stats.get("notes_clipped", 0) + notes_clipped
    _sanitize_tempi(out)
    _ensure_tempo_and_ticks(out, seed_bpm, out.time_signature_changes)
    if stats is not None:
        try:
            stats["out_end_time"] = float(out.get_end_time())
        except Exception:
            stats["out_end_time"] = None
    return out


_BUILD_SPARKLE_MIDI_IMPL = build_sparkle_midi


def main():
    ap = argparse.ArgumentParser(
        description="Convert generic MIDI to UJAM Sparkle-friendly MIDI (chords + common pulse)."
    )
    ap.add_argument("input_midi", type=str, help="Input MIDI file")
    ap.add_argument("--out", type=str, required=True, help="Output MIDI file")
    ap.add_argument(
        "--pulse", type=str, default="1/8", help="Pulse subdivision (e.g., 1/8, 1/16, 1/4)"
    )
    ap.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="Fallback BPM if input has no tempo (also used for bar-index chord CSVs)",
    )
    ap.add_argument(
        "--chords",
        type=str,
        default=None,
        help=(
            "Chord CSV/YAML path or inline spec. CSV may use start,end,root,quality; "
            "start,end,chord; start,chord; bar,chord; bar_start,bar_end,chord; or "
            "bar,beat,chord (beat accepts integers/floats/fractions). Headerless two-"
            "column is auto-detected. Inline examples: 0:G:maj,2:D:maj or a JSON list."
        ),
    )
    ap.add_argument(
        "--chords-ts",
        type=str,
        default=None,
        help=(
            "Optional meter hints for compact chord CSV when MIDI lacks time-signature info. "
            "Format: '12/8@0,4/4@20' (denotes num/den active from time-sec)."
        ),
    )
    ap.add_argument(
        "--strict-chords",
        action="store_true",
        help=(
            "Be strict about overlapping/duplicate chord rows (currently the loader already "
            "raises on invalid or overlapping spans; this flag is reserved for future relaxation)."
        ),
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

    # Phrase/guide/density controls
    ap.add_argument(
        "--phrase-pool",
        type=str,
        default=None,
        help="JSON list or mapping of phrase notes with optional weights",
    )
    ap.add_argument(
        "--phrase-pick",
        choices=["roundrobin", "random", "weighted", "markov"],
        default="roundrobin",
        help="Selection policy for phrase-pool",
    )
    ap.add_argument(
        "--no-repeat-window",
        type=int,
        default=1,
        help="Avoid repeating phrase notes within last K picks (default 1)",
    )
    ap.add_argument("--guide-midi", type=str, default=None, help="Guide MIDI for phrase selection")
    ap.add_argument(
        "--guide-quant",
        choices=["bar", "beat"],
        default="bar",
        help="Quantization unit for guide analysis",
    )
    ap.add_argument(
        "--guide-thresholds",
        type=str,
        default='{"low":24,"mid":26,"high":36}',
        help="JSON mapping for density categories to phrase notes",
    )
    ap.add_argument(
        "--guide-rest-silence-th",
        type=float,
        default=None,
        help="Rest ratio >=th suppresses phrase trigger",
    )
    ap.add_argument(
        "--guide-onset-th",
        type=str,
        default='{"mid":1,"high":3}',
        help="JSON thresholds for onset counts",
    )
    ap.add_argument(
        "--guide-pick",
        choices=["roundrobin", "random", "weighted", "markov"],
        default="roundrobin",
        help="Selection policy when guide thresholds give lists",
    )

    # Auto fill & damping/CC
    ap.add_argument(
        "--auto-fill",
        choices=["off", "section_end", "long_rest"],
        default="off",
        help="Insert style fill once",
    )
    ap.add_argument(
        "--fill-length-beats", type=float, default=0.25, help="Length of style fill in beats"
    )
    ap.add_argument(
        "--fill-min-gap-beats",
        type=float,
        default=0.0,
        help="Minimum gap before inserting another fill",
    )
    ap.add_argument(
        "--fill-avoid-pitches",
        type=str,
        default=None,
        help="Comma-separated pitches to avoid when inserting fills",
    )
    # Unified --damp option (e.g., "vocal:cc=11,channel=1").
    # If provided, it will override individual --damp-* flags below.
    ap.add_argument(
        "--damp", type=str, default=None, help="Unified damping spec, e.g., 'vocal:cc=11,channel=1'"
    )
    ap.add_argument(
        "--damp-cc", type=int, default=None, help="Emit CC from guide rest ratio (default 11)"
    )
    ap.add_argument(
        "--damp-dst",
        choices=["phrase", "chord", "newtrack"],
        default="newtrack",
        help="Destination track for damping CC",
    )
    ap.add_argument(
        "--damp-scale", type=int, nargs=2, default=None, help="Scale damping CC to [lo hi]"
    )
    ap.add_argument(
        "--damp-curve",
        choices=["linear", "exp", "inv"],
        default="linear",
        help="Curve for damping CC mapping",
    )
    ap.add_argument("--damp-gamma", type=float, default=1.6, help="Gamma for exp damping curve")
    ap.add_argument(
        "--damp-smooth-sigma",
        type=float,
        default=0.0,
        help="Gaussian sigma for damping CC smoothing",
    )
    ap.add_argument(
        "--damp-cc-min-interval-beats",
        type=float,
        default=0.0,
        help="Minimum beat interval between damping CC events",
    )
    ap.add_argument(
        "--damp-cc-deadband", type=int, default=0, help="Drop CC if change within this value"
    )
    ap.add_argument(
        "--damp-cc-clip", type=int, nargs=2, default=None, help="Clip damping CC to [lo hi]"
    )

    # Sections & profiles
    ap.add_argument(
        "--sections",
        type=str,
        default=None,
        help=(
            "JSON list: labels [\"A\",\"B\",...] or dicts [{\"start_bar\":0,\"end_bar\":8,\"tag\":\"A\"}]. "
            "Values are normalized into contiguous bar ranges before fill planning (priority: CLI > mapping > guide)."
        ),
    )

    ap.add_argument(
        "--section-profiles", type=str, default=None, help="YAML file of section profiles"
    )
    ap.add_argument(
        "--section-default", type=str, default="verse", help="Default section tag if none"
    )
    ap.add_argument("--section-verbose", action="store_true", help="Log per-bar section tags")

    # Humanize / groove / accents / swing
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
    ap.add_argument(
        "--quantize-strength", type=float, default=0.0, help="Post-humanize quantize strength 0..1"
    )
    ap.add_argument("--seed", type=int, default=None, help="Random seed for humanization")
    ap.add_argument("--swing", type=float, default=0.0, help="Swing amount 0..1")
    ap.add_argument(
        "--swing-unit",
        type=str,
        default="1/8",
        choices=["1/8", "1/12", "1/16"],
        help="Subdivision for swing",
    )
    ap.add_argument(
        "--swing-shape",
        choices=["offbeat", "even", "triplet-emph"],
        default="offbeat",
        help="Swing placement pattern",
    )
    ap.add_argument("--accent", type=str, default=None, help="JSON velocity multipliers per pulse")

    # Phrase behavior / merging / holds
    ap.add_argument(
        "--skip-phrase-in-rests", action="store_true", help="Suppress phrase notes in rest spans"
    )
    ap.add_argument(
        "--rest-silence-hold-off",
        action="store_true",
        help="Release held phrase when rest-silence unit encountered",
    )
    ap.add_argument(
        "--rest-silence-send-stop",
        action="store_true",
        help="Emit Stop key when entering rest-silence unit",
    )
    ap.add_argument(
        "--stop-min-gap-beats", type=float, default=0.0, help="Minimum beats between Stop keys"
    )
    ap.add_argument("--stop-velocity", type=int, default=64, help="Velocity for Stop key")
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
        "--phrase-change-lead-beats",
        type=float,
        default=0.0,
        help="Lead time in beats before phrase change",
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

    # Advanced dynamics/injection/guards
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

    # Channels & misc
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
        "--clone-meta-only",
        action="store_true",
        help="Clone only tempo/time-signature from input (best effort across pretty_midi versions)",
    )

    # Templates & debug/reporting
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
    ap.add_argument(
        "--log-level", type=str, default="info", choices=["debug", "info"], help="Logging level"
    )
    ap.add_argument(
        "--debug-json",
        type=str,
        default=None,
        help="Write merged config to PATH (stats: bar_pulse_grid=grid, bar_triggers=hits)",
    )
    ap.add_argument("--debug-md", type=str, default=None, help="Write debug markdown table")
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
        "--debug-midi-out", type=str, default=None, help="Write phrase-only MIDI to PATH"
    )
    ap.add_argument(
        "--bar-summary", type=str, default=None, help="Write per-bar summary CSV (with --dry-run)"
    )
    ap.add_argument("--debug-csv", type=str, default=None, help="Write per-bar debug CSV")
    ap.add_argument(
        "--legacy-bar-pulses-grid",
        action="store_true",
        help="Mirror meter grid into stats['bar_pulses'] for backward compatibility",
    )
    ap.add_argument(
        "--marker-encoding",
        choices=["raw", "ascii", "escape"],
        default="raw",
        help="Marker label encoding: raw keeps input, ascii strips non-ASCII, escape uses \\uXXXX",
    )

    args, extras = ap.parse_known_args()

    stats: Dict[str, Any] = {}
    legacy_bar_pulses_grid = bool(args.legacy_bar_pulses_grid)
    stats_enabled = bool(
        args.debug_json
        or args.report_md
        or args.debug_md
        or args.bar_summary
        or args.print_plan
        or args.debug_csv
        or args.report_json
        or args.dry_run
    )
    if legacy_bar_pulses_grid:
        stats_enabled = True

    # Back-compat: parse unified --damp if present
    if getattr(args, "damp", None):
        mode, kw = parse_damp_arg(args.damp)
        # Map into individual args when possible
        if "cc" in kw and args.damp_cc is None:
            args.damp_cc = int(kw["cc"])  # type: ignore[attr-defined]
        # destination
        if mode in {"phrase", "chord", "newtrack"} and args.damp_dst is None:
            args.damp_dst = mode  # type: ignore[attr-defined]
        # scale/clip/curve-related
        if "clip" in kw and args.damp_cc_clip is None:
            lo, hi = kw["clip"]
            args.damp_cc_clip = (int(lo), int(hi))  # type: ignore[attr-defined]
        if "deadband" in kw and args.damp_cc_deadband == 0:
            args.damp_cc_deadband = int(kw["deadband"])  # type: ignore[attr-defined]
        if "smooth" in kw and args.damp_smooth_sigma == 0.0:
            args.damp_smooth_sigma = float(kw["smooth"])  # type: ignore[attr-defined]

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

    # Logging
    if args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level)

    # Seeding: Python, NumPy (if present), and local RNGs
    if args.seed is not None:
        random.seed(args.seed)
        try:
            import numpy as np  # type: ignore

            np.random.seed(args.seed)
        except Exception:
            pass
    rng_pool = (
        random.Random(args.seed)
        if args.seed is not None
        else (random.Random(0) if _SPARKLE_DETERMINISTIC else random.Random())
    )
    rng_human = (
        random.Random(args.seed + 1)
        if args.seed is not None
        else (random.Random(0) if _SPARKLE_DETERMINISTIC else random.Random())
    )
    rng = (
        random.Random(args.seed)
        if args.seed is not None
        else (random.Random(0) if _SPARKLE_DETERMINISTIC else random.Random())
    )

    # Mapping template path printing
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
    chord_ts_hints = parse_chords_ts_arg(args.chords_ts)
    bpm = ensure_tempo(pm, args.bpm)
    pulse_beats = parse_pulse(args.pulse)
    swing_unit_beats = parse_pulse(args.swing_unit)
    if not (0.0 <= args.swing < 1.0):
        raise SystemExit("--swing must be 0.0<=s<1.0")
    swing = min(float(args.swing or 0.0), 0.9)
    if swing > 0.0 and not math.isclose(swing_unit_beats, pulse_beats, abs_tol=EPS):
        logging.info("swing disabled: swing unit %s != pulse %s", args.swing_unit, args.pulse)
        swing = 0.0

    beat_times_chord: List[float] = list(pm.get_beats())
    if len(beat_times_chord) < 2:
        fallback_bpm = bpm if bpm and math.isfinite(bpm) and bpm > 0 else 120.0
        step = 60.0 / fallback_bpm
        beat_times_chord = [0.0, step]

    def beat_to_time_chord(b: float) -> float:
        if not beat_times_chord:
            return 0.0
        idx = int(math.floor(b))
        if idx < 0:
            return beat_times_chord[0]
        if idx >= len(beat_times_chord) - 1:
            last = beat_times_chord[-1] - beat_times_chord[-2]
            return beat_times_chord[-1] + (b - (len(beat_times_chord) - 1)) * last
        frac = b - idx
        return beat_times_chord[idx] + frac * (beat_times_chord[idx + 1] - beat_times_chord[idx])

    def time_to_beat_chord(t: float) -> float:
        idx = bisect.bisect_right(beat_times_chord, t) - 1
        if idx < 0:
            return 0.0
        if idx >= len(beat_times_chord) - 1:
            last = beat_times_chord[-1] - beat_times_chord[-2]
            if last == 0:
                return float(len(beat_times_chord) - 1)
            return (len(beat_times_chord) - 1) + (t - beat_times_chord[-1]) / last
        span = beat_times_chord[idx + 1] - beat_times_chord[idx]
        if span <= 0:
            return float(idx)
        return idx + (t - beat_times_chord[idx]) / span

    meter_map_chord: List[Tuple[float, int, int]] = []
    no_ts_chord = bool(not pm.time_signature_changes)
    if pm.time_signature_changes:
        for ts in pm.time_signature_changes:
            meter_map_chord.append((float(ts.time), int(ts.numerator), int(ts.denominator)))
    else:
        meter_map_chord.append((0.0, ts_num, ts_den))
    if len(meter_map_chord) > 1:
        meter_map_chord.sort(key=lambda x: x[0])

    downbeats_chord = resolve_downbeats(
        pm,
        meter_map_chord,
        beat_times_chord,
        beat_to_time_chord,
        time_to_beat_chord,
        allow_meter_mismatch=no_ts_chord,
    )

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
    accent_map_cfg = mapping.get("accent_map")
    accent_map = None
    if accent_map_cfg is not None:
        if not isinstance(accent_map_cfg, dict):
            raise SystemExit("accent_map must be mapping meter->list")
        accent_map = {}
        for m, v in accent_map_cfg.items():
            accent_map[str(m)] = validate_accent(v)
    accent = validate_accent(mapping.get("accent")) if accent_map is None else None
    silent_qualities = mapping.get("silent_qualities", [])
    clone_meta_only = bool(mapping.get("clone_meta_only", False))

    # CLI overrides
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
    if args.accent is not None and accent_map is None:
        accent = parse_accent_arg(args.accent)

    sections = mapping.get("sections")
    if args.sections is not None:
        sections = parse_json_arg("--sections", args.sections, SECTIONS_SCHEMA)
    if sections:
        for sec in sections:
            if not isinstance(sec, dict):
                continue
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
    mapping["accent_map"] = accent_map
    mapping["accent"] = accent
    mapping["sections"] = sections
    mapping["silent_qualities"] = silent_qualities

    if stats_enabled:
        stats["sections"] = sections or []
        normalize_sections(
            sections,
            bar_count=None,
            default_tag=args.section_default,
            stats=stats,
        )

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
        style_cfg = parse_json_arg("--style-inject", args.style_inject, STYLE_INJECT_SCHEMA)
    if style_cfg:
        style_cfg = validate_style_inject_cfg(style_cfg, mapping)
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

    phrase_pool = parse_phrase_pool_arg(args.phrase_pool) if args.phrase_pool else None
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

    if stats_enabled:
        stats["_legacy_bar_pulses_grid"] = legacy_bar_pulses_grid
        stats["_section_verbose"] = bool(args.section_verbose)

    if args.debug_json:
        Path(args.debug_json).write_text(json.dumps(mapping, indent=2))

    # Guide MIDI & damping extraction
    guide_notes = None
    guide_cc = None
    guide_units = None
    rest_ratios = None
    onset_counts = None
    damp_cc_num = args.damp_cc if args.damp_cc is not None else None
    if damp_cc_num is None:
        damp_cc_num = 11
    if not (0 <= damp_cc_num <= 127):
        raise SystemExit("--damp-cc must be 0-127")
    damp_dst = args.damp_dst
    if getattr(args, "damp_on_phrase_track", False):  # hidden/compat
        damp_dst = "phrase"

    if args.guide_midi:
        g_pm = pretty_midi.PrettyMIDI(args.guide_midi)
        thresholds = parse_thresholds_arg(args.guide_thresholds)
        onset_cfg = parse_onset_th_arg(args.guide_onset_th)
        (guide_notes, guide_cc, guide_units_time, rest_ratios, onset_counts, sections) = (
            summarize_guide_midi(
                g_pm,
                args.guide_quant,
                thresholds,
                rest_silence_th=args.guide_rest_silence_th,
                onset_th=onset_cfg,
                note_tokens_allowed=False,
                curve=args.damp_curve,
                gamma=args.damp_gamma,
                smooth_sigma=args.damp_smooth_sigma,
                pick_mode=args.guide_pick,
            )
        )
        guide_cc = thin_cc_events(
            guide_cc,
            min_interval_beats=args.damp_cc_min_interval_beats,
            deadband=args.damp_cc_deadband,
            clip=tuple(args.damp_cc_clip) if args.damp_cc_clip else None,
        )
        if args.damp_scale:
            lo, hi = args.damp_scale
            scaled: List[Tuple[float, int]] = []
            for b, v in guide_cc:
                nv = int(round(lo + (hi - lo) * (v / 127.0)))
                nv = max(0, min(127, nv))
                scaled.append((b, nv))
            guide_cc = scaled
        g_beats = g_pm.get_beats()

        def g_time_to_beat(t: float) -> float:
            idx = bisect.bisect_right(g_beats, t) - 1
            if idx < 0:
                return 0.0
            if idx >= len(g_beats) - 1:
                last = g_beats[-1] - g_beats[-2]
                return (len(g_beats) - 1) + (t - g_beats[-1]) / last
            span = g_beats[idx + 1] - g_beats[idx]
            return idx + (t - g_beats[idx]) / span

        guide_units = [(g_time_to_beat(s), g_time_to_beat(e)) for s, e in guide_units_time]
        if stats_enabled:
            stats["sections"] = sections

    # Chords
    inline_chord_events: Optional[List[InlineChordEvent]] = None
    if args.chords:
        chord_path: Optional[Path]
        parsed_inline: Optional[List[InlineChordEvent]] = None
        try:
            chord_path = Path(args.chords)
            path_exists = chord_path.exists()
        except OSError:
            chord_path = None
            path_exists = False
        if not path_exists:
            parsed_inline = parse_inline_chords(args.chords)
        if path_exists and chord_path is not None:
            if chord_path.suffix in {".yaml", ".yml"}:
                chords = read_chords_yaml(chord_path)
            else:
                # Build meter map from existing TS or CLI hints
                meter_map_cli: List[Tuple[float, int, int]] = [
                    (float(ts.time), int(ts.numerator), int(ts.denominator))
                    for ts in pm.time_signature_changes
                ]
                if not meter_map_cli:
                    chord_ts_hints = _parse_chords_ts_hint(getattr(args, "chords_ts", None))
                    if chord_ts_hints:
                        meter_map_cli = chord_ts_hints
                    else:
                        meter_map_cli = [(0.0, ts_num, ts_den)]
                # Downbeats for compact CSV resolution if available
                try:
                    downbeats_chord = list(pm.get_downbeats())
                except Exception:
                    downbeats_chord = []
                # Prefer flexible reader; fall back to simple signature if needed
                try:
                    chords = read_chords_csv(
                        chord_path,
                        bar_times=downbeats_chord if downbeats_chord else None,
                        beat_times=beat_times_chord,
                        meter_map=meter_map_cli,
                        meter_hints=chord_ts_hints,
                        bpm_hint=bpm,
                        default_meter=(ts_num, ts_den),
                    )
                except TypeError:
                    chords = read_chords_csv(
                        chord_path,
                        bar_times=downbeats_chord if downbeats_chord else None,
                        bpm_hint=bpm,
                        default_meter=(ts_num, ts_den),
                    )
        elif parsed_inline is not None:
            inline_chord_events = parsed_inline
            chords = []
        else:
            raise SystemExit(
                f"--chords path not found or unsupported inline spec: {args.chords}"
            )
    else:
        chords = infer_chords_by_bar(pm, ts_num, ts_den)

    if inline_chord_events is not None:
        beat_times = list(pm.get_beats())
        if len(beat_times) < 2:
            fallback_bpm = bpm if bpm and math.isfinite(bpm) and bpm > 0 else 120.0
            step = 60.0 / fallback_bpm
            beat_times = [0.0, step]

        def beat_to_time_local(b: float) -> float:
            idx = int(math.floor(b))
            frac = b - idx
            if idx >= len(beat_times) - 1:
                last = beat_times[-1] - beat_times[-2]
                if last <= 0.0:
                    return beat_times[-1]
                return beat_times[-1] + (b - (len(beat_times) - 1)) * last
            return beat_times[idx] + frac * (beat_times[idx + 1] - beat_times[idx])

        def time_to_beat_local(t: float) -> float:
            idx = bisect.bisect_right(beat_times, t) - 1
            if idx < 0:
                return 0.0
            if idx >= len(beat_times) - 1:
                last = beat_times[-1] - beat_times[-2]
                if last <= 0.0:
                    return float(len(beat_times) - 1)
                return (len(beat_times) - 1) + (t - beat_times[-1]) / last
            span = beat_times[idx + 1] - beat_times[idx]
            if span <= 0.0:
                return float(idx)
            return idx + (t - beat_times[idx]) / span

        inline_timeline: Optional[_BarTimeline] = None
        events: List[Tuple[float, str]] = []
        for idx, ev in enumerate(inline_chord_events):
            start_beats = ev.start_beats
            if start_beats is None:
                if ev.start_time is not None:
                    start_beats = time_to_beat_local(ev.start_time)
                elif ev.bar is not None:
                    bar_idx = ev.bar
                    if inline_timeline is None:
                        try:
                            inline_timeline = _BarTimeline(
                                path=Path("inline"),
                                bar_times=downbeats_chord,
                                beat_times=beat_times_chord,
                                meter_map=meter_map_chord,
                                meter_hints=chord_ts_hints,
                                bpm_hint=bpm,
                                default_meter=(ts_num, ts_den),
                            )
                        except ChordCsvError as exc:
                            raise SystemExit(f"--chords inline: {exc}") from exc
                    try:
                        start_time = inline_timeline.start_time(bar_idx)
                    except ChordCsvError as exc:
                        raise SystemExit(
                            f"--chords inline element {idx}: {exc}"
                        ) from exc
                    start_beats = time_to_beat_local(start_time)
                else:
                    raise SystemExit(
                        f"--chords inline element {idx} missing timing (start_beats/start/bar)"
                    )
            events.append((start_beats, ev.chord.strip()))

        if not events:
            raise SystemExit("--chords inline: no valid tokens")

        events.sort(key=lambda item: item[0])

        midi_end_beats = time_to_beat_local(pm.get_end_time())
        if not math.isfinite(midi_end_beats):
            midi_end_beats = events[-1][0]

        for idx, (start_beats, symbol) in enumerate(events):
            end_beats = events[idx + 1][0] if idx + 1 < len(events) else midi_end_beats
            if end_beats <= start_beats + EPS:
                raise SystemExit(
                    f"--chords inline events must be strictly increasing (index {idx})"
                )
            start_t = beat_to_time_local(start_beats)
            end_t = beat_to_time_local(end_beats)
            if end_t <= start_t + EPS:
                raise SystemExit(
                    f"--chords inline produced non-positive duration near index {idx}; check beat grid"
                )
            try:
                root_pc, quality = parse_chord_symbol(symbol)
            except ValueError as exc:
                raise SystemExit(f"--chords inline index {idx}: {exc}") from exc
            span = ChordSpan(start_t, end_t, root_pc, quality)
            root_display = symbol.split(":", 1)[0]
            setattr(span, "root_name", _guess_root_display(symbol, root_display))
            setattr(span, "symbol", symbol)
            chords.append(span)

    if args.guide_midi and guide_units_time and chords:
        guide_start = min(start for start, _ in guide_units_time)
        guide_end = max(end for _, end in guide_units_time)
        chord_start = min(span.start for span in chords)
        chord_end = max(span.end for span in chords)
        if chord_start > guide_start + EPS or chord_end < guide_end - EPS:
            msg = (
                "Chord timeline does not fully cover guide MIDI span "
                f"(guide {guide_start:.3f}-{guide_end:.3f}s, chords {chord_start:.3f}-{chord_end:.3f}s)"
            )
            if args.strict_chords:
                raise SystemExit(msg)
            logging.warning(msg)

    section_overrides = None
    if args.sections:
        try:
            section_overrides = json.loads(args.sections)
        except Exception:
            raise SystemExit("--sections must be JSON")
    section_profiles = None
    if args.section_profiles:
        if yaml is None:
            raise SystemExit("PyYAML required for --section-profiles")
        section_profiles = yaml.safe_load(Path(args.section_profiles).read_text()) or {}
    density_rules = None

    seed_beats_exc = False
    try:
        beats_hint = list(pm.get_beats())
    except ValueError:
        beats_hint = []
        seed_beats_exc = True
    except Exception:
        beats_hint = []
        seed_beats_exc = True
    try:
        pm_end_time = float(pm.get_end_time())
    except Exception:
        pm_end_time = 0.0

    seed_needed = pm_end_time <= 0.0 or seed_beats_exc or len(beats_hint) < 2
    if seed_needed:
        try:
            seed_bpm = float(bpm)
        except Exception:
            seed_bpm = 120.0
        if not math.isfinite(seed_bpm) or seed_bpm <= 0.0:
            seed_bpm = 120.0
        seconds_per_bar = 0.0
        if math.isfinite(seed_bpm) and seed_bpm > 0.0 and ts_num > 0:
            seconds_per_bar = (60.0 / seed_bpm) * ts_num

        candidates: List[float] = []
        if pm_end_time > 0.0:
            candidates.append(pm_end_time)

        chord_end = 0.0
        if chords:
            try:
                chord_end = max(float(getattr(ch, "end", 0.0)) for ch in chords)
            except Exception:
                chord_end = 0.0
            if chord_end > 0.0:
                candidates.append(chord_end)

        if seconds_per_bar > 0.0:
            if chord_end > 0.0:
                bars_est = int(math.ceil(chord_end / seconds_per_bar))
                if bars_est > 0:
                    candidates.append(bars_est * seconds_per_bar)
            elif chords:
                candidates.append(len(chords) * seconds_per_bar)
            elif not candidates:
                candidates.append(seconds_per_bar)

        computed_end_sec = max((c for c in candidates if c > 0.0), default=0.0)
        if computed_end_sec <= 0.0:
            computed_end_sec = 4.0
            logging.warning(
                "Seeding input grid fallback with default duration %.1fs",
                computed_end_sec,
            )

        logging.info(
            "Seeding input grid fallback: tempo=%.2f BPM, end_sec=%.2f",
            seed_bpm,
            computed_end_sec,
        )
        _seed_input_grid(pm, seed_bpm, computed_end_sec)

    has_phrase_pool = bool(phrase_pool and phrase_pool.get("pool"))
    has_cycle_notes = bool(cycle_notes)
    has_markov = bool(mapping.get("markov"))
    if (
        not has_phrase_pool
        and not has_markov
        and not has_cycle_notes
        and args.phrase_pool is None
        and args.cycle_phrase_notes is None
    ):
        fallback_note = int(mapping.get("phrase_note", 36))
        phrase_pool = {"pool": [(fallback_note, 1.0)]}
        logging.info(
            "No phrase plan supplied; defaulting to steady phrase_note=%d via CLI failsafe",
            fallback_note,
        )

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
        swing=swing,
        swing_unit_beats=swing_unit_beats,
        phrase_channel=phrase_channel,
        chord_channel=chord_channel,
        cycle_stride=cycle_stride,
        accent=accent,
        accent_map=mapping.get("accent_map"),
        skip_phrase_in_rests=args.skip_phrase_in_rests,
        silent_qualities=silent_qualities,
        clone_meta_only=clone_meta_only,
        stats=stats if stats_enabled else None,
        merge_reset_at=merge_reset_at,
        guide_notes=guide_notes,
        guide_quant=args.guide_quant,
        guide_units=guide_units,
        rest_silence_hold_off=args.rest_silence_hold_off,
        phrase_change_lead_beats=args.phrase_change_lead_beats,
        phrase_pool=phrase_pool,
        phrase_pick=args.phrase_pick,
        no_repeat_window=args.no_repeat_window,
        rest_silence_send_stop=args.rest_silence_send_stop,
        stop_min_gap_beats=args.stop_min_gap_beats,
        stop_velocity=args.stop_velocity,
        section_profiles=section_profiles,
        sections=section_overrides,
        section_default=args.section_default,
        section_verbose=args.section_verbose,
        quantize_strength=args.quantize_strength,
        rng_pool=rng_pool,
        rng_human=rng_human,
        write_markers=getattr(args, "write_markers", False),
        marker_encoding=getattr(args, "marker_encoding", "raw"),
        onset_list=onset_counts,
        rest_list=rest_ratios,
        density_rules=density_rules,
        swing_shape=args.swing_shape,
        section_lfo=section_lfo_obj,
        stable_guard=stable_obj,
        fill_policy=args.fill_policy,
        vocal_adapt=vocal_obj,
        vocal_ducking=args.vocal_ducking,
        lfo_targets=tuple(lfo_apply),
        section_pool_weights=spw,
        rng=rng,
        guide_onsets=guide_onsets,
        guide_onset_th=parse_int_or(args.guide_onset_th, 4),
        guide_style_note=guide_style_note,
    )

    # Emit vocal-based damping CC if requested via unified --damp option.
    if getattr(args, "damp", None):
        try:
            mode, kw = parse_damp_arg(args.damp)
        except Exception:
            mode, kw = "none", {}
        if mode == "vocal":
            emit_damping(
                out_pm,
                mode="vocal",
                cc=max(0, min(int(kw.get("cc", 11)), 127)),
                channel=max(0, min(int(kw.get("channel", 0)), 15)),
                vocal_ratios=(vocal_cfg or {}).get("ratios", []),
                downbeats=stats.get("downbeats") or out_pm.get_downbeats(),
            )

    # Map guide beats to out_pm time for CC and unit reporting
    if guide_units:
        out_beats = out_pm.get_beats()

        def out_beat_to_time(b: float) -> float:
            idx = int(math.floor(b))
            frac = b - idx
            if idx >= len(out_beats) - 1:
                last = out_beats[-1] - out_beats[-2]
                return out_beats[-1] + (b - (len(out_beats) - 1)) * last
            return out_beats[idx] + frac * (out_beats[idx + 1] - out_beats[idx])

        guide_units_time = [(out_beat_to_time(s), out_beat_to_time(e)) for s, e in guide_units]
        guide_cc = [(out_beat_to_time(b), v) for b, v in guide_cc]
    else:
        guide_units_time = None

    chord_inst = next((i for i in out_pm.instruments if i.name == CHORD_INST_NAME), None)
    phrase_inst = next((i for i in out_pm.instruments if i.name == PHRASE_INST_NAME), None)
    downbeats_ref: Optional[List[float]]
    if stats_enabled and stats.get("downbeats"):
        downbeats_ref = stats["downbeats"]
    else:
        try:
            downbeats_ref = out_pm.get_downbeats()
        except Exception:
            downbeats_ref = None

    finalize_phrase_track(
        out_pm,
        args,
        stats if stats_enabled else None,
        mapping,
        section_lfo=section_lfo_obj,
        lfo_targets=tuple(lfo_apply),
        downbeats=downbeats_ref,
        guide_units=guide_units,
        guide_units_time=guide_units_time,
        guide_notes=guide_notes,
        rest_ratios=rest_ratios,
        onset_counts=onset_counts,
        chord_inst=chord_inst,
        phrase_inst=phrase_inst,
        beat_to_time=None,
        time_to_beat=None,
        pulse_subdiv_beats=pulse_beats,
        phrase_vel=int(mapping.get("phrase_velocity", 96)),
        phrase_merge_gap=phrase_merge_gap,
        release_sec=phrase_release_ms / 1000.0,
        min_phrase_len_sec=min_phrase_len_ms / 1000.0,
        stop_min_gap_beats=args.stop_min_gap_beats,
        stop_velocity=args.stop_velocity,
        damp_dst=damp_dst,
        damp_cc_num=damp_cc_num,
        guide_cc=guide_cc,
        bpm=bpm,
        section_overrides=section_overrides,
        fill_map=None,
        rest_silence_send_stop=False,
        quantize_strength=0.0,
        write_markers=False,
        section_labels=(
            (stats.get("section_labels") or stats.get("sections")) if stats_enabled else None
        ),
        section_default=args.section_default,
        chord_merge_gap=mapping.get("chord_merge_gap", 0.01),
        clone_meta_only=clone_meta_only,
        meta_src="cli",
        chords=chords,
    )

    if args.dry_run:
        phrase_inst = None
        for inst in out_pm.instruments:
            if inst.name == PHRASE_INST_NAME:
                phrase_inst = inst
                break
        bar_pulses = stats.get("bar_pulses", {}) or {}
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
            prev = stats["bar_phrase_notes"].get(i - 1) if i > 0 else None
            pulses = bar_pulses.get(i, [])
            vels = stats["bar_velocities"].get(i, [])
            if str(mapping.get("phrase_hold")) != "off" and phrase_inst is not None:
                bar_start = stats["downbeats"][i]
                bar_end = (
                    stats["downbeats"][i + 1]
                    if i + 1 < len(stats["downbeats"])
                    else out_pm.get_end_time()
                )
                trig = sum(1 for n in phrase_inst.notes if bar_start <= n.start < bar_end)
                logging.info(
                    "bar %d | phrase %s->%s | triggers %d | vel %s", i, prev, pn, trig, vels
                )
            else:
                logging.info(
                    "bar %d | phrase %s->%s | pulses %d | vel %s", i, prev, pn, len(pulses), vels
                )
        if args.verbose and phrase_inst:
            logging.debug("bar note len_ms vel")
            for n in phrase_inst.notes:
                bar_idx = max(0, bisect.bisect_right(stats["downbeats"], n.start) - 1)
                if bar_idx >= 10:
                    break
                logging.debug(
                    "%3d %4d %7.1f %3d", bar_idx, n.pitch, (n.end - n.start) * 1000.0, n.velocity
                )
        if stats.get("guide_keys"):
            logging.info(
                "guide_keys=%s guide_index=%s",
                stats.get("guide_keys"),
                "beat" if args.guide_quant == "beat" else "bar",
            )
        if stats.get("guide_sample"):
            logging.info("guide_sample=%s", stats.get("guide_sample"))
        if stats.get("auto_fill"):
            logging.info("auto_fill=%s", stats.get("auto_fill"))
        if stats.get("rest_silence") is not None:
            logging.info("rest_silence=%s bars", stats.get("rest_silence"))
        if stats.get("damp_stats"):
            logging.info("damp_cc=%s", stats.get("damp_stats"))
        if args.verbose and bar_pulses:
            for b_idx in sorted(bar_pulses.keys()):
                logging.info("bar %d pulses %s", b_idx, bar_pulses[b_idx])
        globals()["build_sparkle_midi"] = _BUILD_SPARKLE_MIDI_IMPL
        return

    _sanitize_midi_for_mido(out_pm)
    out_pm.write(args.out)
    logging.info("Wrote %s", args.out)
    globals()["build_sparkle_midi"] = _BUILD_SPARKLE_MIDI_IMPL
def _parse_chords_ts_hint(s: Optional[str]) -> Optional[List[Tuple[float, int, int]]]:
    if not s:
        return None
    out: List[Tuple[float, int, int]] = []
    tokens: List[str] = []
    text = s.strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except Exception as exc:
            logging.warning("--chords-ts hint JSON parse failed: %s", exc)
            return None
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str):
                    tokens.append(item.strip())
                else:
                    logging.warning("--chords-ts hint ignoring non-string entry %r", item)
        else:
            logging.warning("--chords-ts hint JSON must be array of strings")
            return None
    else:
        tokens = [tok.strip() for tok in text.split(",") if tok.strip()]

    for tok in tokens:
        if not tok:
            continue
        if "@" in tok:
            sig, at = tok.split("@", 1)
            try:
                t = float(at.strip())
            except Exception as exc:
                logging.warning("--chords-ts hint ignoring %r (%s)", tok, exc)
                continue
        else:
            sig, t = tok, 0.0
        if "/" not in sig:
            logging.warning("--chords-ts hint token %r missing '/'", tok)
            continue
        num_str, den_str = sig.split("/", 1)
        try:
            num = int(num_str.strip())
            den = int(den_str.strip())
        except Exception as exc:
            logging.warning("--chords-ts hint ignoring %r (%s)", tok, exc)
            continue
        out.append((float(t), num, den))
    # sort by start time
    out.sort(key=lambda x: x[0])
    return out


if __name__ == "__main__":
    main()
