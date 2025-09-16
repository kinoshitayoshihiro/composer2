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
import json
import os
from functools import lru_cache
import collections
import itertools
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union, Set, Any, Callable
from textwrap import dedent

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

NOTE_RE = re.compile(r"^([A-G][b#]?)(-?\d+)$")

NOTE_ALIASES: Dict[str, int] = {}
NOTE_ALIAS_INV: Dict[int, str] = {}

EPS = 1e-9
MAX_ITERS = 1024

# Set SPARKLE_DETERMINISTIC=1 to force deterministic RNG defaults for tests.
# Opt-in keeps normal randomness intact while letting tests pin results.
_SPARKLE_DETERMINISTIC = os.getenv("SPARKLE_DETERMINISTIC") == "1"


# Helper utilities ---------------------------------------------------------

def pulses_per_bar(num: int, den: int, unit: float) -> int:
    """Return the pulse count implied by a time signature and subdivision."""

    if unit <= 0.0:
        raise ValueError("pulse subdivision must be positive")
    if den == 0:
        raise ValueError("time signature denominator must be non-zero")
    if den == 8:
        return max(1, int(num))
    if den == 4:
        return max(1, int(num * 2))
    bar_beats = num * (4.0 / den)
    total = int(math.floor(bar_beats / unit + EPS))
    return max(1, total)


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
    try:
        tempi_list = list(tempi_seq)
    except Exception:
        tempi_list = []
    try:
        time_list = list(tempo_times)
    except Exception:
        time_list = []
    current_initial = getattr(pm, "initial_tempo", None)
    has_initial = (
        current_initial is not None and math.isfinite(current_initial) and current_initial > 0.0
    )
    if len(tempi_list) == 0 and not has_initial:
        seeded = False
        if hasattr(pm, "_add_tempo_change"):
            try:
                pm._add_tempo_change(bpm, 0.0)  # type: ignore[attr-defined]
                seeded = True
            except Exception:
                seeded = False
        if not seeded:
            pm.initial_tempo = bpm
    elif len(tempi_list) > 0:
        first_time = float(time_list[0]) if time_list else 0.0
        first_tempo = float(tempi_list[0])
        if first_time > EPS:
            if hasattr(pm, "_add_tempo_change"):
                try:
                    pm._add_tempo_change(first_tempo, 0.0)  # type: ignore[attr-defined]
                except Exception:
                    if not has_initial:
                        pm.initial_tempo = first_tempo
            elif not has_initial:
                pm.initial_tempo = first_tempo

    setattr(pm, "_sparkle_meta_seed_fallback", False)

    def _seed_private_tick_tables(pm_obj: "pretty_midi.PrettyMIDI") -> bool:
        used_private = False
        if hasattr(pm_obj, "_tick_scales"):
            try:
                current = pm_obj._tick_scales  # type: ignore[attr-defined]
            except Exception:
                current = None
            need_scale = False
            if current is None:
                need_scale = True
            else:
                try:
                    need_scale = len(current) == 0
                except Exception:
                    need_scale = True
            if need_scale:
                try:
                    pm_obj._tick_scales = [(0, 1.0)]  # type: ignore[attr-defined]
                    used_private = True
                except Exception:
                    pass
        tick_to_time = getattr(pm_obj, "_PrettyMIDI__tick_to_time", None)
        need_tick = False
        if tick_to_time is None:
            need_tick = True
        else:
            try:
                need_tick = len(tick_to_time) == 0
            except Exception:
                need_tick = True
        if need_tick:
            try:
                setattr(pm_obj, "_PrettyMIDI__tick_to_time", [0.0])
                used_private = True
            except Exception:
                pass
        return used_private

    try:
        pm.get_beats()
    except Exception:
        pass

    private_used = False

    def _probe_time(value: float) -> None:
        nonlocal private_used
        try:
            pm.time_to_tick(value)
        except Exception:
            if _seed_private_tick_tables(pm):
                private_used = True
            try:
                pm.time_to_tick(value)
            except Exception:
                pass

    _probe_time(0.0)
    end_probe = 0.0
    if hasattr(pm, "get_end_time"):
        try:
            end_probe = float(pm.get_end_time())
        except Exception:
            end_probe = 0.0
    if end_probe > EPS:
        _probe_time(end_probe)

    if not private_used and _seed_private_tick_tables(pm):
        private_used = True

    if private_used:
        setattr(pm, "_sparkle_meta_seed_fallback", True)


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
    if not rebuild:
        num0, den0 = meter_map[0][1], meter_map[0][2]
        bar_beats = num0 * (4.0 / den0)
        start_b = time_to_beat(downbeats[0])
        next_b = time_to_beat(downbeats[1])
        actual_beats = next_b - start_b
        if abs(actual_beats - bar_beats) > 1e-3:
            rebuild = True

    if rebuild:
        downbeats = []
        for idx, (mt, num, den) in enumerate(meter_map):
            next_t = meter_map[idx + 1][0] if idx + 1 < len(meter_map) else end_t
            start_b = time_to_beat(mt)
            end_b = time_to_beat(next_t)
            bar_beats = num * (4.0 / den)
            if bar_beats <= 0:
                continue
            bar = start_b
            while bar < end_b - EPS:
                downbeats.append(beat_to_time(bar))
                bar += bar_beats
        if not downbeats:
            downbeats = list(beat_times[::4])

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
                interval *= (1 - ctx.swing) if idx % 2 == 0 else (1 + ctx.swing)
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
        if ctx.stats is not None:
            ctx.stats.setdefault("bar_trigger_pulses", {}).setdefault(bar_idx, []).append((b, t))
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
        name, octv = m.groups()
        pc = PITCH_CLASS.get(name)
        if pc is None:
            raise SystemExit(f"Unknown pitch class: {name}")
        val = (int(octv) + 1) * 12 + pc
        if not (0 <= val <= 127):
            raise SystemExit(f"Note out of MIDI range: {token}")
        return val


def parse_int_or(x, default: int) -> int:
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


def parse_note_token(tok: Union[str, int], *, warn_unknown: bool = False) -> Optional[int]:
    """Normalize note token to MIDI int or None for rests."""
    if isinstance(tok, str):
        t = tok.strip()
        if t.lower() in {"rest", "silence", ""}:
            return None
        aliases = NOTE_ALIASES or {}
        if not aliases:
            root_mod = sys.modules.get("sparkle_convert")
            if root_mod is not None:
                aliases = getattr(root_mod, "NOTE_ALIASES", {}) or {}
        if t in aliases:
            return int(aliases[t])
        try:
            return parse_midi_note(t)
        except SystemExit:
            if warn_unknown:
                logging.warning("unknown note alias: %s", tok)
                return None
            raise
    if tok is None:
        return None
    try:
        return validate_midi_note(int(tok))
    except Exception as e:  # pragma: no cover - unlikely
        raise SystemExit(f"Invalid note token: {tok}") from e


def parse_int_or(x, default: int) -> int:
    """Parse int safely (None → default, invalid → default)."""
    if x is None:
        return default
    try:
        return int(str(x).strip())
    except Exception:
        return default


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
    try:
        cfg = json.loads(s)
    except Exception:
        raise SystemExit("--guide-thresholds must be JSON")
    if not isinstance(cfg, dict):
        raise SystemExit("--guide-thresholds must be JSON object")
    for k in ("low", "mid", "high"):
        if k not in cfg:
            raise SystemExit(f"--guide-thresholds missing key {k}")
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
            raise SystemExit(f"--guide-thresholds {k} must be int, note token, or list")
    return cfg


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
    beats = pm.get_beats()
    if not beats:
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

    downs = pm.get_downbeats() if quant == "bar" else beats
    if quant == "bar" and not downs:
        downs = beats[::4]
        if not downs:
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


def insert_style_fill(
    pm_out: "pretty_midi.PrettyMIDI",
    mode: str,
    units: List[Tuple[float, float]],
    mapping: Dict,
    *,
    sections: Optional[List[Dict]] = None,
    rest_ratio_list: Optional[List[float]] = None,
    rest_th: float = 0.75,
    fill_length_beats: float = 0.25,
    bpm: float = 120.0,
    min_gap_beats: float = 0.0,
    avoid_pitches: Optional[Set[int]] = None,
    filled_bars: Optional[List[int]] = None,
) -> int:
    """Insert style fills based on mode."""
    phrase_inst = None
    for inst in pm_out.instruments:
        if inst.name == PHRASE_INST_NAME:
            phrase_inst = inst
            break
    if phrase_inst is None or not units:
        return 0
    pitch = mapping.get("style_fill")
    if pitch is None:
        pitch = 34
        used_notes = {mapping.get("phrase_note")}
        used_notes.update(n for n in mapping.get("cycle_phrase_notes", []) if n is not None)
        if pitch in used_notes:
            pitch = 35
    pitch = int(pitch)
    seed_bpm = float(bpm) if bpm is not None else 120.0
    if not math.isfinite(seed_bpm) or seed_bpm <= 0.0:
        seed_bpm = 120.0
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

    count = 0
    used: set = set()
    if mode == "section_end" and sections:
        for sec in sections:
            idx = int(sec.get("end_bar", 0)) - 1
            if 0 <= idx < len(units) and idx not in used:
                start = units[idx][0]
                start_b = time_to_beat(start)
                length = beat_to_time(start_b + fill_length_beats) - start
                conflict = False
                if avoid_pitches or min_gap_beats > 0.0:
                    for n in phrase_inst.notes:
                        if n.pitch == pitch or (avoid_pitches and n.pitch in avoid_pitches):
                            if start < n.end + EPS and n.start < start + length - EPS:
                                conflict = True
                                break
                            gap = start_b - time_to_beat(n.start)
                            if gap < min_gap_beats:
                                conflict = True
                                break
                if conflict:
                    continue
                phrase_inst.notes.append(
                    pretty_midi.Note(
                        velocity=int(mapping.get("phrase_velocity", 96)),
                        pitch=pitch,
                        start=start,
                        end=start + length,
                    )
                )
                used.add(idx)
                count += 1
                if filled_bars is not None:
                    filled_bars.append(idx)
    elif mode == "long_rest" and rest_ratio_list:
        i = 0
        n = len(rest_ratio_list)
        while i < n:
            if rest_ratio_list[i] >= rest_th:
                idx = i - 1 if i > 0 else 0
                if idx not in used:
                    start = units[idx][0]
                    start_b = time_to_beat(start)
                    length = beat_to_time(start_b + fill_length_beats) - start
                    conflict = False
                    if avoid_pitches or min_gap_beats > 0.0:
                        for n in phrase_inst.notes:
                            if n.pitch == pitch or (avoid_pitches and n.pitch in avoid_pitches):
                                if start < n.end + EPS and n.start < start + length - EPS:
                                    conflict = True
                                    break
                                gap = start_b - time_to_beat(n.start)
                                if gap < min_gap_beats:
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
                        count += 1
                        if filled_bars is not None:
                            filled_bars.append(idx)
                while i < n and rest_ratio_list[i] >= rest_th:
                    i += 1
                continue
            i += 1
    return count


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
) -> int:
    if mode == "off" or picker is None or not units:
        return 0
    phrase_inst = None
    for inst in pm_out.instruments:
        if inst.name == PHRASE_INST_NAME:
            phrase_inst = inst
            break
    if phrase_inst is None:
        return 0
    seed_bpm = float(bpm) if bpm is not None else 120.0
    if not math.isfinite(seed_bpm) or seed_bpm <= 0.0:
        seed_bpm = 120.0
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
        pitch = picker.pick()
        phrase_inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + length)
        )
        count += 1
    return count


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
    sections: Optional[List[Dict]] = None,
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
    onset_list: Optional[List[int]] = None,
    rest_list: Optional[List[float]] = None,
    density_rules: Optional[List[Dict[str, Any]]] = None,
    swing_shape: str = "offbeat",
) -> "pretty_midi.PrettyMIDI":
    """Render Sparkle-compatible MIDI with optional statistics payload.

    When ``stats`` is supplied a schema version of ``2`` is recorded alongside:
    ``bar_pulses`` (density-aware meter grid for backward compatibility) and
    ``bar_trigger_pulses`` (actual emitted trigger pulses, also mirrored to
    ``bar_trigger_pulses_compat`` for transitionary consumers).
    """
    rng = rng_human or rng
    if rng is None:
        rng = random.Random(0) if _SPARKLE_DETERMINISTIC else random.Random()

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
        if hasattr(dest_pm, "_tempo_changes"):
            dest_pm._tempo_changes = []  # type: ignore[attr-defined]
        if hasattr(dest_pm, "_tick_scales"):
            dest_pm._tick_scales = []  # type: ignore[attr-defined]
        if hasattr(dest_pm, "_tick_to_time"):
            dest_pm._tick_to_time = []  # type: ignore[attr-defined]
        if hasattr(dest_pm, "_add_tempo_change"):
            for t, tempo in zip(times, tempos):
                try:
                    dest_pm._add_tempo_change(tempo, t)  # type: ignore[attr-defined]
                except Exception:
                    pass
        elif tempos:
            dest_pm.initial_tempo = tempos[0]

        tempo_changes = getattr(src_pm, "_tempo_changes", None)
        if tempo_changes is not None and hasattr(dest_pm, "_tempo_changes"):
            used_private = True
            try:
                tempo_cls = pretty_midi.containers.TempoChange  # type: ignore[attr-defined]
            except Exception:
                tempo_cls = None
            if tempo_cls is not None:
                dest_pm._tempo_changes = [  # type: ignore[attr-defined]
                    tempo_cls(tc.tempo, tc.time) for tc in tempo_changes
                ]
            else:
                dest_pm._tempo_changes = [  # type: ignore[attr-defined]
                    type(tc)(tc.tempo, tc.time) if hasattr(tc, "tempo") else tc
                    for tc in tempo_changes
                ]

        tick_scales = getattr(src_pm, "_tick_scales", None)
        if tick_scales is not None and hasattr(dest_pm, "_tick_scales"):
            used_private = True
            dest_pm._tick_scales = list(tick_scales)  # type: ignore[attr-defined]

        tick_to_time = getattr(src_pm, "_tick_to_time", None)
        if tick_to_time is not None and hasattr(dest_pm, "_tick_to_time"):
            used_private = True
            dest_pm._tick_to_time = list(tick_to_time)  # type: ignore[attr-defined]

        return "private" if used_private else "public"

    def _new_pretty_midi_with_meta(
        src_pm: "pretty_midi.PrettyMIDI",
    ) -> Tuple["pretty_midi.PrettyMIDI", str]:
        try:
            dest_pm = pretty_midi.PrettyMIDI()
        except TypeError:
            dest_pm = pretty_midi.PrettyMIDI(None)
        _copy_time_signatures_meta(src_pm, dest_pm)
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

    beat_times = pm_in.get_beats()
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
    )
    meter_times = [mt for mt, _, _ in meter_map]
    if cycle_notes and (len(downbeats) - 1) < 2 and cycle_mode == "bar":
        logging.info("cycle disabled; using fixed phrase_note=%d", phrase_note)
        cycle_notes = []
        if stats is not None:
            stats["cycle_disabled"] = True

    if stats is not None:
        stats["schema_version"] = 2
        stats["downbeats"] = downbeats
        # Dict[bar_index, List[(beat_position_in_beats, absolute_time_seconds)]]
        # capturing the meter-derived reference grid per bar. Stored as floats to
        # ease JSON export without further casting. ``bar_trigger_pulses`` records
        # the actual trigger placements when phrases are emitted so analytics can
        # distinguish between the theoretical grid and realised pulses.

        stats["bar_pulses"] = {}
        stats["bar_trigger_pulses"] = {}
        stats["bar_trigger_pulses_compat"] = stats["bar_trigger_pulses"]
        stats["bar_pulse_grid"] = {}
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
        for i in range(len(downbeats)):
            tag = section_default
            for sec in sections:
                if sec.get("start_bar", 0) <= i < sec.get("end_bar", 0):
                    tag = sec.get("tag", section_default)
                    break
            section_labels.append(tag)
    elif stats is not None and stats.get("sections"):
        section_labels = stats["sections"]
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
            for i, pn in enumerate(phrase_plan):
                stats["bar_reason"][i] = {"source": bar_sources[i], "note": pn}
            stats["fill_sources"].update(fill_src)

    # Precompute pulse timestamps per bar (meter-derived grid; swing shifts timing only)
    if stats is not None:
        bar_grid = stats["bar_pulses"]
        bar_full = stats.get("bar_pulse_grid")
        bar_grid.clear()
        if bar_full is not None:
            bar_full.clear()
        for i, start in enumerate(downbeats[:-1]):
            num, den = get_meter_at(meter_map, start, times=meter_times)
            bar_beats = num * (4.0 / den)
            sb = time_to_beat(start)
            if bar_beats <= 0.0:
                pulse = (float(sb), float(start))
                bar_grid[i] = [pulse]
                if bar_full is not None:
                    bar_full[i] = [pulse]
                continue
            base_total = max(1, pulses_per_bar(num, den, pulse_subdiv_beats))
            next_start = downbeats[i + 1]
            bar_end = time_to_beat(next_start)
            if bar_end <= sb:
                bar_end = sb + bar_beats
            base_step = bar_beats / base_total
            full_pulses: List[Tuple[float, float]] = []
            cur = sb
            for base_idx in range(base_total):
                full_pulses.append((float(cur), float(beat_to_time(cur))))
                if base_idx + 1 >= base_total:
                    break
                interval = base_step
                if swing > 0.0 and math.isclose(base_step, swing_unit_beats, abs_tol=EPS):
                    if swing_shape == "offbeat":
                        interval *= (1 + swing) if base_idx % 2 == 0 else (1 - swing)
                    elif swing_shape == "even":
                        interval *= (1 - swing) if base_idx % 2 == 0 else (1 + swing)
                    else:
                        mod = base_idx % 3
                        if mod == 0:
                            interval *= 1 + swing
                        elif mod == 1:
                            interval *= 1 - swing
                cur = clip_to_bar(cur + interval, sb, bar_end)
            if not full_pulses:
                full_pulses = [(float(sb), float(beat_to_time(sb)))]
            stride = 1
            if density_map is not None:
                label = density_map.get(i)
                if label in DENSITY_PRESETS:
                    try:
                        stride = max(1, int(DENSITY_PRESETS[label].get("stride", 1)))
                    except Exception:
                        stride = 1
            pulses = [full_pulses[idx] for idx in range(0, len(full_pulses), stride)]
            if not pulses:
                pulses = [full_pulses[0]]
            if bar_full is not None:
                bar_full[i] = full_pulses
            bar_grid[i] = pulses

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
        for i in range(len(downbeats)):
            tag = section_default
            for sec in sections:
                if sec.get("start_bar", 0) <= i < sec.get("end_bar", 0):
                    tag = sec.get("tag", section_default)
                    break
            section_labels.append(tag)
    elif stats is not None and stats.get("sections"):
        section_labels = stats["sections"]
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

    bar_presets = {
        i: (
            DENSITY_PRESETS.get(stats.get("bar_density", {}).get(i, "med"), DENSITY_PRESETS["med"])
            if stats
            else DENSITY_PRESETS["med"]
        )
        for i in range(len(downbeats))
    }
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

    # --- merged: section-end fills + stop keys + quantize + markers ---

    # 1) Insert section-end fills from main branch plan (with optional LFO scaling)
    fill_inserted = 0
    for bar_idx, pitch in fill_map.items():
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
                if stats is not None:
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
        fill_inserted += 1

    # 2) Send STOP key on long guide rests
    if rest_silence_send_stop and guide_units and guide_notes is not None:
        stop_pitch = mapping.get("style_stop") if isinstance(mapping, dict) else None
        if stop_pitch is not None:
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

    # 3) Quantize phrase notes towards the subdivision grid
    qs_list = quantize_strength if isinstance(quantize_strength, list) else None
    qs_val = float(quantize_strength) if not isinstance(quantize_strength, list) else 0.0
    if (qs_list and any(x > 0 for x in qs_list)) or (qs_list is None and qs_val > 0.0):
        for idx, n in enumerate(phrase_inst.notes):
            sb = time_to_beat(n.start)
            eb = time_to_beat(n.end)
            gs = round(sb / pulse_subdiv_beats) * pulse_subdiv_beats
            ge = round(eb / pulse_subdiv_beats) * pulse_subdiv_beats
            if qs_list:
                strength = qs_list[idx % len(qs_list)]
            else:
                strength = qs_val
            sb = sb + (gs - sb) * strength
            eb = eb + (ge - eb) * strength
            n.start = beat_to_time(sb)
            n.end = beat_to_time(eb)

    # 4) Write section markers
    if write_markers and section_labels:
        for i, t in enumerate(downbeats):
            label = section_labels[i] if i < len(section_labels) else section_default
            try:
                out.markers.append(pretty_midi.Marker(label.upper(), t))
            except Exception:
                pass

    _legato_merge_chords(chord_inst, chord_merge_gap)
    out.instruments.append(chord_inst)
    out.instruments.append(phrase_inst)
    if clone_meta_only and logging.getLogger().isEnabledFor(logging.INFO):
        logging.info("clone_meta_only tempo/time-signature via %s API", meta_src)
    if stats is not None:
        stats["pulse_count"] = len(phrase_inst.notes)
        stats["bar_count"] = len(downbeats)
        stats["fill_count"] = fill_inserted
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
        "--sections", type=str, default=None, help="JSON list of sections (start_bar,end_bar,tag)"
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
    ap.add_argument("--debug-json", type=str, default=None, help="Write merged config to PATH")
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

    args, extras = ap.parse_known_args()

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
    bpm = ensure_tempo(pm, args.bpm)
    pulse_beats = parse_pulse(args.pulse)
    swing_unit_beats = parse_pulse(args.swing_unit)
    if not (0.0 <= args.swing < 1.0):
        raise SystemExit("--swing must be 0.0<=s<1.0")
    swing = min(float(args.swing or 0.0), 0.9)
    if swing > 0.0 and not math.isclose(swing_unit_beats, pulse_beats, abs_tol=EPS):
        logging.info("swing disabled: swing unit %s != pulse %s", args.swing_unit, args.pulse)
        swing = 0.0

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
    mapping["accent_map"] = accent_map
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
        stats["sections"] = sections

    # Chords
    if args.chords:
        chord_path = Path(args.chords)
        if chord_path.suffix in {".yaml", ".yml"}:
            chords = read_chords_yaml(chord_path)
        else:
            chords = read_chords_csv(chord_path)
    else:
        chords = infer_chords_by_bar(pm, ts_num, ts_den)

    stats: Dict = {}

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
        stats=stats,
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

    if section_overrides and stats.get("sections"):
        for sec in section_overrides:
            s = int(sec.get("start_bar", 0))
            e = int(sec.get("end_bar", s))
            tag = sec.get("tag", "section")
            for b in range(s, e):
                if 0 <= b < len(stats["sections"]):
                    stats["sections"][b] = tag

    # Auto fill insertion on guide rests
    fill_count = 0
    sections = stats.get("sections")
    if args.auto_fill != "off" and guide_units_time:
        avoid = None
        if args.fill_avoid_pitches:
            toks = [t.strip() for t in args.fill_avoid_pitches.split(",") if t.strip()]
            avoid = set()
            for tok in toks:
                val = parse_note_token(tok)
                if val is None:
                    raise SystemExit("fill-avoid-pitches cannot include 'rest'")
                avoid.add(val)
        fill_count = insert_style_fill(
            out_pm,
            args.auto_fill,
            guide_units_time,
            mapping,
            sections=sections,
            rest_ratio_list=rest_ratios,
            rest_th=args.guide_rest_silence_th or 0.75,
            fill_length_beats=args.fill_length_beats,
            bpm=bpm,
            min_gap_beats=args.fill_min_gap_beats,
            avoid_pitches=avoid,
            filled_bars=stats.setdefault("fill_bars", []),
        )

    # Emit damping CC to selected destination
    cc_stats = None
    if guide_cc:
        if damp_dst == "phrase":
            inst = next((i for i in out_pm.instruments if i.name == PHRASE_INST_NAME), None)
            if inst is None:
                inst = pretty_midi.Instrument(program=0, name=PHRASE_INST_NAME)
                out_pm.instruments.append(inst)
        elif damp_dst == "chord":
            inst = next((i for i in out_pm.instruments if i.name == CHORD_INST_NAME), None)
            if inst is None:
                inst = chord_inst
                if inst not in out_pm.instruments:
                    out_pm.instruments.append(inst)
        else:
            inst = pretty_midi.Instrument(program=0, name=DAMP_INST_NAME)
            out_pm.instruments.append(inst)
        for t, v in guide_cc:
            inst.control_changes.append(
                pretty_midi.ControlChange(number=damp_cc_num, value=v, time=t)
            )
        if stats is not None:
            vals = [v for _, v in guide_cc]
            cc_stats = {
                "min": min(vals),
                "max": max(vals),
                "mean": sum(vals) / len(vals),
                "count": len(vals),
            }

    # Reports / debug artifacts
    if stats is not None and guide_notes is not None:
        stats["guide_keys"] = [guide_notes.get(i) for i in sorted(guide_notes.keys())]
        stats["fill_count"] = fill_count
        if rest_ratios is not None and onset_counts is not None and guide_cc:
            sample = []
            for i in range(min(4, len(guide_cc))):
                sample.append(
                    {
                        "bar": i,
                        "onset": onset_counts[i],
                        "rest": rest_ratios[i],
                        "cc": guide_cc[i][1],
                    }
                )
            stats["guide_sample"] = sample
            if args.guide_rest_silence_th is not None:
                stats["rest_silence"] = sum(
                    1 for r in rest_ratios if r >= args.guide_rest_silence_th
                )
        if cc_stats:
            stats["damp_stats"] = cc_stats
        stats["auto_fill"] = {
            "mode": args.auto_fill,
            "count": fill_count,
            "length_beats": args.fill_length_beats,
        }

    if args.debug_csv and rest_ratios is not None and onset_counts is not None:
        with open(args.debug_csv, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                ["bar", "onset_count", "rest_ratio", "phrase_note", "cc_value", "section"]
            )
            cc_map = {i: v for i, (_, v) in enumerate(guide_cc)} if guide_cc else {}
            sect = stats.get("sections") or []
            for i in range(len(onset_counts)):
                pn = stats.get("bar_phrase_notes", {}).get(i)
                writer.writerow(
                    [
                        i,
                        onset_counts[i],
                        rest_ratios[i],
                        pn if pn is not None else "",
                        cc_map.get(i, ""),
                        sect[i] if i < len(sect) else "",
                    ]
                )

    if args.bar_summary and rest_ratios is not None and onset_counts is not None:
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
            sect = stats.get("sections") or []
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
                        sect[i] if i < len(sect) else args.section_default,
                        pn if pn is not None else "",
                        pulses,
                        onset_counts[i],
                        rest_ratios[i],
                        avg_vel,
                        1 if i in fill_bars else 0,
                        cc_map.get(i, ""),
                    ]
                )

    if args.report_json:
        p = Path(args.report_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(stats, indent=2, sort_keys=True))

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
            prev = stats["bar_phrase_notes"].get(i - 1) if i > 0 else None
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
        if args.verbose:
            for b_idx in sorted(stats["bar_pulses"].keys()):
                logging.info("bar %d pulses %s", b_idx, stats["bar_pulses"][b_idx])
        return

    out_pm.write(args.out)
    logging.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
