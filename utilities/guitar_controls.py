#!/usr/bin/env python3
"""
Minimal Guitar Controls
----------------------
Non-destructive micro-timing/strum and optional CC/keyswitch helpers for guitar-ish tracks.

Design goals
- Only adjust ONSET timing (and optional note-end for strum spill). Do not change pitch.
- Leave velocity/duration "musical shaping" to DUV (run DUV after micro-timing).
- Safe defaults; all effects are bounded and reversible.

Typical order in pipeline
  1) Generate notes (BasePartGenerator)
  2) apply_guitar_controls(...)   # micro-timing & strum (this module)
  3) DUV (velocity/duration humanize)
  4) Optional CC/vibrato/keyswitch scheduling

Dependencies: pretty_midi, numpy
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List
import math
import re
import numpy as np
import pretty_midi as pm

EPS = 1e-9


@dataclass
class StrumConfig:
    include_regex: str = r"(?i)guitar|gtr"
    exclude_regex: str = r"(?i)drum|perc"
    # notes starting within this window (seconds) are treated as a chord cluster
    chord_window_s: float = 0.030
    # total strum span for a cluster (seconds). actual will jitter within ±10%
    span_s: float = 0.018
    # direction: "down" (low→high), "up" (high→low), or "auto" (alternate by beat)
    direction: str = "auto"
    # add small random jitter to every onset (±jitter_s)
    jitter_s: float = 0.004
    # cap for spill: ensure n.end ≥ n.start + min_dur_s
    min_dur_s: float = 0.030
    # optionally tilt note ends to keep chord tail coherent (0..1)
    tail_lock: float = 0.5


@dataclass
class CCConfig:
    enable_cc: bool = False
    cc_number: int = 11  # expression
    # envelope per section: list of (time_s, value0-127)
    envelope: Optional[List[Tuple[float, int]]] = None


@dataclass
class KeyswitchConfig:
    enable_keyswitch: bool = False
    note_number: int = 24  # C1 (example; depends on library)
    velocity: int = 20
    # place a very short KS before the first note of an instrument (seconds)
    pre_time_s: float = 0.010
    dur_s: float = 0.020


@dataclass
class GuitarControlsConfig:
    strum: StrumConfig = StrumConfig()
    cc: CCConfig = CCConfig()
    ks: KeyswitchConfig = KeyswitchConfig()


# -------------------------- helpers --------------------------


def _match_instrument(name: str, inc: str, exc: str) -> bool:
    if re.search(exc, name or ""):
        return False
    return bool(re.search(inc, name or ""))


def _group_chords(notes: List[pm.Note], window_s: float) -> List[List[pm.Note]]:
    if not notes:
        return []
    # Ensure notes are sorted by start time then pitch
    sorted_notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    groups: List[List[pm.Note]] = []
    cur: List[pm.Note] = [sorted_notes[0]]
    last_t = sorted_notes[0].start
    for n in sorted_notes[1:]:
        if abs(n.start - last_t) <= window_s:
            cur.append(n)
        else:
            groups.append(cur)
            cur = [n]
        last_t = n.start
    groups.append(cur)
    return groups


def _apply_strum_to_group(
    group: List[pm.Note],
    base_span: float,
    direction: str,
    jitter_s: float,
    tail_lock: float,
    min_dur_s: float,
) -> None:
    if len(group) <= 1:
        # single note: just jitter
        n = group[0]
        n.start += float(np.random.uniform(-jitter_s, jitter_s))
        n.end = max(n.end, n.start + min_dur_s)
        return

    # decide order
    order = sorted(range(len(group)), key=lambda i: group[i].pitch)
    if direction == "up":
        order = list(reversed(order))
    elif direction == "auto":
        # alternate by average pitch: odd/even heuristic (slight randomness)
        if np.random.rand() < 0.5:
            order = list(reversed(order))
    # direction == "down": keep low→high

    span = base_span * float(np.random.uniform(0.9, 1.1))
    per = span / max(1, len(group) - 1)

    base_t = min(n.start for n in group)
    # apply offsets
    for rank, idx in enumerate(order):
        n = group[idx]
        # center strum around base_t; rank 0 gets no offset
        offset = rank * per
        # add small local jitter
        offset += float(np.random.uniform(-jitter_s, jitter_s))
        n.start = base_t + offset

    # enforce minimal duration and optional tail lock
    tail = max(n.end for n in group)
    for n in group:
        n.end = max(n.end, n.start + min_dur_s)
    if tail_lock > 0:
        # partially pull ends towards common tail to avoid overly fanned releases
        new_tail = tail
        for n in group:
            n.end = n.end * (1 - tail_lock) + new_tail * tail_lock


def _insert_cc(inst: pm.Instrument, cc_number: int, envelope: Sequence[Tuple[float, int]]):
    if not envelope:
        return
    # keep any existing CCs; append ours
    for t, v in envelope:
        v = int(max(0, min(127, v)))
        inst.control_changes.append(pm.ControlChange(cc_number, t, v))
    # keep CCs sorted
    inst.control_changes.sort(key=lambda c: (c.time, c.number))


def _insert_keyswitch(inst: pm.Instrument, ks: KeyswitchConfig):
    if not inst.notes:
        return
    t0 = min(n.start for n in inst.notes)
    ks_time = max(0.0, t0 - ks.pre_time_s)
    inst.notes.append(
        pm.Note(velocity=ks.velocity, pitch=ks.note_number, start=ks_time, end=ks_time + ks.dur_s)
    )
    inst.notes.sort(key=lambda n: (n.start, n.pitch))


# ----------------------- public API --------------------------


def apply_guitar_controls(
    midi: pm.PrettyMIDI, cfg: Optional[GuitarControlsConfig] = None
) -> pm.PrettyMIDI:
    """Apply minimal guitar controls in-place and return the same object.

    Operations:
      - Strumize chord clusters within a small window (default 30ms) with ~18ms span
      - Add tiny onset jitter (±4ms)
      - Keep durations ≥ min_dur_s and softly lock chord tails
      - Optional CC11 envelope & optional keyswitch pre-note

    Notes:
      - This does not touch velocities; run DUV afterwards for vel/dur shaping.
      - Safe for non-drum tracks only; selection is name-based include/exclude.
    """
    if cfg is None:
        cfg = GuitarControlsConfig()

    inc, exc = cfg.strum.include_regex, cfg.strum.exclude_regex

    # Pre-compute bar-alternating direction if auto: per instrument we flip per chord group
    for inst in midi.instruments:
        name = inst.name or ""
        if inst.is_drum:
            continue
        if not _match_instrument(name, inc, exc):
            continue
        # sort notes to obtain clusters
        inst.notes.sort(key=lambda n: (n.start, n.pitch))
        groups = _group_chords(inst.notes, cfg.strum.chord_window_s)

        flip = False
        for g in groups:
            if cfg.strum.direction == "auto":
                dir_now = "down" if not flip else "up"
                flip = not flip
            else:
                dir_now = cfg.strum.direction
            _apply_strum_to_group(
                g,
                base_span=cfg.strum.span_s,
                direction=dir_now,
                jitter_s=cfg.strum.jitter_s,
                tail_lock=cfg.strum.tail_lock,
                min_dur_s=cfg.strum.min_dur_s,
            )

        # Optional CC / KS
        if cfg.cc.enable_cc and cfg.cc.envelope:
            _insert_cc(inst, cfg.cc.cc_number, cfg.cc.envelope)
        if cfg.ks.enable_keyswitch:
            _insert_keyswitch(inst, cfg.ks)

    return midi


# ----------------------- CLI (optional) ----------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Apply minimal guitar controls (strum/jitter/CC/KS) to a MIDI file."
    )
    p.add_argument("input")
    p.add_argument("-o", "--output")
    p.add_argument("--include", default=StrumConfig.include_regex)
    p.add_argument("--exclude", default=StrumConfig.exclude_regex)
    p.add_argument("--span", type=float, default=StrumConfig.span_s)
    p.add_argument("--window", type=float, default=StrumConfig.chord_window_s)
    p.add_argument("--jitter", type=float, default=StrumConfig.jitter_s)
    p.add_argument("--direction", choices=["down", "up", "auto"], default=StrumConfig.direction)
    p.add_argument("--tail-lock", type=float, default=StrumConfig.tail_lock)
    p.add_argument("--cc", action="store_true", help="Enable CC11 default envelope 0→96→80")
    p.add_argument(
        "--ks", action="store_true", help="Insert a short keyswitch note before first note"
    )
    args = p.parse_args()

    midi = pm.PrettyMIDI(args.input)
    cfg = GuitarControlsConfig(
        strum=StrumConfig(
            include_regex=args.include,
            exclude_regex=args.exclude,
            span_s=args.span,
            chord_window_s=args.window,
            jitter_s=args.jitter,
            direction=args.direction,
            tail_lock=args.tail_lock,
        ),
        cc=CCConfig(
            enable_cc=bool(args.cc),
            cc_number=11,
            envelope=[(0.0, 0), (0.1, 96), (1.0, 80)] if args.cc else None,
        ),
        ks=KeyswitchConfig(enable_keyswitch=bool(args.ks)),
    )
    apply_guitar_controls(midi, cfg)
    out = args.output or (args.input.rsplit(".", 1)[0] + ".gtr.mid")
    midi.write(out)
    print(f"[guitar_controls] wrote {out}")
