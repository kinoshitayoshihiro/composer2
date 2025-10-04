#!/usr/bin/env python3
"""
Controls Bundle (Piano / Bass / Drums) — Minimal, Safe Defaults
---------------------------------------------------------------
This module provides three non-destructive, rule-based humanization utilities:
  • apply_piano_controls
  • apply_bass_controls
  • apply_drum_controls

Design
- Touch ONSET timing and a few articulation events (ghost notes, flams, pedal/CC).
- Leave velocity & duration shaping to DUV (run DUV after these controls).
- Keep operations bounded; no negative/zero-length notes.
- PrettyMIDI in-place, return the same object.

Order in pipeline
  1) Generate notes →
  2) apply_*_controls →
  3) DUV (velocity/duration) →
  4) (optional) further CC/PB/KS

Dependencies: pretty_midi, numpy
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List, Iterable
import math
import re
import numpy as np
import pretty_midi as pm

EPS = 1e-9

# --------------------- Common helpers ---------------------


def _match(name: str, include: str, exclude: str) -> bool:
    if re.search(exclude, name or ""):
        return False
    return bool(re.search(include, name or ""))


def _ensure_min_dur(n: pm.Note, min_dur: float) -> None:
    if n.end < n.start + min_dur:
        n.end = n.start + min_dur


def _safe_insert(inst: pm.Instrument, note: pm.Note) -> None:
    # Prevent negative/NaN
    note.start = float(max(0.0, note.start))
    note.end = float(max(note.start + 1e-4, note.end))
    note.velocity = int(max(1, min(127, note.velocity)))
    note.pitch = int(max(0, min(127, note.pitch)))
    inst.notes.append(note)


# --------------------- Piano controls ---------------------


@dataclass
class PianoControlsConfig:
    include_regex: str = r"(?i)piano|keys|grand|upright|ep|rhodes|wurli"
    exclude_regex: str = r"(?i)drum|perc"
    chord_window_s: float = 0.030  # notes within this start window are one chord
    spread_s: float = 0.010  # chord spread (low→high) total span
    jitter_s: float = 0.003  # tiny random jitter on all starts
    min_dur_s: float = 0.040
    sustain_enable: bool = True
    sustain_gap_s: float = 0.025  # pedal up before chord change
    sustain_on_vel: int = 100  # CC64 values (0/127 typical). keep mid for compatibility
    sustain_off_vel: int = 0
    expression_cc11: bool = False
    expr_env: Optional[List[Tuple[float, int]]] = None  # [(time,value0-127), ...]


def _group_by_window(notes: List[pm.Note], window_s: float) -> List[List[pm.Note]]:
    if not notes:
        return []
    out: List[List[pm.Note]] = []
    cur: List[pm.Note] = [notes[0]]
    base = notes[0].start
    for n in notes[1:]:
        if abs(n.start - base) <= window_s:
            cur.append(n)
        else:
            out.append(cur)
            cur = [n]
            base = n.start
    out.append(cur)
    return out


def apply_piano_controls(
    midi: pm.PrettyMIDI, cfg: Optional[PianoControlsConfig] = None
) -> pm.PrettyMIDI:
    if cfg is None:
        cfg = PianoControlsConfig()
    for inst in midi.instruments:
        name = inst.name or ""
        if inst.is_drum or not _match(name, cfg.include_regex, cfg.exclude_regex):
            continue
        inst.notes.sort(key=lambda n: (n.start, n.pitch))
        groups = _group_by_window(inst.notes, cfg.chord_window_s)

        # Chord spread & jitter
        for g in groups:
            if len(g) > 1:
                span = cfg.spread_s * float(np.random.uniform(0.8, 1.2))
                per = span / max(1, len(g) - 1)
                base_t = min(n.start for n in g)
                for i, n in enumerate(sorted(g, key=lambda x: x.pitch)):
                    off = i * per + float(np.random.uniform(-cfg.jitter_s, cfg.jitter_s))
                    n.start = base_t + off
                    _ensure_min_dur(n, cfg.min_dur_s)
            else:
                n = g[0]
                n.start += float(np.random.uniform(-cfg.jitter_s, cfg.jitter_s))
                _ensure_min_dur(n, cfg.min_dur_s)

        # Sustain pedal CC64 around chord boundaries
        if cfg.sustain_enable and inst.notes:
            cc = inst.control_changes
            cc64 = 64
            # collect chord boundary times
            starts = [min(n.start for n in g) for g in groups]
            # schedule: OFF a bit before each boundary (except first), then ON at boundary
            for i, t in enumerate(starts):
                if i > 0:
                    cc.append(
                        pm.ControlChange(cc64, max(0.0, t - cfg.sustain_gap_s), cfg.sustain_off_vel)
                    )
                cc.append(pm.ControlChange(cc64, t + EPS, cfg.sustain_on_vel))
            cc.sort(key=lambda c: (c.time, c.number))

        # Optional expression curve
        if cfg.expression_cc11 and cfg.expr_env:
            for t, v in cfg.expr_env:
                inst.control_changes.append(
                    pm.ControlChange(11, max(0.0, float(t)), int(max(0, min(127, v))))
                )
            inst.control_changes.sort(key=lambda c: (c.time, c.number))
    return midi


# --------------------- Bass controls ---------------------


@dataclass
class BassControlsConfig:
    include_regex: str = r"(?i)bass|contra|upright"
    exclude_regex: str = r"(?i)drum|perc"
    min_dur_s: float = 0.050
    # micro-lag (+) or push (-) in seconds, random within range
    lag_range_s: Tuple[float, float] = (0.003, 0.008)
    jitter_s: float = 0.002
    # ghost notes
    ghost_enable: bool = True
    ghost_prob: float = 0.18
    ghost_lookahead_s: float = 0.08  # place just before accented note
    ghost_dur_s: Tuple[float, float] = (0.06, 0.11)
    ghost_vel_factor: float = 0.35  # relative to next note velocity (if available)
    min_gap_s: float = 0.04  # don't insert if too crowded


def apply_bass_controls(
    midi: pm.PrettyMIDI, cfg: Optional[BassControlsConfig] = None
) -> pm.PrettyMIDI:
    if cfg is None:
        cfg = BassControlsConfig()
    rng = np.random.default_rng()
    for inst in midi.instruments:
        name = inst.name or ""
        if inst.is_drum or not _match(name, cfg.include_regex, cfg.exclude_regex):
            continue
        inst.notes.sort(key=lambda n: (n.start, n.pitch))

        # micro-lag & jitter
        lag = rng.uniform(cfg.lag_range_s[0], cfg.lag_range_s[1])
        for n in inst.notes:
            n.start = max(0.0, n.start + lag + rng.uniform(-cfg.jitter_s, cfg.jitter_s))
            _ensure_min_dur(n, cfg.min_dur_s)

        # ghost notes before accented notes (probabilistic)
        if cfg.ghost_enable and inst.notes:
            new_notes: List[pm.Note] = []
            prev_end = 0.0
            for i, n in enumerate(inst.notes):
                new_notes.append(n)
                if rng.random() < cfg.ghost_prob:
                    start = n.start - cfg.ghost_lookahead_s
                    if start > prev_end + cfg.min_gap_s and start > 0.0:
                        dur = float(rng.uniform(*cfg.ghost_dur_s))
                        vel = int(
                            max(
                                1,
                                min(
                                    127,
                                    int((getattr(n, "velocity", 96) or 96) * cfg.ghost_vel_factor),
                                ),
                            )
                        )
                        ghost = pm.Note(velocity=vel, pitch=n.pitch, start=start, end=start + dur)
                        _safe_insert(inst, ghost)
                        prev_end = ghost.end
            # keep sorted
            inst.notes.sort(key=lambda n: (n.start, n.pitch))
    return midi


# --------------------- Drum controls ---------------------


@dataclass
class DrumControlsConfig:
    include_regex: str = r"(?i)drum|kit|perc|drs"
    exclude_regex: str = r"$^"  # nothing by default
    jitter_s: float = 0.0015
    # flam/drag on snare
    flam_enable: bool = True
    flam_delta_s: Tuple[float, float] = (0.015, 0.025)
    flam_vel_factor: float = 0.6
    flam_prob: float = 0.22
    # push/layback (relative offsets)
    kick_push_s: float = -0.003
    snare_layback_s: float = 0.004


# GM mapping helpers (basic)
GM_KICK = {35, 36}
GM_SNARE = {38, 40}


def apply_drum_controls(
    midi: pm.PrettyMIDI, cfg: Optional[DrumControlsConfig] = None
) -> pm.PrettyMIDI:
    if cfg is None:
        cfg = DrumControlsConfig()
    rng = np.random.default_rng()

    for inst in midi.instruments:
        name = inst.name or ""
        if (not inst.is_drum) or not _match(name, cfg.include_regex, cfg.exclude_regex):
            continue
        inst.notes.sort(key=lambda n: (n.start, n.pitch))

        # base jitter & push/layback
        for n in inst.notes:
            j = rng.uniform(-cfg.jitter_s, cfg.jitter_s)
            if n.pitch in GM_KICK:
                n.start = max(0.0, n.start + cfg.kick_push_s + j)
            elif n.pitch in GM_SNARE:
                n.start = max(0.0, n.start + cfg.snare_layback_s + j)
            else:
                n.start = max(0.0, n.start + j)
            # keep original duration as-is (DUV will handle vel/dur later)

        # flams on snare
        if cfg.flam_enable:
            to_add: List[pm.Note] = []
            for n in list(inst.notes):
                if n.pitch in GM_SNARE and rng.random() < cfg.flam_prob:
                    d = rng.uniform(*cfg.flam_delta_s)
                    vel = int(
                        max(
                            1,
                            min(127, int((getattr(n, "velocity", 96) or 96) * cfg.flam_vel_factor)),
                        )
                    )
                    flam = pm.Note(
                        velocity=vel,
                        pitch=n.pitch,
                        start=max(0.0, n.start - d),
                        end=max(n.start - d + 0.02, n.start - EPS),
                    )
                    to_add.append(flam)
            for g in to_add:
                _safe_insert(inst, g)
            inst.notes.sort(key=lambda n: (n.start, n.pitch))

    return midi


# --------------------- CLI (optional quick test) ---------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Apply minimal controls to a MIDI file (choose instrument kind)."
    )
    p.add_argument("input")
    p.add_argument("-o", "--output")
    p.add_argument("--kind", choices=["piano", "bass", "drums"], required=True)
    args = p.parse_args()

    midi = pm.PrettyMIDI(args.input)

    if args.kind == "piano":
        apply_piano_controls(midi)
    elif args.kind == "bass":
        apply_bass_controls(midi)
    elif args.kind == "drums":
        apply_drum_controls(midi)

    out = args.output or (args.input.rsplit(".", 1)[0] + f".{args.kind}.mid")
    midi.write(out)
    print(f"[controls] wrote {out}")
