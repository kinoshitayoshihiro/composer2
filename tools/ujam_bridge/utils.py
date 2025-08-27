from __future__ import annotations

from collections.abc import Iterable
from typing import List, Dict
import random

import pretty_midi

from utilities.live_buffer import apply_late_humanization
import groove_profile as gp


def _tick_to_time(pm: pretty_midi.PrettyMIDI, tick: float) -> float:
    if hasattr(pm, "tick_to_time"):
        return pm.tick_to_time(tick)  # type: ignore[arg-type]
    return tick / (pm.resolution * 2)


def _time_to_tick(pm: pretty_midi.PrettyMIDI, time: float) -> float:
    if hasattr(pm, "time_to_tick"):
        return pm.time_to_tick(time)  # type: ignore[arg-type]
    return time * pm.resolution * 2


def quantize(pm: pretty_midi.PrettyMIDI, grid: int, swing: float = 0.0) -> None:
    """Quantize note start/end times to *grid* ticks with optional swing.

    Parameters
    ----------
    pm : pretty_midi.PrettyMIDI
        MIDI container to quantise in-place.
    grid : int
        Grid size in ticks (e.g. 120 for 16th notes when PPQ=480).
    swing : float, optional
        Amount of swing applied to odd subdivisions, by default 0.0.
    """
    for inst in pm.instruments:
        for note in inst.notes:
            start_tick = _time_to_tick(pm, note.start)
            end_tick = _time_to_tick(pm, note.end)
            start_idx = round(start_tick / grid)
            end_idx = round(end_tick / grid)
            start_tick = start_idx * grid
            end_tick = max(end_idx * grid, start_tick + grid)
            if swing and start_idx % 2 == 1:
                start_tick += swing * grid / 2.0
                end_tick += swing * grid / 2.0
            note.start = _tick_to_time(pm, start_tick)
            note.end = _tick_to_time(pm, end_tick)


def chordify(pitches: Iterable[int], play_range: tuple[int, int], *, power_chord: bool = True) -> List[int]:
    """Constrain *pitches* to *play_range* and optionally reduce to power chord.

    Parameters
    ----------
    pitches : Iterable[int]
        Input MIDI note numbers forming a chord.
    play_range : (int, int)
        Inclusive low/high bounds for the instrument's playable range.
    power_chord : bool, optional
        If True, reduce the chord to root+fifth, by default True.
    """
    low, high = play_range
    if not pitches:
        return []
    root = min(pitches)
    if power_chord:
        while root < low:
            root += 12
        while root > high:
            root -= 12
        fifth = root + 7
        if fifth > high:
            fifth -= 12
        notes = [root, fifth]
    else:
        notes = sorted(set(pitches))
        adjusted: List[int] = []
        for n in notes:
            while n < low:
                n += 12
            while n > high:
                n -= 12
            adjusted.append(n)
        notes = adjusted
    return notes


def apply_groove_profile(pm: pretty_midi.PrettyMIDI, profile: Dict[str, float], *, max_ms: float | None = None) -> None:
    """Shift note offsets according to a groove *profile*.

    Parameters
    ----------
    pm : pretty_midi.PrettyMIDI
        MIDI container to modify in-place.
    profile : Dict[str, float]
        Mapping of beat offsets as produced by :mod:`groove_profile`.
    max_ms : float, optional
        If given, cap the absolute timing shift to this many milliseconds.
    """
    for inst in pm.instruments:
        for note in inst.notes:
            beat = _time_to_tick(pm, note.start) / pm.resolution
            new_beat = gp.apply_groove(beat, profile)
            new_start = _tick_to_time(pm, new_beat * pm.resolution)
            delta = new_start - note.start
            if max_ms is not None:
                limit = max_ms / 1000.0
                if delta > limit:
                    delta = limit
                elif delta < -limit:
                    delta = -limit
                new_start = note.start + delta
            note.start = new_start
            note.end += delta


def humanize(pm: pretty_midi.PrettyMIDI, amount: float, *, rng: random.Random | None = None) -> None:
    """Apply slight random timing jitter to *pm* based on *amount*.

    Parameters
    ----------
    pm : pretty_midi.PrettyMIDI
    amount : float
        0.0..1.0 scaling for jitter strength.
    rng : Random, optional
        Source of randomness for deterministic tests.
    """
    if amount <= 0.0:
        return
    rng = rng or random.Random()
    tempo_times, tempo_bpm = pm.get_tempo_changes()
    bpm = float(tempo_bpm[0]) if len(tempo_bpm) else 120.0
    jitter = (5.0 * amount, 10.0 * amount)
    wrappers: List[Dict[str, float]] = []
    mapping: List[tuple[Dict[str, float], pretty_midi.Note, float]] = []
    for inst in pm.instruments:
        for note in inst.notes:
            beat = _time_to_tick(pm, note.start) / pm.resolution
            d: Dict[str, float] = {"offset": beat}
            wrappers.append(d)
            duration = note.end - note.start
            mapping.append((d, note, duration))
    apply_late_humanization(wrappers, jitter_ms=jitter, bpm=bpm, rng=rng)
    for data, note, dur in mapping:
        new_beat = float(data["offset"])
        note.start = _tick_to_time(pm, int(round(new_beat * pm.resolution)))
        note.end = note.start + dur
