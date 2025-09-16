from __future__ import annotations

from collections.abc import Iterable
from typing import List, Dict
import random
import bisect
import os

import pretty_midi

from utilities.live_buffer import apply_late_humanization
import groove_profile as gp

# Set SPARKLE_DETERMINISTIC=1 to force deterministic RNG defaults for tests.
_SPARKLE_DETERMINISTIC = os.getenv("SPARKLE_DETERMINISTIC") == "1"


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


def apply_groove_profile(
    pm: pretty_midi.PrettyMIDI,
    profile: Dict[str, float],
    *,
    beats_per_bar: int | None = 4,
    clip_head_ms: float | None = None,
    clip_other_ms: float | None = None,
    max_ms: float | None = None,
) -> None:
    """Shift note offsets according to a groove *profile*.

    Parameters
    ----------
    pm : pretty_midi.PrettyMIDI
        MIDI container to modify in-place.
    profile : Dict[str, float]
        Mapping of beat offsets as produced by :mod:`groove_profile`.
    beats_per_bar : int, optional
        Legacy default when no time signatures are present, by default 4.
    clip_head_ms, clip_other_ms : float, optional
        Positional clipping in milliseconds for bar-downbeats and other
        positions respectively.  If provided, these take precedence over
        *max_ms*.
    max_ms : float, optional
        Legacy absolute clip used when positional clips are not given.
    """
    downbeats = pm.get_downbeats() if hasattr(pm, "get_downbeats") else [0.0]
    ts_changes = getattr(pm, "time_signature_changes", [])
    ts_idx = 0
    bar_beats: Dict[int, int] = {}
    for i, db in enumerate(downbeats):
        while ts_idx + 1 < len(ts_changes) and ts_changes[ts_idx + 1].time <= db:
            ts_idx += 1
        num = ts_changes[ts_idx].numerator if ts_changes else (beats_per_bar or 4)
        bar_beats[i] = int(num)

    for inst in pm.instruments:
        for note in inst.notes:
            beat = _time_to_tick(pm, note.start) / pm.resolution
            new_beat = gp.apply_groove(beat, profile)
            new_start = _tick_to_time(pm, new_beat * pm.resolution)
            delta = new_start - note.start
            if clip_head_ms is not None and clip_other_ms is not None:
                bar_idx = bisect.bisect_right(downbeats, note.start) - 1
                bar_start_tick = _time_to_tick(pm, downbeats[bar_idx])
                local_beats = bar_beats.get(bar_idx, beats_per_bar or 4)
                beat_idx = int(round((beat * pm.resolution - bar_start_tick) / pm.resolution)) % local_beats
                limit = (clip_head_ms if beat_idx == 0 else clip_other_ms) / 1000.0
                if delta > limit:
                    delta = limit
                elif delta < -limit:
                    delta = -limit
                new_start = note.start + delta
            elif max_ms is not None:
                limit = max_ms / 1000.0
                if delta > limit:
                    delta = limit
                elif delta < -limit:
                    delta = -limit
                new_start = note.start + delta
            note.start = max(0.0, new_start)
            note.end = max(note.start + 0.005, note.end + delta)


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
    if rng is None:
        rng = random.Random(0) if _SPARKLE_DETERMINISTIC else random.Random()
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
        if note.start < 0.0:
            note.end -= note.start
            note.start = 0.0
        note.end = note.start + max(0.005, dur)
