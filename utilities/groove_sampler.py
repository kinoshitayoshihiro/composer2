"""Utilities for generating drum grooves using a simple n-gram model."""

from __future__ import annotations

from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, TypedDict
from random import Random

import pretty_midi
from .drum_map_registry import GM_DRUM_MAP

_PITCH_TO_LABEL: dict[int, str] = {}
"""Mapping from MIDI pitch number to instrument label."""

for lbl, (_, midi_pitch) in GM_DRUM_MAP.items():
    _PITCH_TO_LABEL.setdefault(midi_pitch, lbl)

_VEL_JITTER: tuple[float, float] = (-5, 5)
"""Range of velocity jitter in percent."""

State = tuple[int, str]
"""Represents a model state as ``(bin_index, instrument_label)``."""


class Model(TypedDict):
    """n-gram frequency model used for groove generation."""

    n: int
    freq: dict[State, Counter[State]]


def _iter_drum_notes(midi_path: Path) -> list[tuple[float, int]]:
    """Extract drum notes from a MIDI file.

    Parameters
    ----------
    midi_path : Path
        Path to a MIDI file.

    Returns
    -------
    list[tuple[float, int]]
        Sequence of ``(beat_offset, pitch)`` tuples sorted by offset.
    """
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return []

    drum_hits: list[tuple[float, int]] = []
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        tempo = (
            pm.get_tempo_changes()[1][0] if pm.get_tempo_changes()[1].size else 120.0
        )
        sec_per_beat = 60.0 / tempo
        for n in inst.notes:
            beat = n.start / sec_per_beat
            drum_hits.append((beat, n.pitch))
    drum_hits.sort(key=lambda x: x[0])
    return drum_hits


def load_grooves(dir_path: Path, resolution: int = 16, n: int = 2) -> Model:
    """Build an n-gram model from all MIDI files in ``dir_path``.

    Parameters
    ----------
    dir_path : Path
        Directory containing MIDI files.
    resolution : int, optional
        Number of subdivisions per bar (default ``16``).
    n : int, optional
        Order of the n-gram model to store in the returned object (default ``2``).

    Returns
    -------
    Model
        Mapping of states to transition frequency counters.
    """
    freq: dict[State, Counter[State]] = defaultdict(Counter)

    for midi_path in dir_path.glob("*.mid"):
        notes = _iter_drum_notes(midi_path)
        if not notes:
            continue
        for idx, (off, pitch) in enumerate(notes):
            cur_label = _PITCH_TO_LABEL.get(pitch, str(pitch))
            cur_bin = int(round((off % 4) * (resolution / 4)))
            cur_state: State = (cur_bin, cur_label)
            if idx + 1 < len(notes):
                nxt_off, nxt_pitch = notes[idx + 1]
                nxt_label = _PITCH_TO_LABEL.get(nxt_pitch, str(nxt_pitch))
                nxt_bin = int(round((nxt_off % 4) * (resolution / 4)))
                nxt_state: State = (nxt_bin, nxt_label)
                freq[cur_state][nxt_state] += 1

    return {"n": n, "freq": freq}


def sample_next(prev: State, model: Model, rng: Random) -> State | None:
    """Return the next state sampled from ``model``.

    Parameters
    ----------
    prev : State
        Current state from which to transition.
    model : Model
        Frequency model returned by :func:`load_grooves`.
    rng : Random
        Random number generator used for sampling.

    Returns
    -------
    State | None
        The sampled state or ``None`` if ``prev`` has no outgoing transitions.
    """
    trans = model["freq"].get(prev)
    if not trans:
        return None
    total = sum(trans.values())
    r = rng.uniform(0, total)
    cum = 0.0
    for state, count in trans.items():
        cum += count
        if r <= cum:
            return state
    return None


def generate_bar(
    prev_history: list[str],
    model: Model,
    rng: Random,
    *,
    resolution: int = 16,
) -> list[dict[str, float | str]]:
    """Generate drum events filling one bar.

    Parameters
    ----------
    prev_history : list[str]
        Previous instrument labels providing context for the Markov chain.
    model : Model
        n-gram model returned by :func:`load_grooves`.
    rng : Random
        Source of randomness used during sampling.
    resolution : int, optional
        Number of subdivisions per bar (default ``16``).

    Returns
    -------
    list[dict[str, float | str]]
        Events sorted by ``offset`` covering up to one bar.
    """
    if not model or not model.get("freq"):
        return []

    n = int(model.get("n", 2))  # 将来の n-gram 拡張用

    events: list[dict[str, float | str]] = []
    # Seed with previous instrument if available
    prev_label = prev_history[-1] if prev_history else None
    prev_state = None
    if prev_label is not None:
        # Use bin 0 for history; actual bin doesn't matter for transition lookup
        prev_state = (0, prev_label)

    beat = 0
    iterations = 0
    max_iter = resolution * 4
    while beat < resolution and iterations < max_iter:
        if prev_state is None:
            # random starting state
            candidates = list(model["freq"].keys())
            prev_state = rng.choice(candidates)

        next_state = sample_next(prev_state, model, rng)
        if not next_state:
            break
        bin_idx, inst = next_state
        if bin_idx >= resolution:
            break
        if prev_state is not None and bin_idx <= prev_state[0]:
            bin_idx = prev_state[0] + 1
        vel = 1.0 + rng.uniform(*_VEL_JITTER) / 100.0
        events.append(
            {
                "instrument": inst,
                "offset": bin_idx / resolution,
                "duration": 0.25 / resolution,
                "velocity_factor": vel,
            }
        )
        beat = bin_idx + 1
        prev_state = (bin_idx, inst)
        iterations += 1

    events.sort(key=lambda e: e["offset"])
    return events
