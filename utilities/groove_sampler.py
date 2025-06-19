"""Utilities for generating drum grooves using a simple n-gram model."""

from __future__ import annotations

from pathlib import Path
from collections import Counter, defaultdict
from typing import Callable, Optional, Sequence, TypedDict, Union
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
    freq: dict[tuple[State, ...], Counter[State]]
    unigram: Counter[State]
    resolution: int


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


def load_grooves(dir_path: Path, *, resolution: int = 16, n: int = 2) -> Model:
    """Build an *n*-gram transition-frequency model from ``dir_path``.

    Args:
        dir_path: Directory containing ``.mid`` files.
        resolution: Number of subdivisions per bar.
        n: Order of the n-gram model. Must be at least ``2``.

    Returns:
        Model describing transition frequencies and metadata.

    Raises:
        ValueError: If ``n`` is less than ``2``.

    Notes:
        Offsets are quantized to ``resolution`` steps assuming 4 beats per bar.
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    freq: dict[tuple[State, ...], Counter[State]] = defaultdict(Counter)
    unigram: Counter[State] = Counter()

    for midi_path in dir_path.glob("*.mid"):
        notes = _iter_drum_notes(midi_path)
        if not notes:
            continue
        states: list[State] = []
        for off, pitch in notes:
            label = _PITCH_TO_LABEL.get(pitch, str(pitch))
            bin_idx = int(round((off % 4) * (resolution / 4)))
            state: State = (bin_idx, label)
            unigram[state] += 1
            states.append(state)

        for i in range(len(states) - n + 1):
            ctx = tuple(states[i : i + n - 1])
            nxt = states[i + n - 1]
            freq[ctx][nxt] += 1

    return {"n": n, "freq": freq, "unigram": unigram, "resolution": resolution}


def _choose_weighted(counter: Counter[State], rng: Random) -> Optional[State]:
    """Return one element from ``counter`` sampled proportionally to its count."""

    if not counter:
        return None
    total = sum(counter.values())
    r = rng.uniform(0, total)
    acc = 0.0
    for item, count in counter.items():
        acc += count
        if r <= acc:
            return item
    return None


def sample_next(prev: Sequence[State], model: Model, rng: Random) -> Optional[State]:
    """Return the next state using n-gram back-off.

    Args:
        prev: History states ordered oldest to newest.
        model: Frequency model created by :func:`load_grooves`.
        rng: Random number generator used for sampling.

    Returns:
        The sampled state or ``None`` when the model is empty.

    Notes:
        When the exact ``prev`` context is unseen, the oldest token is dropped
        until a match is found. If no bigram matches exist, sampling falls back
        to the model's unigram frequency distribution.
    """

    if not model.get("freq"):
        return None

    n = max(2, int(model.get("n", 2)))
    ctx = tuple(prev)[-(n - 1) :]
    while ctx:
        counter = model["freq"].get(ctx)
        state = _choose_weighted(counter, rng) if counter else None
        if state is not None:
            return state
        ctx = ctx[1:]

    uni = model.get("unigram")
    return _choose_weighted(uni, rng) if uni else None


def generate_bar(
    prev_history: list[str],
    model: Model,
    rng: Random,
    *,
    length_beats: float = 4.0,
    resolution: int = 16,
    velocity_jitter: Union[tuple[int, int], Callable[[Random], float]] = _VEL_JITTER,
) -> list[dict[str, float | str]]:
    """Generate drum events for a fixed-length segment.

    Args:
        prev_history: Instrument labels providing Markov history.
        model: n-gram model returned by :func:`load_grooves`.
        rng: Source of randomness used during sampling.
        length_beats: Desired length in beats.
        resolution: Subdivisions per bar.
        velocity_jitter: Either a ``(min,max)`` percentage range or a callable
            returning a percentage jitter.

    Returns:
        Events sorted by ``offset`` covering up to ``length_beats`` beats.
    """
    if not model or not model.get("freq"):
        return []

    n = int(model.get("n", 2))

    events: list[dict[str, float | str]] = []
    context: list[State] = []
    for lbl in prev_history[-(n - 1) :]:
        context.append((0, lbl))

    beat = 0
    iterations = 0
    max_step = int(resolution * length_beats / 4)
    max_iter = max_step * 4

    while beat < max_step and iterations < max_iter:
        if not context:
            start_state = _choose_weighted(model["unigram"], rng)
            context.append(start_state)

        next_state = sample_next(tuple(context), model, rng)
        if not next_state:
            break
        bin_idx, inst = next_state
        if bin_idx >= max_step:
            break
        if context and bin_idx <= context[-1][0]:
            bin_idx = context[-1][0] + 1
            if bin_idx >= max_step:
                break
        if isinstance(velocity_jitter, tuple):
            jitter = rng.uniform(*velocity_jitter)
        else:
            jitter = velocity_jitter(rng)
        vel = 1.0 + jitter / 100.0
        events.append(
            {
                "instrument": inst,
                "offset": bin_idx / resolution,
                "duration": 0.25 / resolution,
                "velocity_factor": vel,
            }
        )
        beat = bin_idx + 1
        context.append((bin_idx, inst))
        if len(context) > n - 1:
            context.pop(0)
        iterations += 1

    events.sort(key=lambda e: e["offset"])
    return events
