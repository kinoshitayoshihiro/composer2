"""Utilities for generating drum grooves using a simple n-gram model."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from math import ceil, gcd
from pathlib import Path
from random import Random
from typing import TypedDict

import pretty_midi

from .drum_map_registry import GM_DRUM_MAP

logger = logging.getLogger(__name__)

_PITCH_TO_LABEL: dict[int, str] = {}
"""Mapping from MIDI pitch number to instrument label."""

for lbl, (_, midi_pitch) in GM_DRUM_MAP.items():
    _PITCH_TO_LABEL.setdefault(midi_pitch, lbl)

_VEL_JITTER: tuple[float, float] = (-5, 5)
"""Range of velocity jitter in percent."""

State = tuple[int, str]
"""Represents a model state as ``(bin_index, instrument_label)``."""


class Model(TypedDict):
    """n-gram model for drum groove generation."""

    n: int
    resolution: int
    smoothing: float
    freq: dict[int, dict[tuple[State, ...], Counter[State]]]
    prob: dict[int, dict[tuple[State, ...], dict[State, float]]]


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
        tempo_arr = pm.get_tempo_changes()[0]
        tempo = float(tempo_arr[0]) if tempo_arr.size else 120.0
        sec_per_beat = 60.0 / tempo
        for n in inst.notes:
            beat = n.start / sec_per_beat
            drum_hits.append((beat, n.pitch))
    drum_hits.sort(key=lambda x: x[0])
    return drum_hits


def infer_resolution(
    beats: Sequence[float],
    max_denominator: int = 96,
    *,
    beats_per_bar: int | None = None,
) -> int:
    """Return quantization resolution inferred from beat offsets.

    Parameters
    ----------
    beats:
        Offsets in beats.
    max_denominator:
        Upper bound for rational approximation.

    Returns
    -------
    int
        Step count per bar approximating note positions.
    """

    if not beats:
        return 16

    ticks = [round(b * max_denominator) for b in beats]
    base = abs(ticks[0])
    for t in ticks[1:]:
        base = gcd(base, abs(t))

    steps_per_beat = max_denominator // base
    length = beats_per_bar if beats_per_bar is not None else ceil(max(beats)) or 1
    resolution = steps_per_beat * length
    if steps_per_beat == 3:
        resolution = 12
    return resolution


def infer_multistage_resolution(
    beats: Sequence[float], *, beats_per_bar: int = 4
) -> dict[str, int]:
    """Return coarse and fine resolutions for the given offsets."""

    fine = infer_resolution(beats, max_denominator=96, beats_per_bar=beats_per_bar)
    coarse = infer_resolution(beats, max_denominator=24, beats_per_bar=beats_per_bar)
    if coarse > fine:
        coarse = fine
    return {"coarse": coarse, "fine": fine}


def load_grooves(
    dir_path: Path,
    *,
    resolution: int | str = 16,
    n: int = 2,
    smoothing: float = 0.0,
    beats_per_bar: int | None = None,
) -> Model:
    """Create an ``n``-gram model from a folder of MIDI files.

    Parameters
    ----------
    dir_path:
        Directory containing ``.mid`` files.
    resolution:
        Subdivisions per bar or ``"auto"`` to infer from the data.
    n:
        Order of the n-gram model (``>=2``).
    smoothing:
        Weight used when backing off to lower-order distributions.

    Returns
    -------
    Model
        The resulting model with probability tables.
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    all_offsets: list[float] = []
    note_seqs: list[list[tuple[float, int]]] = []
    for midi_path in dir_path.glob("*.mid"):
        notes = _iter_drum_notes(midi_path)
        if not notes:
            logger.warning("No drum notes found in %s", midi_path)
            continue
        note_seqs.append(notes)
        all_offsets.extend(off for off, _ in notes)

    if resolution == "auto":
        resolution = infer_resolution(all_offsets, beats_per_bar=beats_per_bar) if all_offsets else 16

    res_int = int(resolution)

    freq: dict[int, dict[tuple[State, ...], Counter[State]]] = {
        i: defaultdict(Counter) for i in range(1, n)
    }
    freq[0] = {(): Counter()}

    for notes in note_seqs:
        states: list[State] = []
        for off, pitch in notes:
            label = _PITCH_TO_LABEL.get(pitch, str(pitch))
            bin_idx = int(round(off * res_int))
            state = (bin_idx, label)
            freq[0][()][state] += 1
            states.append(state)

        for order in range(1, n):
            for i in range(len(states) - order):
                ctx = tuple(states[i : i + order])
                nxt = states[i + order]
                freq[order][ctx][nxt] += 1

    prob: dict[int, dict[tuple[State, ...], dict[State, float]]] = {i: {} for i in range(n)}
    for order, ctx_dict in freq.items():
        for ctx, counter in ctx_dict.items():
            total = sum(counter.values())
            if total:
                prob[order][ctx] = {s: c / total for s, c in counter.items()}

    return {
        "n": n,
        "resolution": res_int,
        "smoothing": float(smoothing),
        "freq": freq,
        "prob": prob,
    }


def _choose_weighted(prob: dict[State, float], rng: Random) -> State | None:
    """Sample one element from a probability mapping."""

    if not prob:
        return None
    states = list(prob.keys())
    weights = list(prob.values())
    return rng.choices(states, weights=weights, k=1)[0]


def sample_next(
    prev: Sequence[State],
    model: Model,
    rng: Random,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> State | None:
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

    if not model.get("prob"):
        return None
    n = max(2, int(model["n"]))
    order = min(len(prev), n - 1)
    probs: dict[State, float] | None = None
    ctx_seq = tuple(prev)
    for o in range(order, -1, -1):
        ctx = ctx_seq[-o:] if o else ()
        p = model["prob"].get(o, {}).get(ctx)
        if p:
            probs = p
            order = o
            break
    if probs is None:
        return None

    smoothing = float(model.get("smoothing", 0.0))
    while order > 0 and smoothing > 0:
        lower_ctx = ctx_seq[-(order - 1) :] if order - 1 else ()
        lower = model["prob"].get(order - 1, {}).get(lower_ctx)
        if not lower:
            break
        blended: dict[State, float] = {}
        keys = set(probs) | set(lower)
        for k in keys:
            blended[k] = (1 - smoothing) * probs.get(k, 0.0) + smoothing * lower.get(k, 0.0)
        s = sum(blended.values())
        if s:
            for k in blended:
                blended[k] /= s
        probs = blended
        order -= 1

    if top_k is not None:
        sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        probs = {k: v for k, v in sorted_items}

    if temperature != 1.0:
        probs = {k: v ** (1.0 / temperature) for k, v in probs.items()}
        total_t = sum(probs.values())
        if total_t:
            for k in probs:
                probs[k] /= total_t

    return _choose_weighted(probs, rng)


def generate_bar(
    prev_history: list[str],
    model: Model,
    rng: Random,
    *,
    length_beats: float = 4.0,
    resolution: int | None = None,
    velocity_jitter: tuple[int, int] | Callable[[Random], float] = _VEL_JITTER,
    temperature: float = 1.0,
    top_k: int | None = None,
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
    if not model or not model.get("prob"):
        return []

    n = int(model["n"])
    res = resolution or model["resolution"]

    events: list[dict[str, float | str]] = []
    context: list[State] = []
    for lbl in prev_history[-(n - 1) :]:
        context.append((0, lbl))

    beat = 0
    iterations = 0
    max_step = int(res * length_beats)
    max_iter = max_step * 4

    while beat < max_step and iterations < max_iter:
        if not context:
            start_state = _choose_weighted(model["prob"][0][()], rng)
            if not start_state:
                break
            context.append(start_state)

        next_state = sample_next(
            tuple(context), model, rng, temperature=temperature, top_k=top_k
        )
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
        while any(
            e["instrument"] == inst
            and int(round(e["offset"] * res)) == bin_idx
            for e in events
        ):
            bin_idx += 1
        events.append(
            {
                "instrument": inst,
                "offset": bin_idx / res,
                "duration": 0.25 / res,
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


if __name__ == "__main__":  # pragma: no cover - manual use
    import argparse

    parser = argparse.ArgumentParser(description="Build n-gram groove model")
    parser.add_argument("loop_dir", type=Path, help="Directory with MIDI loops")
    parser.add_argument("--n", type=int, default=2, help="Order of the model")
    parser.add_argument(
        "--resolution",
        default="auto",
        help="Subdivisions per bar or 'auto'",
    )
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--stats", action="store_true", help="Print model stats")
    args = parser.parse_args()

    model = load_grooves(
        args.loop_dir,
        resolution=args.resolution,
        n=args.n,
        smoothing=args.smoothing,
    )

    if args.stats:
        print(f"n={model['n']} resolution={model['resolution']}")
