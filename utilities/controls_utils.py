from __future__ import annotations

import math
from typing import Literal

import pretty_midi

try:  # optional
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from utilities.audio_to_midi_batch import _coerce_controller_times
from utilities import pb_math


def _tri(phase: float) -> float:
    """Return triangle wave value for ``phase`` (radians)."""
    return 2.0 / math.pi * math.asin(math.sin(phase))


def synthesize_vibrato(
    inst: pretty_midi.Instrument,
    depth_semitones: float,
    rate_hz: float | None,
    *,
    shape: Literal["sine", "triangle"] = "sine",
) -> list[pretty_midi.PitchBend]:
    """Return pitch-bend vibrato events for ``inst``.

    ``depth_semitones`` is peak-to-peak; the waveform swings by half this
    amount against a ±1 semitone reference before conversion via
    :func:`pb_math.semi_to_pb`.
    """
    if rate_hz is None or depth_semitones <= 0 or not inst.notes:
        return []
    start = min(n.start for n in inst.notes)
    end = max(n.end for n in inst.notes)
    step = 0.02  # 20 ms resolution
    scale = depth_semitones / 2.0
    t = start
    prev = None
    bends: list[pretty_midi.PitchBend] = []
    while t <= end:
        phase = 2 * math.pi * rate_hz * (t - start)
        if shape == "triangle":
            val = _tri(phase)
        else:
            val = math.sin(phase)
        bend = int(pb_math.semi_to_pb(val * scale, 1.0))
        if prev is None or bend != prev:
            bends.append(pretty_midi.PitchBend(pitch=bend, time=float(t)))
            prev = bend
        t += step
    bends.append(pretty_midi.PitchBend(pitch=0, time=float(end)))
    return bends


def synthesize_portamento(
    inst: pretty_midi.Instrument,
    portamento_ms: float | None,
) -> list[pretty_midi.PitchBend]:
    """Return pitch-bend ramps between near-adjacent notes."""
    if portamento_ms is None or portamento_ms <= 0:
        return []
    notes = sorted(inst.notes, key=lambda n: n.start)
    if len(notes) < 2:
        return []
    max_gap = portamento_ms / 1000.0
    bends: list[pretty_midi.PitchBend] = []
    for a, b in zip(notes, notes[1:]):
        gap = b.start - a.end
        if 0 < gap <= max_gap:
            semis = b.pitch - a.pitch
            bend = int(pb_math.semi_to_pb(semis, 2.0))
            bends.append(pretty_midi.PitchBend(pitch=bend, time=float(a.end)))
            bends.append(pretty_midi.PitchBend(pitch=0, time=float(b.start)))
    return bends


def apply_post_bend_policy(
    inst: pretty_midi.Instrument,
    new_bends: list[pretty_midi.PitchBend],
    policy: Literal["skip", "add", "replace"] = "skip",
) -> None:
    """Merge ``new_bends`` into ``inst`` according to ``policy``."""
    if not new_bends:
        return
    if policy == "skip" and inst.pitch_bends:
        return
    if policy == "replace" or not inst.pitch_bends:
        inst.pitch_bends = sorted(new_bends, key=lambda b: b.time)
    elif policy == "add":
        merged: dict[float, int] = {
            round(pb.time, 6): int(pb.pitch) for pb in inst.pitch_bends
        }
        for b in new_bends:
            t = round(b.time, 6)
            merged[t] = int(
                max(pb_math.PB_MIN, min(pb_math.PB_MAX, merged.get(t, 0) + int(b.pitch)))
            )
        inst.pitch_bends = [
            pretty_midi.PitchBend(pitch=p, time=float(t))
            for t, p in sorted(merged.items())
        ]
    _coerce_controller_times(inst)
