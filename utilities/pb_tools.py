"""Convenience helpers for scheduling PrettyMIDI pitch-bend gestures."""

from __future__ import annotations

import math

try:
    import pretty_midi
except Exception:  # pragma: no cover - optional dependency
    pretty_midi = None  # type: ignore

from .pb_math import PB_MIN, PB_MAX, norm_to_pb

__all__ = ["add_pb_slide", "add_pb_vibrato"]


def _require_pretty_midi(func_name: str) -> None:
    if pretty_midi is None:  # pragma: no cover - runtime safeguard
        raise ImportError(f"pretty_midi is required for {func_name}")


def _pb_from_norm(norm: float) -> int:
    """Convert a normalized bend value ``[-1..1]`` to 14-bit PB using project math."""

    value = norm_to_pb(norm)
    if isinstance(value, (list, tuple)):
        # norm_to_pb returns homogeneous type; ensure scalar for safety.
        return int(max(PB_MIN, min(PB_MAX, value[0])))
    return int(max(PB_MIN, min(PB_MAX, value)))


def add_pb_slide(
    inst: "pretty_midi.Instrument",
    t0: float,
    t1: float,
    semitones: float,
    *,
    bend_range: float = 2.0,
    steps_hint: int | None = None,
) -> None:
    """Append a linear pitch-bend slide from ``t0`` to ``t1`` reaching ``semitones``."""

    _require_pretty_midi("add_pb_slide")
    bend_span = abs(bend_range)
    if t1 <= t0 or bend_span == 0:
        return
    span = max(8, int((t1 - t0) * 40.0)) if steps_hint is None else max(1, steps_hint)
    target = max(-bend_span, min(bend_span, semitones))
    target_norm = max(-1.0, min(1.0, target / bend_span))
    for i in range(span + 1):
        frac = i / span
        t = t0 + (t1 - t0) * frac
        pb_val = _pb_from_norm(target_norm * frac)
        inst.pitch_bends.append(pretty_midi.PitchBend(pb_val, t))
    inst.pitch_bends.append(pretty_midi.PitchBend(0, t1 + 1e-3))


def add_pb_vibrato(
    inst: "pretty_midi.Instrument",
    t0: float,
    t1: float,
    *,
    depth_semi: float = 0.15,
    freq_hz: float = 5.5,
    bend_range: float = 2.0,
    steps_hint: int | None = None,
) -> None:
    """Append a sine vibrato curve within ``[t0, t1]`` seconds."""

    _require_pretty_midi("add_pb_vibrato")
    bend_span = abs(bend_range)
    if t1 <= t0 or bend_span == 0:
        return
    span = max(16, int((t1 - t0) * 80.0)) if steps_hint is None else max(1, steps_hint)
    depth_norm = max(0.0, min(1.0, abs(depth_semi) / bend_span))
    for i in range(span + 1):
        frac = i / span
        t = t0 + (t1 - t0) * frac
        norm = depth_norm * math.sin(2.0 * math.pi * freq_hz * (t - t0))
        pb_val = _pb_from_norm(norm)
        inst.pitch_bends.append(pretty_midi.PitchBend(pb_val, t))
    inst.pitch_bends.append(pretty_midi.PitchBend(0, t1 + 1e-3))

