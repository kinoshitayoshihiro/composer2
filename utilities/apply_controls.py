from __future__ import annotations

from typing import Callable, Dict, Optional

import pretty_midi

from .controls_spline import ControlCurve


def ensure_instrument_for_channel(
    pm: pretty_midi.PrettyMIDI, ch: int
) -> pretty_midi.Instrument:
    """Return an instrument named for ``ch`` creating one if needed."""

    name = f"channel{ch}"
    for inst in pm.instruments:
        if inst.name == name:
            return inst
    inst = pretty_midi.Instrument(program=0, name=name)
    pm.instruments.append(inst)
    return inst


def write_bend_range_rpn(inst: pretty_midi.Instrument, range_semitones: float) -> None:
    """Emit RPN messages configuring pitch-bend range."""
    msb = int(range_semitones)
    lsb = int(round((range_semitones - msb) * 100))
    msb = max(0, min(127, msb))
    lsb = max(0, min(127, lsb))
    inst.control_changes.extend(
        [
            pretty_midi.ControlChange(number=101, value=0, time=0.0),
            pretty_midi.ControlChange(number=100, value=0, time=0.0),
            pretty_midi.ControlChange(number=6, value=msb, time=0.0),
            pretty_midi.ControlChange(number=38, value=lsb, time=0.0),
            pretty_midi.ControlChange(number=101, value=127, time=0.0),
            pretty_midi.ControlChange(number=100, value=127, time=0.0),
        ]
    )


def apply_controls(
    pm: pretty_midi.PrettyMIDI,
    curves: Dict[str, ControlCurve],
    channel_map: Dict[str, int],
    *,
    bend_range_semitones: float = 2.0,
    write_rpn: bool = True,
    tempo_map: Optional[Callable[[float], float]] = None,
    cc_max_events: int | None = None,
    bend_max_events: int | None = None,
) -> pretty_midi.PrettyMIDI:
    """Render ``curves`` onto ``pm`` according to ``channel_map``.

    ``curves`` keys should be ``"cc11"``, ``"cc64"`` and/or ``"bend"``. Events are
    routed to instruments named after their channels. When ``write_rpn`` is
    ``True`` the pitch-bend range is configured via RPN (0,0) prior to sending
    any bend events. Each channel receives this configuration at most once.
    ``tempo_map`` converts beat positions to BPM for curves defined in the
    ``"beats"`` domain. ``cc_max_events`` and ``bend_max_events`` cap the number
    of rendered events when provided.
    """

    for key in ("cc11", "cc64"):
        curve = curves.get(key)
        chan = channel_map.get(key)
        if curve and chan is not None:
            inst = ensure_instrument_for_channel(pm, chan)
            kwargs = {}
            if curve.domain == "beats":
                kwargs["tempo_map"] = tempo_map
            if cc_max_events is not None:
                kwargs["max_events"] = cc_max_events
            inst.control_changes.extend(curve.to_midi_cc(chan, int(key[2:]), **kwargs))

    curve = curves.get("bend")
    chan = channel_map.get("bend")
    if curve and chan is not None:
        inst = ensure_instrument_for_channel(pm, chan)
        if write_rpn and not any(
            cc.number in {101, 100, 6, 38} for cc in inst.control_changes
        ):
            write_bend_range_rpn(inst, bend_range_semitones)
        kwargs = {"range_semitones": bend_range_semitones}
        if curve.domain == "beats":
            kwargs["tempo_map"] = tempo_map
        if bend_max_events is not None:
            kwargs["max_events"] = bend_max_events
        inst.pitch_bends.extend(curve.to_pitch_bend(chan, **kwargs))

    return pm
