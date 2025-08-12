from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import pretty_midi

from .controls_spline import ControlCurve


class TargetMap(TypedDict, total=False):
    cc11: ControlCurve
    cc64: ControlCurve
    bend: ControlCurve


Routing = dict[int, TargetMap]


TempoFunc = Callable[[float], float]
TempoMap = TempoFunc | dict[int, TempoFunc] | dict[int, dict[str, TempoFunc]]


class EventCaps(TypedDict, total=False):
    cc11: int
    cc64: int
    bend: int


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


def _write_bend_range_rpn(
    inst: pretty_midi.Instrument,
    range_semitones: float,
    *,
    time: float = 0.0,
    rpn_null: bool = False,
) -> None:
    """Emit RPN messages configuring pitch-bend range.

    Parameters
    ----------
    inst:
        Instrument to receive the control changes.
    range_semitones:
        Pitch-bend range in semitones.
    rpn_null:
        If ``True``, emit ``RPN Null`` (101=127, 100=127) after setting the range.
    """

    msb = int(range_semitones)
    frac = range_semitones - msb
    lsb = int(round(frac * 128))
    msb = max(0, min(127, msb))
    lsb = max(0, min(127, lsb))
    inst.control_changes.extend(
        [
            pretty_midi.ControlChange(number=101, value=0, time=time),
            pretty_midi.ControlChange(number=100, value=0, time=time),
            pretty_midi.ControlChange(number=6, value=msb, time=time),
            pretty_midi.ControlChange(number=38, value=lsb, time=time),
        ]
    )
    if rpn_null:
        inst.control_changes.extend(
            [
                pretty_midi.ControlChange(number=101, value=127, time=time),
                pretty_midi.ControlChange(number=100, value=127, time=time),
            ]
        )


def _resolve_tempo_map(
    tempo_map: TempoMap | None, ch: int, target: str
) -> TempoFunc | None:
    """Return tempo-map callable for ``ch``/``target``."""

    if tempo_map is None:
        return None
    if callable(tempo_map):
        return tempo_map
    tm = tempo_map.get(ch)
    if tm is None:
        return None
    if callable(tm):
        return tm
    if isinstance(tm, dict):
        func = tm.get(target)
        if func is None or not callable(func):
            return None
        return func
    raise TypeError("Invalid tempo_map structure")


def apply_controls(
    pm: pretty_midi.PrettyMIDI,
    routing: Routing,
    *,
    bend_range_semitones: float = 2.0,
    write_rpn: bool = True,
    rpn_null: bool = False,
    per_target_max_events: EventCaps | None = None,
    tempo_map: TempoMap | None = None,
    eps_override: dict[str, float] | None = None,
) -> pretty_midi.PrettyMIDI:
    """Render ``routing`` onto ``pm``.

    Parameters
    ----------
    pm:
        Destination MIDI object.
    routing:
        Mapping of MIDI channels to control curves.
    bend_range_semitones:
        Pitch-bend range in semitones.
    write_rpn:
        When ``True``, configure pitch-bend range via RPN messages.
    rpn_null:
        Emit ``RPN Null`` after the range when ``True``.
    per_target_max_events:
        Optional maximum event counts for each target.
    """

    per_target_max_events = per_target_max_events or {}
    eps_override = eps_override or {}
    for ch, curves in routing.items():
        if not 0 <= ch <= 15:
            raise ValueError("MIDI channel must be in 0..15")
        inst = ensure_instrument_for_channel(pm, ch)

        # Control changes
        for key, num in (("cc11", 11), ("cc64", 64)):
            curve = curves.get(key)
            if curve is None:
                continue
            max_events = per_target_max_events.get(key)
            tm = _resolve_tempo_map(tempo_map, ch, key)
            orig_eps = curve.eps_cc
            curve.eps_cc = eps_override.get(key, orig_eps)
            events = curve.to_midi_cc(
                ch,
                num,
                max_events=max_events,
                tempo_map=tm,
            )
            curve.eps_cc = orig_eps
            inst.control_changes.extend(events)

        # Pitch bend
        bend_curve = curves.get("bend")
        if bend_curve is not None:
            max_events = per_target_max_events.get("bend")
            tm = _resolve_tempo_map(tempo_map, ch, "bend")
            orig_eps = bend_curve.eps_bend
            bend_curve.eps_bend = eps_override.get("bend", orig_eps)
            pb_events = bend_curve.to_pitch_bend(
                ch,
                range_semitones=bend_range_semitones,
                max_events=max_events,
                tempo_map=tm,
            )
            bend_curve.eps_bend = orig_eps
            if pb_events:
                first_time = pb_events[0].time
                if (
                    write_rpn
                    and getattr(inst, "_bend_range_written", None)
                    != bend_range_semitones
                ):
                    _write_bend_range_rpn(
                        inst,
                        bend_range_semitones,
                        time=max(0.0, first_time - 0.005),
                        rpn_null=rpn_null,
                    )
                    inst._bend_range_written = bend_range_semitones
                inst.pitch_bends.extend(pb_events)

        inst.control_changes.sort(key=lambda cc: cc.time)
        inst.pitch_bends.sort(key=lambda pb: pb.time)

    return pm


def _load_curve_from_json(path: Path, target: str) -> ControlCurve:
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Curve file {path} must contain a JSON object")
    domain = data.get("domain", "time")
    knots = data.get("knots")
    if not isinstance(knots, list):
        raise ValueError(f"Curve file {path} missing 'knots' list")
    try:
        kts = [(float(t), float(v)) for t, v in knots]
    except Exception as exc:  # pragma: no cover - validation
        raise ValueError(f"Invalid knots in {path}: {exc}") from exc
    units = data.get("units", "semitones")
    return ControlCurve(target=target, domain=domain, knots=kts, units=units)


def _load_tempo_func(spec: str) -> TempoFunc:
    module_path, func_name = spec.split(":", 1)
    module_path = str(module_path)
    mod_name = Path(module_path).stem
    spec_obj = importlib.util.spec_from_file_location(mod_name, module_path)
    if spec_obj is None or spec_obj.loader is None:
        raise ValueError(f"Cannot load tempo-map module {module_path}")
    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)
    func = getattr(module, func_name, None)
    if not callable(func):
        raise ValueError(f"tempo-map function {func_name} not found in {module_path}")
    return func


def main(argv: list[str] | None = None) -> pretty_midi.PrettyMIDI:
    parser = argparse.ArgumentParser(description="Apply control curves to MIDI")
    parser.add_argument("in_mid")
    parser.add_argument("routing_json")
    parser.add_argument("--out", default="out.mid")
    parser.add_argument("--bend-range-semitones", type=float, default=2.0)
    parser.add_argument("--rpn-null", action="store_true")
    parser.add_argument("--max-bend", type=int)
    parser.add_argument("--max-cc11", type=int)
    parser.add_argument("--max-cc64", type=int)
    parser.add_argument("--tempo-map")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        pm = pretty_midi.PrettyMIDI(args.in_mid)
    except TypeError:  # pragma: no cover - stub compatibility
        pm = pretty_midi.PrettyMIDI()

    with Path(args.routing_json).open() as fh:
        routing_desc = json.load(fh)
    if not isinstance(routing_desc, dict):
        raise ValueError("routing JSON must be an object mapping channels")
    routing: Routing = {}
    for ch_str, targets in routing_desc.items():
        ch = int(ch_str)
        mapping: TargetMap = {}
        for tgt, file_path in targets.items():
            mapping[tgt] = _load_curve_from_json(Path(file_path), tgt)
        routing[ch] = mapping

    caps = {
        k: v
        for k, v in {
            "bend": args.max_bend,
            "cc11": args.max_cc11,
            "cc64": args.max_cc64,
        }.items()
        if v is not None
    }

    tempo_func: TempoFunc | None = None
    if args.tempo_map is not None:
        tempo_func = _load_tempo_func(args.tempo_map)

    apply_controls(
        pm,
        routing,
        bend_range_semitones=args.bend_range_semitones,
        rpn_null=args.rpn_null,
        per_target_max_events=caps or None,
        tempo_map=tempo_func,
    )

    if not args.dry_run and hasattr(pm, "write"):
        pm.write(args.out)

    return pm


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
