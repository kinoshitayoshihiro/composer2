from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from collections.abc import Callable, Mapping
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
# CLI helper type (not used by the core API)
TempoMap = TempoFunc | dict[int, TempoFunc] | dict[int, dict[str, TempoFunc]]


class EventCaps(TypedDict, total=False):
    cc11: int
    cc64: int
    bend: int


def ensure_instrument_for_channel(
    pm: pretty_midi.PrettyMIDI, ch: int
) -> pretty_midi.Instrument:
    """Return an instrument in ``pm`` used for control events on ``ch``.

    Preference order:
      1) An instrument already tagged for this control channel (private attr).
      2) An existing "channel{ch}" (or suffixed) instrument that has no CC/PB yet.
      3) Create a fresh instrument named "channel{ch}" (or with a numeric suffix
         if the base name is taken).
    """
    # 1) Already tagged for this control channel
    for inst in pm.instruments:
        if getattr(inst, "_control_channel", None) == ch:
            return inst

    # 2) Reuse a clean candidate with matching base name
    name_base = f"channel{ch}"
    candidates = [
        inst
        for inst in pm.instruments
        if inst.name == name_base or inst.name.startswith(f"{name_base}_")
    ]
    for inst in candidates:
        if not inst.control_changes and not inst.pitch_bends:
            inst._control_channel = ch  # type: ignore[attr-defined]
            return inst

    # 3) Create a new uniquely-named instrument
    idx = 0
    name = name_base
    while any(i.name == name for i in pm.instruments):
        idx += 1
        name = f"{name_base}_{idx}"
    inst = pretty_midi.Instrument(program=0, name=name)
    inst._control_channel = ch  # type: ignore[attr-defined]
    pm.instruments.append(inst)
    return inst


def write_bend_range_rpn(
    inst: pretty_midi.Instrument,
    bend_range_semitones: float,
    *,
    t: float = 0.0,
    send_rpn_null: bool = True,
    coarse_only: bool = False,
    lsb_mode: str = "128th",
) -> None:
    """Emit an RPN 0,0 sequence to set pitch-bend range for ``inst``.

    MSB (CC#6) encodes the coarse semitone count while LSB (CC#38) encodes
    the remaining fraction in either ``"128th"`` semitones (default) or
    ``"cents"`` depending on ``lsb_mode``. When ``coarse_only`` is ``True``
    the LSB is omitted. If ``send_rpn_null`` is ``True`` an RPN Null
    (``101=127``, ``100=127``) is emitted after data entry to clear the active RPN.

    The function is idempotent at time ``t`` and will avoid duplicating
    an identical RPN sequence for the same ``bend_range_semitones``.
    """
    if lsb_mode not in {"128th", "cents"}:
        raise ValueError("lsb_mode must be '128th' or 'cents'")

    # Avoid duplicating the same bend range for the instrument.
    if getattr(inst, "_bend_range_written", None) == bend_range_semitones:
        return

    msb = int(max(0, min(127, int(bend_range_semitones))))
    frac = bend_range_semitones - msb
    if lsb_mode == "cents":
        lsb = int(max(0, min(127, round(frac * 100))))
    else:  # "128th"
        lsb = int(max(0, min(127, round(frac * 128))))

    # If an identical sequence is already present at time t, do nothing.
    eps = 1e-9
    cc = inst.control_changes
    existing = [c for c in cc if abs(c.time - t) <= eps]
    if (
        any(c.number == 101 and c.value == 0 for c in existing)
        and any(c.number == 100 and c.value == 0 for c in existing)
        and any(c.number == 6 and c.value == msb for c in existing)
        and (
            coarse_only
            or any(c.number == 38 and c.value == lsb for c in existing)
            or not any(c.number == 38 for c in existing)
        )
    ):
        inst._bend_range_written = bend_range_semitones  # type: ignore[attr-defined]
        return

    cc.append(pretty_midi.ControlChange(number=101, value=0, time=float(t)))
    cc.append(pretty_midi.ControlChange(number=100, value=0, time=float(t)))
    cc.append(pretty_midi.ControlChange(number=6, value=msb, time=float(t)))
    if not coarse_only:
        cc.append(pretty_midi.ControlChange(number=38, value=lsb, time=float(t)))
    if send_rpn_null:
        cc.append(pretty_midi.ControlChange(number=101, value=127, time=float(t)))
        cc.append(pretty_midi.ControlChange(number=100, value=127, time=float(t)))
    inst._bend_range_written = bend_range_semitones  # type: ignore[attr-defined]


_CC_MAP: dict[str, int] = {"cc11": 11, "cc64": 64}


def apply_controls(
    pm: pretty_midi.PrettyMIDI,
    curves_by_channel: Mapping[int, Mapping[str, ControlCurve]],
    *,
    write_rpn: bool = False,
    bend_range_semitones: float = 2.0,
    send_rpn_null: bool = True,
    lsb_mode: str = "128th",
    cc_max_events: int | None = None,
    bend_max_events: int | None = None,
    value_eps: float = 1e-6,
    time_eps: float = 1e-9,
    tempo_map: float | list[tuple[float, float]] | Callable[[float], float] | None = None,
) -> None:
    """Apply ``curves_by_channel`` to ``pm`` grouped by MIDI channel.

    Parameters
    ----------
    pm
        Target PrettyMIDI object to which events will be written.
    curves_by_channel
        ``{channel: {target_name: ControlCurve, ...}, ...}``
        where ``target_name`` is one of ``"bend"``, ``"cc11"``, ``"cc64"``.
    write_rpn
        If ``True``, emit an RPN(0,0) bend-range configuration before any bends.
    bend_range_semitones
        Pitch-bend range in semitones for encoding bend curves.
    send_rpn_null
        If ``True``, emit RPN Null (101=127,100=127) after data entry.
    lsb_mode
        How to encode fractional bend range into CC#38: ``"128th"`` or ``"cents"``.
    cc_max_events, bend_max_events
        Optional caps on emitted event counts for CC and bend curves.
    value_eps, time_eps
        Epsilon thresholds used by curve rendering to deduplicate near-identical
        consecutive events.
    tempo_map
        Either a constant BPM (float), a list of ``(beat, bpm)`` tuples, a
        ``Callable[beat]->bpm``, or ``None`` (seconds domain).
    """
    if lsb_mode not in {"128th", "cents"}:
        raise ValueError("lsb_mode must be '128th' or 'cents'")

    for ch, targets in curves_by_channel.items():
        if not 0 <= ch <= 15:
            raise ValueError("MIDI channel must be in 0..15")

        inst = ensure_instrument_for_channel(pm, ch)

        if write_rpn:
            write_bend_range_rpn(
                inst,
                bend_range_semitones,
                t=0.0,
                send_rpn_null=send_rpn_null,
                lsb_mode=lsb_mode,
            )

        for name, curve in targets.items():
            if name == "bend":
                kwargs: dict[str, object] = {
                    "bend_range_semitones": bend_range_semitones,
                    "max_events": bend_max_events,
                    "value_eps": value_eps,
                    "time_eps": time_eps,
                }
                if tempo_map is not None:
                    kwargs["tempo_map"] = tempo_map
                # Writes into inst in-place
                curve.to_pitch_bend(inst, **kwargs)  # type: ignore[arg-type]
            else:
                cc_num = _CC_MAP.get(name)
                if cc_num is None:
                    warnings.warn(f"Unknown control target {name}")
                    continue
                kwargs = {
                    "max_events": cc_max_events,
                    "value_eps": value_eps,
                    "time_eps": time_eps,
                }
                if tempo_map is not None:
                    kwargs["tempo_map"] = tempo_map
                # Writes into inst in-place
                curve.to_midi_cc(inst, cc_num, **kwargs)  # type: ignore[arg-type]

        # keep deterministic ordering
        inst.control_changes.sort(key=lambda c: c.time)
        inst.pitch_bends.sort(key=lambda b: b.time)


# ----------------------------- CLI helpers ---------------------------------


def _load_curve_from_json(path: Path) -> ControlCurve:
    """Load a curve description JSON and return a ControlCurve(times, values,...).

    Expected JSON schema:
    {
      "domain": "time" | "beats",
      "knots": [[t0, v0], [t1, v1], ...]
    }
    """
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Curve file {path} must contain a JSON object")
    domain = data.get("domain", "time")
    knots = data.get("knots")
    if not isinstance(knots, list) or not knots:
        raise ValueError(f"Curve file {path} missing non-empty 'knots' list")
    try:
        times = [float(t) for t, _ in knots]
        values = [float(v) for _, v in knots]
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Invalid knots in {path}: {exc}") from exc
    return ControlCurve(times, values, domain=domain)


def _load_tempo_func(spec: str) -> TempoFunc:
    """Load a tempo-map callable from 'FILE.py:FUNC'."""
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
    return func  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> pretty_midi.PrettyMIDI:
    """CLI: Apply control curves described in a routing JSON onto a MIDI file."""
    parser = argparse.ArgumentParser(description="Apply control curves to MIDI")
    parser.add_argument("in_mid")
    parser.add_argument("routing_json")
    parser.add_argument("--out", default="out.mid")
    parser.add_argument("--bend-range-semitones", type=float, default=2.0)
    parser.add_argument(
        "--write-rpn", action="store_true", help="Emit RPN(0,0) bend range before bends"
    )
    parser.add_argument(
        "--no-rpn-null",
        dest="send_rpn_null",
        action="store_false",
        default=True,
        help="Do not append RPN Null after range",
    )
    parser.add_argument(
        "--lsb-mode",
        choices=["128th", "cents"],
        default="128th",
        help="Fractional bend-range encoding for CC#38",
    )
    parser.add_argument("--max-bend", type=int)
    parser.add_argument("--max-cc11", type=int)
    parser.add_argument("--max-cc64", type=int)
    parser.add_argument(
        "--tempo-map",
        help="Tempo map callable as FILE.py:FUNC for beats-domain curves",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        pm = pretty_midi.PrettyMIDI(args.in_mid)
    except TypeError:  # pragma: no cover - stub compatibility
        pm = pretty_midi.PrettyMIDI()

    # Load routing JSON: { "0": {"cc11": "expr.json", "bend": "vibrato.json"}, ... }
    with Path(args.routing_json).open() as fh:
        routing_desc = json.load(fh)
    if not isinstance(routing_desc, dict):
        raise ValueError("routing JSON must be an object mapping channels")

    curves_by_channel: dict[int, dict[str, ControlCurve]] = {}
    for ch_str, targets in routing_desc.items():
        ch = int(ch_str)
        mapping: dict[str, ControlCurve] = {}
        if not isinstance(targets, dict):
            raise ValueError(f"routing for channel {ch} must be an object")
        for tgt, file_path in targets.items():
            mapping[tgt] = _load_curve_from_json(Path(file_path))
        curves_by_channel[ch] = mapping

    # Build event caps (shared cap per class of target)
    cc_cap = min(x for x in [args.max_cc11, args.max_cc64] if x is not None) if any(
        v is not None for v in (args.max_cc11, args.max_cc64)
    ) else None
    bend_cap = args.max_bend

    tempo_callable: TempoFunc | None = None
    if args.tempo_map is not None:
        tempo_callable = _load_tempo_func(args.tempo_map)

    apply_controls(
        pm,
        curves_by_channel,
        write_rpn=args.write_rpn,
        bend_range_semitones=args.bend_range_semitones,
        send_rpn_null=args.send_rpn_null,
        lsb_mode=args.lsb_mode,
        cc_max_events=cc_cap,
        bend_max_events=bend_cap,
        tempo_map=tempo_callable,
    )

    if not args.dry_run and hasattr(pm, "write"):
        pm.write(args.out)

    return pm


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
