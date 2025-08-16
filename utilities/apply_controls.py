from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
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
    if not 0 <= ch <= 15:
        raise ValueError("MIDI channel must be in 0..15")
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
    at_time: float = 0.0,
    reset: bool = False,
    coarse_only: bool = False,
    lsb_mode: str = "128th",
    send_rpn_null: bool | None = None,
) -> None:
    """Emit an RPN 0,0 sequence to set pitch-bend range for ``inst``.

    MSB (CC#6) encodes the coarse semitone count while LSB (CC#38) encodes
    the remaining fraction in either ``"128th"`` semitone steps or ``"cents"``
    (1/100 semitone) depending on ``lsb_mode``. When ``coarse_only`` is ``True``
    the LSB is omitted. If ``reset`` is ``True`` an RPN Null
    (``101=127``, ``100=127``) is emitted after data entry to clear the active RPN.

    The function is idempotent at time ``t`` and will avoid duplicating
    an identical RPN sequence for the same ``bend_range_semitones``.
    """
    if lsb_mode not in {"128th", "cents"}:
        raise ValueError("lsb_mode must be '128th' or 'cents'")
    if send_rpn_null is not None:
        warnings.warn(
            "send_rpn_null is deprecated; use reset",
            DeprecationWarning,
            stacklevel=2,
        )
        reset = send_rpn_null

    ch = getattr(inst, "_control_channel", None)
    if ch is not None and not 0 <= ch <= 15:
        raise ValueError("MIDI channel must be in 0..15")

    msb = int(max(0, min(127, int(bend_range_semitones))))
    frac = bend_range_semitones - msb
    if not coarse_only:
        scale = 100 if lsb_mode == "cents" else 128
        lsb = int(max(0, min(127, round(frac * scale))))
        use_lsb = True
    else:
        lsb = 0
        use_lsb = False

    cc = inst.control_changes
    cc_sorted = sorted(cc, key=lambda c: c.time)
    first_bend_time = min((pb.time for pb in inst.pitch_bends), default=float("inf"))
    existing_time: float | None = None
    existing_vals: tuple[int, int] | None = None
    existing_null = False
    for i in range(len(cc_sorted) - 1):
        n1, v1, t1 = (
            cc_sorted[i].number,
            cc_sorted[i].value,
            cc_sorted[i].time,
        )
        n2, v2 = cc_sorted[i + 1].number, cc_sorted[i + 1].value
        if t1 < first_bend_time and (n1, v1, n2, v2) == (101, 127, 100, 127):
            existing_null = True
        if i >= len(cc_sorted) - 2:
            continue
        n1, v1, t1 = (
            cc_sorted[i].number,
            cc_sorted[i].value,
            cc_sorted[i].time,
        )
        n2, v2 = (
            cc_sorted[i + 1].number,
            cc_sorted[i + 1].value,
        )
        n3, v3, t3 = (
            cc_sorted[i + 2].number,
            cc_sorted[i + 2].value,
            cc_sorted[i + 2].time,
        )
        if t3 >= first_bend_time:
            break
        if (n1, v1) == (101, 0) and (n2, v2) == (100, 0) and n3 == 6:
            msb_old = v3
            lsb_old = 0
            j = i + 3
            if use_lsb and j < len(cc_sorted) and cc_sorted[j].number == 38:
                lsb_old = cc_sorted[j].value
            existing_time = t1
            existing_vals = (msb_old, lsb_old)
            break

    if existing_vals is not None:
        msb_old, lsb_old = existing_vals
        if msb_old == msb and (coarse_only or lsb_old == lsb):
            inst._bend_range_written = bend_range_semitones  # type: ignore[attr-defined]
            return
        logging.info(
            "Overwriting existing bend-range RPN (%s,%s) on channel %s",
            msb_old,
            lsb_old,
            ch,
        )
        cc[:] = [
            c
            for c in cc
            if c.time != existing_time or c.number not in {101, 100, 6, 38}
        ]

    t = existing_time if existing_time is not None else at_time
    if first_bend_time < float("inf") and t >= first_bend_time:
        t = max(0.0, first_bend_time - 1e-9)
    cc.append(pretty_midi.ControlChange(number=101, value=0, time=float(t)))
    cc.append(pretty_midi.ControlChange(number=100, value=0, time=float(t)))
    cc.append(pretty_midi.ControlChange(number=6, value=msb, time=float(t)))
    if use_lsb:
        cc.append(pretty_midi.ControlChange(number=38, value=lsb, time=float(t)))
    if reset and not existing_null:
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
    rpn_reset: bool = False,
    rpn_coarse_only: bool = False,
    lsb_mode: str = "128th",
    ensure_zero_at_edges: bool = True,
    sample_rate_hz: Mapping[str, float] | None = None,
    cc_max_events: int | None = None,
    bend_max_events: int | None = None,
    value_eps: float = 1e-6,
    time_eps: float = 1e-9,
    time_offset_map: Mapping[str, float] | None = None,
    tempo_map: (
        float | list[tuple[float, float]] | Callable[[float], float] | None
    ) = None,
    fold_halves: bool = False,
    rpn_at: float = 0.0,
    total_max_events: int | None = None,
) -> None:
    """Apply ``curves_by_channel`` to ``pm`` grouped by MIDI channel."""
    if lsb_mode not in {"128th", "cents"}:
        raise ValueError("lsb_mode must be '128th' or 'cents'")

    if (
        tempo_map is not None
        and not callable(tempo_map)
        and not isinstance(tempo_map, int | float)
    ):
        events = (
            list(tempo_map.items()) if isinstance(tempo_map, dict) else list(tempo_map)
        )
        last = -float("inf")
        for beat, bpm in events:
            beat_f = float(beat)
            bpm_f = float(bpm)
            if not math.isfinite(beat_f) or not math.isfinite(bpm_f) or bpm_f <= 0:
                raise ValueError("tempo-map bpm must be finite and >0")
            if beat_f <= last:
                raise ValueError("tempo-map beats must be strictly increasing")
            last = beat_f
        tempo_map = events

    written_rpn: set[int] = set()
    for ch, targets in curves_by_channel.items():
        if not 0 <= ch <= 15:
            raise ValueError("MIDI channel must be in 0..15")

        inst = ensure_instrument_for_channel(pm, ch)

        if write_rpn and ch not in written_rpn:
            write_bend_range_rpn(
                inst,
                bend_range_semitones,
                at_time=rpn_at,
                reset=rpn_reset,
                coarse_only=rpn_coarse_only,
                lsb_mode=lsb_mode,
            )
            written_rpn.add(ch)

        for name, curve in targets.items():
            if sample_rate_hz is not None:
                sr = sample_rate_hz.get(name)
                if sr is None and name.startswith("cc"):
                    sr = sample_rate_hz.get("cc")
                if sr is not None:
                    curve.sample_rate_hz = float(sr)
            if not curve.sample_rate_hz:
                curve.sample_rate_hz = 120.0 if name == "bend" else 30.0
            ve = value_eps
            te = time_eps
            max_ev = bend_max_events if name == "bend" else cc_max_events
            offset = (time_offset_map or {}).get(name, 0.0)
            if name == "bend":
                curve.ensure_zero_at_edges = ensure_zero_at_edges
                kwargs: dict[str, object] = {
                    "bend_range_semitones": bend_range_semitones,
                    "max_events": max_ev,
                    "value_eps": ve,
                    "time_eps": te,
                    "time_offset": offset,
                    "fold_halves": fold_halves,
                }
                if tempo_map is not None:
                    kwargs["tempo_map"] = tempo_map
                curve.to_pitch_bend(inst, **kwargs)  # type: ignore[arg-type]
            else:
                cc_num = _CC_MAP.get(name)
                if cc_num is None:
                    logging.warning("Unknown control target %s", name)
                    continue
                kwargs = {
                    "max_events": max_ev,
                    "value_eps": ve,
                    "time_eps": te,
                    "time_offset": offset,
                    "channel": ch,
                    "fold_halves": fold_halves,
                }
                if tempo_map is not None:
                    kwargs["tempo_map"] = tempo_map
                curve.to_midi_cc(inst, cc_num, **kwargs)  # type: ignore[arg-type]

        inst.control_changes.sort(key=lambda c: (c.time, c.number, c.value))
        inst.pitch_bends.sort(key=lambda b: (b.time, b.pitch))

    if total_max_events is not None:
        _enforce_total_cap(pm, total_max_events)


# ----------------------------- CLI helpers ---------------------------------


def _enforce_total_cap(pm: pretty_midi.PrettyMIDI, cap: int) -> None:
    events: dict[str, list[tuple[pretty_midi.Instrument, object]]] = {
        "cc11": [],
        "cc64": [],
        "bend": [],
    }
    for inst in pm.instruments:
        for cc in inst.control_changes:
            if cc.number == 11:
                events["cc11"].append((inst, cc))
            elif cc.number == 64:
                events["cc64"].append((inst, cc))
        for pb in inst.pitch_bends:
            events["bend"].append((inst, pb))
    total = sum(len(v) for v in events.values())
    if total <= cap:
        return
    factor = cap / total
    counts = {
        k: (len(v) if len(v) < 2 else max(2, int(len(v) * factor)))
        for k, v in events.items()
    }
    while sum(counts.values()) > cap:
        key = max(counts, key=lambda k: counts[k])
        if counts[key] > 2:
            counts[key] -= 1
        else:
            break
    for name, lst in events.items():
        keep = counts[name]
        if len(lst) <= keep:
            continue
        step = (len(lst) - 1) / (keep - 1) if keep > 1 else float("inf")
        keep_idx = {int(round(i * step)) for i in range(keep)}
        for idx, (inst, ev) in enumerate(list(lst)):
            if idx not in keep_idx:
                if isinstance(ev, pretty_midi.PitchBend):
                    inst.pitch_bends.remove(ev)
                else:
                    inst.control_changes.remove(ev)


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
    sr = data.get("sample_rate_hz")
    res = data.get("resolution_hz")
    return ControlCurve(
        times, values, domain=domain, sample_rate_hz=sr or 0.0, resolution_hz=res
    )


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
        "--rpn-reset", dest="rpn_reset", action="store_true", default=False
    )
    parser.add_argument("--no-rpn-reset", dest="rpn_reset", action="store_false")
    parser.add_argument(
        "--rpn-null", dest="rpn_reset", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--no-rpn-null", dest="rpn_reset", action="store_false", help=argparse.SUPPRESS
    )
    parser.add_argument("--coarse-only", dest="rpn_coarse_only", action="store_true")
    parser.add_argument(
        "--lsb-mode",
        choices=["128th", "cents"],
        default="128th",
        help="Fractional bend-range encoding for CC#38",
    )
    parser.add_argument(
        "--ensure-zero-at-edges", dest="ensure_zero", action="store_true", default=True
    )
    parser.add_argument(
        "--no-ensure-zero-at-edges", dest="ensure_zero", action="store_false"
    )
    parser.add_argument("--sample-rate-hz", help="Overrides like 'bend=80,cc=30'")
    parser.add_argument("--max-bend", type=int)
    parser.add_argument("--max-cc11", type=int)
    parser.add_argument("--max-cc64", type=int)
    parser.add_argument("--value-eps", type=float, default=1e-6)
    parser.add_argument("--time-eps", type=float, default=1e-9)
    parser.add_argument(
        "--tempo-map",
        help="Tempo map as FILE.py:FUNC or JSON string/list for beats-domain curves",
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

    sr_map: dict[str, float] | None = None
    if args.sample_rate_hz:
        sr_map = {}
        for part in args.sample_rate_hz.split(","):
            if not part:
                continue
            key, val = part.split("=")
            sr_map[key.strip()] = float(val)

    max_events = {
        k: v
        for k, v in {
            "cc11": args.max_cc11,
            "cc64": args.max_cc64,
            "bend": args.max_bend,
        }.items()
        if v is not None
    }

    tempo_obj: TempoFunc | list[tuple[float, float]] | None = None
    if args.tempo_map is not None:
        spec = args.tempo_map
        try:
            tempo_obj = json.loads(spec)
        except json.JSONDecodeError:
            path = Path(spec)
            if path.exists():
                tempo_obj = json.load(path.open())
            else:
                tempo_obj = _load_tempo_func(spec)

    apply_controls(
        pm,
        curves_by_channel,
        write_rpn=args.write_rpn,
        bend_range_semitones=args.bend_range_semitones,
        rpn_reset=args.rpn_reset,
        rpn_coarse_only=args.rpn_coarse_only,
        lsb_mode=args.lsb_mode,
        ensure_zero_at_edges=args.ensure_zero,
        sample_rate_hz=sr_map,
        max_events=max_events,
        value_eps=args.value_eps,
        time_eps=args.time_eps,
        tempo_map=tempo_obj,
    )

    total_cc = sum(len(i.control_changes) for i in pm.instruments)
    total_pb = sum(len(i.pitch_bends) for i in pm.instruments)
    print(f"Applied controls: {total_cc} CC, {total_pb} bend events")

    if not args.dry_run and hasattr(pm, "write"):
        pm.write(args.out)

    return pm


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
