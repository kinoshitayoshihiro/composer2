"""Library for rendering control curves onto PrettyMIDI objects.

This module intentionally exposes a small surface area; command line parsing
is handled by :mod:`utilities.apply_controls_cli`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

try:  # optional dependency
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover - fallback stub in tests
    from tests._stubs import pretty_midi  # type: ignore

from .controls_spline import ControlCurve

logger = logging.getLogger(__name__)


def ensure_instrument_for_channel(
    pm: pretty_midi.PrettyMIDI, ch: int
) -> pretty_midi.Instrument:
    """Return an instrument routed to MIDI channel ``ch``.

    Events in PrettyMIDI carry no channel attribute; instead we keep a
    dedicated instrument per channel named ``"channel{ch}"``.
    """

    name = f"channel{ch}"
    for inst in pm.instruments:
        if inst.name == name:
            return inst
    inst = pretty_midi.Instrument(program=0, name=name)
    pm.instruments.append(inst)
    return inst


def _sort_events(inst: pretty_midi.Instrument) -> None:
    """Sort control and bend events so that RPNs come first."""

    events: list[tuple[float, int, str, object]] = []
    for cc in inst.control_changes:
        pr = 0 if cc.number in {101, 100, 6, 38} else 1
        events.append((cc.time, pr, "cc", cc))
    for pb in inst.pitch_bends:
        events.append((pb.time, 2, "pb", pb))
    events.sort(key=lambda e: (e[0], e[1]))
    inst.control_changes = [e[3] for e in events if e[2] == "cc"]
    inst.pitch_bends = [e[3] for e in events if e[2] == "pb"]


def write_bend_range_rpn(
    inst: pretty_midi.Instrument, range_semitones: float, *, at_time: float = 0.0
) -> None:
    """Emit bend-range RPN (101,100→6,38) onto ``inst`` at ``at_time``.

    The sequence is written at most once per instrument.
    The LSB encodes the fractional semitone portion in 1/128th steps.
    """

    msb = int(range_semitones)
    lsb = int(round((range_semitones - msb) * 128))
    lsb = max(0, min(127, lsb))

    for i in range(len(inst.control_changes) - 3):
        a, b, c, d = inst.control_changes[i : i + 4]
        if (a.number, a.value) == (101, 0) and (b.number, b.value) == (100, 0) and c.number == 6:
            old_msb = c.value
            old_lsb = 0
            if d.number == 38:
                old_lsb = d.value
            inst._rpn_written = True  # type: ignore[attr-defined]
            inst._rpn_range = old_msb + old_lsb / 128.0  # type: ignore[attr-defined]
            if old_msb == msb and old_lsb == lsb:
                return
            del inst.control_changes[i : i + 4]
            break

    if getattr(inst, "_rpn_written", False) and getattr(inst, "_rpn_range", None) == range_semitones:
        return

    t = float(at_time)
    inst.control_changes.extend(
        [
            pretty_midi.ControlChange(number=101, value=0, time=t),
            pretty_midi.ControlChange(number=100, value=0, time=t),
            pretty_midi.ControlChange(number=6, value=msb, time=t),
            pretty_midi.ControlChange(number=38, value=lsb, time=t),
        ]
    )
    inst._rpn_written = True  # type: ignore[attr-defined]
    inst._rpn_range = range_semitones  # type: ignore[attr-defined]


_CC_MAP = {"cc11": 11, "cc64": 64}


def apply_controls(
    pm: pretty_midi.PrettyMIDI,
    curves_by_channel: Mapping[int, Mapping[str, ControlCurve]],
    *,
    bend_range_semitones: float = 2.0,
    write_rpn: bool = True,
    rpn_at: float = 0.0,
    sample_rate_hz: Mapping[str, float] | None = None,
    max_events: Mapping[str, int] | None = None,
    value_eps: float = 1e-6,
    time_eps: float = 1e-9,
    tempo_map: (
        float | list[tuple[float, float]] | Callable[[float], float] | None
    ) = None,
    simplify_mode: str = "rdp",
    total_max_events: int | None = None,
) -> pretty_midi.PrettyMIDI:
    """Render ``curves_by_channel`` onto ``pm``.

    Parameters mirror :meth:`ControlCurve.to_midi_cc` and
    :meth:`ControlCurve.to_pitch_bend` and are forwarded accordingly.
    """

    sr_map = sample_rate_hz or {"cc11": 30.0, "cc64": 30.0, "bend": 120.0}
    max_map = max_events or {}

    for ch, mapping in curves_by_channel.items():
        inst = ensure_instrument_for_channel(pm, ch)
        for name, curve in mapping.items():
            if name == "bend":
                curve.to_pitch_bend(
                    inst,
                    bend_range_semitones=bend_range_semitones,
                    sample_rate_hz=sr_map.get(name),
                    max_events=max_map.get(name),
                    value_eps=value_eps,
                    time_eps=time_eps,
                    tempo_map=tempo_map,
                    simplify_mode=simplify_mode,
                )
                if write_rpn and not getattr(inst, "_rpn_written", False):
                    first_pb = min(pb.time for pb in inst.pitch_bends)
                    t = min(rpn_at, first_pb - 1e-9)
                    write_bend_range_rpn(inst, bend_range_semitones, at_time=t)
            elif name in _CC_MAP:
                cc_num = _CC_MAP[name]
                sr = sr_map.get(name)
                curve.to_midi_cc(
                    inst,
                    cc_num,
                    sample_rate_hz=sr,
                    max_events=max_map.get(name),
                    value_eps=value_eps,
                    time_eps=time_eps,
                    tempo_map=tempo_map,
                    simplify_mode=simplify_mode,
                )
        _sort_events(inst)

    if total_max_events:
        _enforce_total_cap(pm, total_max_events)

    return pm


def _downsample_keep_endpoints(events: list, target: int) -> list:
    if target >= len(events) or target <= 0:
        return list(events)
    idxs = [round(i * (len(events) - 1) / (target - 1)) for i in range(target)]
    seen: set[int] = set()
    out = []
    for idx in idxs:
        if idx not in seen:
            out.append(events[idx])
            seen.add(idx)
    return out


def _enforce_total_cap(pm: pretty_midi.PrettyMIDI, cap: int) -> None:
    lists: list[tuple[pretty_midi.Instrument, str, list]] = []
    total = 0
    for inst in pm.instruments:
        cc11 = [c for c in inst.control_changes if c.number == 11]
        cc64 = [c for c in inst.control_changes if c.number == 64]
        bend = list(inst.pitch_bends)
        lists.extend([(inst, "cc11", cc11), (inst, "cc64", cc64), (inst, "bend", bend)])
        total += len(cc11) + len(cc64) + len(bend)
    if cap <= 0 or total <= cap:
        return
    ratio = cap / total
    entries: list[dict[str, object]] = []
    for inst, name, lst in lists:
        n = len(lst)
        target = int(round(n * ratio))
        if n > 1 and target < 2:
            target = 2
        kept = _downsample_keep_endpoints(lst, target)
        entries.append({"inst": inst, "name": name, "orig": lst, "kept": kept})
    total_new = sum(len(e["kept"]) for e in entries)
    while total_new > cap:
        entry = max(entries, key=lambda e: len(e["kept"]))
        if len(entry["kept"]) <= 2:
            break
        entry["kept"] = _downsample_keep_endpoints(
            entry["kept"], len(entry["kept"]) - 1
        )
        total_new = sum(len(e["kept"]) for e in entries)
    new_cc: dict[pretty_midi.Instrument, list[pretty_midi.ControlChange]] = {}
    new_pb: dict[pretty_midi.Instrument, list[pretty_midi.PitchBend]] = {}
    for e in entries:
        inst = e["inst"]
        name = e["name"]
        kept = e["kept"]
        if name == "bend":
            new_pb[inst] = kept
        else:
            new_cc.setdefault(inst, []).extend(kept)
    for inst in pm.instruments:
        if inst in new_pb:
            inst.pitch_bends = new_pb[inst]
        if inst in new_cc:
            others = [c for c in inst.control_changes if c.number not in {11, 64}]
            inst.control_changes = others + new_cc[inst]
        _sort_events(inst)


__all__ = [
    "apply_controls",
    "ensure_instrument_for_channel",
    "write_bend_range_rpn",
]


def _parse_kv_pairs(spec: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in spec.split(","):
        if not part:
            continue
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        try:
            out[key.strip()] = float(val.strip())
        except ValueError:
            continue
    return out


def _load_curve(path: str) -> ControlCurve:
    import json

    with open(path) as fh:
        data = json.load(fh)
    domain = data.get("domain", "time")
    knots = data.get("knots", [])
    times = [float(t) for t, _ in knots]
    values = [float(v) for _, v in knots]
    sr = data.get("sample_rate_hz") or data.get("resolution_hz")
    return ControlCurve(times, values, domain=domain, sample_rate_hz=sr or 0.0)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Apply control curves to MIDI")
    parser.add_argument("in_mid")
    parser.add_argument("routing_json")
    parser.add_argument("--out")
    parser.add_argument("--bend-range-semitones", type=float, default=2.0)
    parser.add_argument("--write-rpn", action="store_true")
    parser.add_argument("--rpn-at", type=float, default=0.0)
    parser.add_argument("--sample-rate-hz", default="")
    parser.add_argument("--max-bend", type=int)
    parser.add_argument("--max-cc11", type=int)
    parser.add_argument("--max-cc64", type=int)
    parser.add_argument("--value-eps", type=float, default=1e-6)
    parser.add_argument("--time-eps", type=float, default=1e-9)
    parser.add_argument("--tempo-map")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    pm = pretty_midi.PrettyMIDI(args.in_mid)
    routing_path = Path(args.routing_json)
    with routing_path.open() as fh:
        routing = json.load(fh)

    curves_by_channel: dict[int, dict[str, ControlCurve]] = {}
    for ch_str, mapping in routing.items():
        ch = int(ch_str)
        ch_curves: dict[str, ControlCurve] = {}
        for tgt, path in mapping.items():
            ch_curves[tgt] = _load_curve(path)
        curves_by_channel[ch] = ch_curves

    sr_map = _parse_kv_pairs(args.sample_rate_hz) if args.sample_rate_hz else None
    max_map: dict[str, int] = {}
    if args.max_bend is not None:
        max_map["bend"] = args.max_bend
    if args.max_cc11 is not None:
        max_map["cc11"] = args.max_cc11
    if args.max_cc64 is not None:
        max_map["cc64"] = args.max_cc64
    tempo_map = None
    if args.tempo_map:
        tm_path = Path(args.tempo_map)
        if tm_path.exists():
            with tm_path.open() as fh:
                tempo_map = json.load(fh)
        else:
            tempo_map = json.loads(args.tempo_map)

    apply_controls(
        pm,
        curves_by_channel,
        bend_range_semitones=args.bend_range_semitones,
        write_rpn=args.write_rpn,
        rpn_at=args.rpn_at,
        sample_rate_hz=sr_map,
        max_events=max_map or None,
        value_eps=args.value_eps,
        time_eps=args.time_eps,
        tempo_map=tempo_map,
    )

    cc_count = sum(len(inst.control_changes) for inst in pm.instruments)
    bend_count = sum(len(inst.pitch_bends) for inst in pm.instruments)
    print(f"Applied controls: {cc_count} CC, {bend_count} bend events")

    if not args.dry_run:
        out_path = args.out or args.in_mid
        pm.write(out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
