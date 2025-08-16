"""Lightweight CLI to render control curves onto a MIDI file."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import pretty_midi

try:  # optional
    from ruamel.yaml import YAML  # type: ignore
except Exception:  # pragma: no cover
    YAML = None  # type: ignore

from .apply_controls import apply_controls
from .controls_spline import ControlCurve


def _load_curves(
    desc: dict, default_domain: str, default_sr: float
) -> dict[str, ControlCurve]:
    curves: dict[str, ControlCurve] = {}
    for name, spec in desc.items():
        knots = spec.get("knots")
        if not isinstance(knots, list):
            continue
        times = [float(t) for t, _ in knots]
        values = [float(v) for _, v in knots]
        domain = spec.get("domain", default_domain)
        sr = float(spec.get("sample_rate_hz", spec.get("resolution_hz", default_sr)))
        curves[name] = ControlCurve(times, values, domain=domain, sample_rate_hz=sr)
    return curves


def main(argv: list[str] | None = None) -> pretty_midi.PrettyMIDI:
    parser = argparse.ArgumentParser(description="Render control curves onto MIDI")
    parser.add_argument("in_mid")
    parser.add_argument("--out", default="out.mid")
    parser.add_argument("--curves", required=True, help="JSON file describing curves")
    parser.add_argument("--apply", default="bend,cc11,cc64")
    parser.add_argument("--domain", choices=["time", "beats"], default="time")
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--resolution-hz", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--controls-total-max-events", type=int)
    parser.add_argument("--cc-max-events", type=int)
    parser.add_argument("--bend-max-events", type=int)
    parser.add_argument("--value-eps", type=float, default=1e-6)
    parser.add_argument("--time-eps", type=float, default=1e-9)
    parser.add_argument("--bend-range", type=float, default=2.0)
    parser.add_argument("--lsb-mode", choices=["128th", "cents"], default="128th")
    parser.add_argument("--coarse-only", action="store_true")
    parser.add_argument("--rpn-at", type=float, default=0.0)
    parser.add_argument("--no-rpn-null", dest="send_rpn_null", action="store_false")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--time-offset-bend", type=float, default=0.0)
    parser.add_argument("--time-offset-cc11", type=float, default=0.0)
    parser.add_argument("--time-offset-cc64", type=float, default=0.0)
    parser.add_argument(
        "--controls-channel-map",
        default="bend:0,cc11:0,cc64:0",
        help="Mapping like 'bend:0,cc11:0'",
    )
    parser.add_argument("--tempo-json", help="Optional tempo map JSON")
    args = parser.parse_args(argv)
    if args.quiet:
        logging.basicConfig(level=logging.WARNING)
    elif args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    path = Path(args.curves)
    with path.open() as fh:
        if path.suffix in {".yaml", ".yml"}:
            if YAML is None:
                raise RuntimeError("ruamel.yaml required for YAML input")
            desc = YAML(typ="safe").load(fh)
        else:
            desc = json.load(fh)

    if getattr(args, "resolution_hz", None) is not None:
        warnings.warn(
            "--resolution-hz is deprecated; use --sample-rate-hz",
            DeprecationWarning,
            stacklevel=2,
        )
        if args.sample_rate_hz == parser.get_default("sample_rate_hz"):
            args.sample_rate_hz = args.resolution_hz
    curves_all = _load_curves(desc, args.domain, args.sample_rate_hz)
    targets = [t.strip() for t in args.apply.split(",") if t.strip()]
    curves = {k: v for k, v in curves_all.items() if k in targets}

    pm = pretty_midi.PrettyMIDI(args.in_mid)

    def _parse_map(spec: str) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for item in spec.split(","):
            if not item or ":" not in item:
                continue
            key, val = item.split(":", 1)
            ch = int(val)
            if not 0 <= ch <= 15:
                raise ValueError("channel must be in 0..15")
            mapping[key.strip()] = ch
        return mapping

    ch_map = _parse_map(args.controls_channel_map)
    channel_map: dict[int, dict[str, ControlCurve]] = {}
    for name, curve in curves.items():
        ch = ch_map.get(name, 0)
        channel_map.setdefault(ch, {})[name] = curve

    offsets = {
        "bend": args.time_offset_bend,
        "cc11": args.time_offset_cc11,
        "cc64": args.time_offset_cc64,
    }
    tempo_map = None
    if args.tempo_json:
        with Path(args.tempo_json).open() as fh:
            tempo_map = json.load(fh)

    apply_controls(
        pm,
        channel_map,
        write_rpn="bend" in curves,
        bend_range_semitones=args.bend_range,
        lsb_mode=args.lsb_mode,
        coarse_only=args.coarse_only,
        cc_max_events=args.cc_max_events,
        bend_max_events=args.bend_max_events,
        value_eps=args.value_eps,
        time_eps=args.time_eps,
        time_offset_map=offsets,
        tempo_map=tempo_map,
        send_rpn_null=getattr(args, "send_rpn_null", True),
        rpn_at=args.rpn_at,
        total_max_events=args.controls_total_max_events,
    )

    total_cc = sum(len(i.control_changes) for i in pm.instruments)
    total_pb = sum(len(i.pitch_bends) for i in pm.instruments)
    print(f"Rendered {total_cc} CC and {total_pb} pitch-bend events")
    if args.out:
        pm.write(args.out)
        print(f"Wrote {args.out}")
    return pm


if __name__ == "__main__":  # pragma: no cover
    main()
