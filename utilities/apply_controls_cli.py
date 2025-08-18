"""Command line interface for :mod:`utilities.apply_controls`.

The CLI renders control curves described in a JSON/YAML file onto a MIDI file
using PrettyMIDI.  It purposely exposes only a minimal set of flags so the API
remains stable.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

try:  # optional YAML input
    from ruamel.yaml import YAML  # type: ignore
except Exception:  # pragma: no cover
    YAML = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover
    from tests._stubs import pretty_midi  # type: ignore

from .apply_controls import apply_controls
from .controls_spline import ControlCurve, tempo_map_from_prettymidi

logger = logging.getLogger(__name__)


def _parse_controls(spec: str) -> dict[str, bool]:
    out: dict[str, bool] = {"bend": False, "cc11": False, "cc64": False}
    for part in spec.split(","):
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        out[k.strip()] = v.strip().lower() == "on"
    return out


def _load_curves(path: Path, domain: str, sr: float) -> dict[str, ControlCurve]:
    with path.open() as fh:
        if path.suffix in {".yaml", ".yml"}:
            if YAML is None:
                raise RuntimeError("ruamel.yaml not installed")
            desc = YAML(typ="safe").load(fh)
        else:
            desc = json.load(fh)
    curves: dict[str, ControlCurve] = {}
    for name, spec in desc.items():
        knots = spec.get("knots")
        if not isinstance(knots, list):
            continue
        times = [float(t) for t, _ in knots]
        vals = [float(v) for _, v in knots]
        cur_domain = spec.get("domain", domain)
        curves[name] = ControlCurve(
            times,
            vals,
            domain=cur_domain,
            sample_rate_hz=sr,
        )
    return curves


def main(argv: list[str] | None = None) -> pretty_midi.PrettyMIDI:
    parser = argparse.ArgumentParser(description="Apply control curves to MIDI")
    parser.add_argument("in_mid")
    parser.add_argument("--curves", required=True, help="JSON/YAML control spec")
    parser.add_argument("--out", default="out.mid")
    parser.add_argument(
        "--controls",
        default="bend:on,cc11:off,cc64:off",
        help="Which controls to apply (e.g. 'bend:on,cc11:on')",
    )
    parser.add_argument(
        "--controls-domain",
        choices=["time", "beats"],
        default="time",
        help="Curve domain (seconds or beats)",
    )
    parser.add_argument(
        "--controls-resolution-hz",
        type=float,
        default=100.0,
        help="Sampling rate for the curves",
    )
    parser.add_argument(
        "--controls-max-events",
        type=int,
        default=200,
        help="Event cap per curve (0 disables)",
    )
    parser.add_argument(
        "--controls-total-max-events",
        type=int,
        default=0,
        help="Global cap across all emitted events",
    )
    parser.add_argument("--dedup-eps-time", type=float, default=1e-4)
    parser.add_argument("--dedup-eps-value", type=float, default=1.0)
    parser.add_argument("--bend-range-semitones", type=float, default=2.0)
    parser.add_argument("--write-rpn", action="store_true")
    parser.add_argument("--rpn-at", type=float, default=0.0)
    parser.add_argument(
        "--tempo-map-from-midi",
        help="Extract tempo map from MIDI for beats-domain curves",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    controls = _parse_controls(args.controls)
    sr = float(args.controls_resolution_hz)
    curves_all = _load_curves(Path(args.curves), args.controls_domain, sr)
    curves = {k: v for k, v in curves_all.items() if controls.get(k, False)}

    pm = pretty_midi.PrettyMIDI(args.in_mid)

    tempo_map = None
    if args.controls_domain == "beats" and args.tempo_map_from_midi:
        tempo_map = tempo_map_from_prettymidi(
            pretty_midi.PrettyMIDI(args.tempo_map_from_midi)
        )
    elif args.controls_domain == "beats" and not args.tempo_map_from_midi:
        raise ValueError("beats domain requires --tempo-map-from-midi")

    ch_map: dict[int, dict[str, ControlCurve]] = {}
    for name, curve in curves.items():
        ch_map.setdefault(0, {})[name] = curve

    max_events = None
    if args.controls_max_events:
        max_events = {
            "bend": args.controls_max_events,
            "cc11": args.controls_max_events,
            "cc64": args.controls_max_events,
        }

    sample_rate = {"bend": sr, "cc11": sr, "cc64": sr}

    apply_controls(
        pm,
        ch_map,
        bend_range_semitones=args.bend_range_semitones,
        write_rpn=args.write_rpn,
        rpn_at=args.rpn_at,
        sample_rate_hz=sample_rate,
        max_events=max_events,
        total_max_events=args.controls_total_max_events or None,
        value_eps=args.dedup_eps_value,
        time_eps=args.dedup_eps_time,
        tempo_map=tempo_map,
    )

    total_cc = sum(len(i.control_changes) for i in pm.instruments)
    total_pb = sum(len(i.pitch_bends) for i in pm.instruments)
    print(f"Rendered {total_cc} CC and {total_pb} pitch-bend events")

    if not args.dry_run:
        pm.write(args.out)
        print(f"Wrote {args.out}")
    return pm


if __name__ == "__main__":  # pragma: no cover
    main()
