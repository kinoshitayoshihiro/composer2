from __future__ import annotations

from typing import Mapping
from music21 import stream

from .cc_tools import merge_cc_events, CCEvent


def apply(part: stream.Part, envelope_map: Mapping, *, bpm: float = 120.0) -> None:
    """Apply CC envelope to ``part``.

    Example::

        apply(part, {0.0: {"type": "reverb", "start": 0, "end": 100,
                        "duration_ql": 4.0}})

    ``envelope_map`` should be ``{offset_ql: spec}`` where ``spec`` contains
    ``cc`` or ``type`` (``reverb``, ``delay``, ``chorus``, ``brightness``),
    ``start``/``end`` (or ``start_val``/``end_val``), ``duration_ql`` and
    ``shape`` (``lin``/``exp``/``log``).
    """

    STEP_MAP = {
        "reverb": 91,
        "delay": 93,
        "chorus": 94,
        "brightness": 74,
    }

    step_ql = float(bpm) / 3000.0  # ~20 ms per step
    events: list[CCEvent] = []

    for off, spec in envelope_map.items():
        try:
            start = float(off)
        except Exception:
            continue

        if "cc" not in spec and "type" not in spec:
            raise ValueError("fx envelope entry missing 'cc' or 'type'")

        dur = float(spec.get("duration_ql", 1.0))
        cc_num = (
            int(spec["cc"])
            if "cc" in spec
            else STEP_MAP.get(str(spec.get("type")).lower())
        )
        if cc_num is None:
            raise ValueError(f"Unknown envelope type: {spec.get('type')}")

        start_val = int(spec.get("start_val", spec.get("start", 0)))
        end_val = int(spec.get("end_val", spec.get("end", 127)))

        if not 0 <= start_val <= 127 or not 0 <= end_val <= 127:
            raise ValueError("MIDI values must be in range 0-127")

        shape = str(spec.get("shape", "lin"))
        if shape not in {"lin", "exp", "log"}:
            raise ValueError("shape must be one of 'lin', 'exp', 'log'")

        steps = max(1, int(dur / step_ql))
        for i in range(steps + 1):
            frac = i / steps
            if shape == "exp":
                frac *= frac
            elif shape == "log":
                frac = frac**0.5
            val = int(round(start_val + (end_val - start_val) * frac))
            t = start + dur * frac
            events.append((t, cc_num, val))

    base: set[CCEvent] = set(getattr(part, "_extra_cc", set()))
    merged = merge_cc_events(base, [(float(t), int(c), int(v)) for (t, c, v) in events])
    part._extra_cc = set(merged)


__all__ = ["apply"]
