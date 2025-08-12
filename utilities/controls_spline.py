"""Utilities for rendering sparse control curves to pretty_midi events.

PrettyMIDI's :class:`ControlChange` and :class:`PitchBend` objects do not
contain a channel attribute. Routing is therefore achieved by assigning events
to per-channel instruments, e.g. instruments named ``"channel0"``.
"""

from __future__ import annotations

import math
import warnings
from bisect import bisect_right
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

try:  # optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore
import pretty_midi

__all__ = [
    "ControlCurve",
    "catmull_rom_monotone",
    "dedupe_events",
    "ensure_scalar_floats",
]


def ensure_scalar_floats(seq: Iterable[float]) -> list[float]:
    """Return ``seq`` as a list of Python ``float`` values."""

    return [float(x) for x in seq]


def dedupe_events(
    times: Sequence[float],
    values: Sequence[float],
    eps: float = 0.5,
) -> tuple[list[float], list[float]]:
    """Collapse consecutive ``values`` within ``eps`` keeping first ``time``."""

    if not times:
        return [], []
    out_t = [times[0]]
    out_v = [values[0]]
    prev = values[0]
    for t, v in zip(times[1:], values[1:]):
        if abs(v - prev) > eps:
            out_t.append(t)
            out_v.append(v)
            prev = v
    return out_t, out_v


def _dedupe_by_tolerance(
    times: Sequence[float], values: Sequence[float], tol: float
) -> tuple[list[float], list[float]]:
    """Collapse consecutive ``values`` differing by less than ``tol``."""

    if not times:
        return [], []
    out_t = [float(times[0])]
    out_v = [float(values[0])]
    prev = values[0]
    for t, v in zip(times[1:], values[1:]):
        if abs(v - prev) >= tol:
            out_t.append(float(t))
            out_v.append(float(v))
            prev = v
    return out_t, out_v


def catmull_rom_monotone(
    times: Sequence[float], values: Sequence[float], query_times: Sequence[float]
) -> list[float]:
    """Return monotone cubic interpolation of ``values`` over ``times``.

    Uses the Fritsch–Carlson method which ensures the interpolant is
    monotone when the input data is monotone.
    """

    x = ensure_scalar_floats(times)
    y = ensure_scalar_floats(values)
    q = ensure_scalar_floats(query_times)
    if len(x) == 1:
        return [y[0] for _ in q]

    h = [x[i + 1] - x[i] for i in range(len(x) - 1)]
    delta = [(y[i + 1] - y[i]) / h[i] for i in range(len(x) - 1)]

    m = [0.0] * len(x)
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, len(x) - 1):
        if delta[i - 1] * delta[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    res = []
    for t in q:
        if t <= x[0]:
            res.append(y[0])
            continue
        if t >= x[-1]:
            res.append(y[-1])
            continue
        j = bisect_right(x, t) - 1
        h_j = x[j + 1] - x[j]
        s = (t - x[j]) / h_j
        s2 = s * s
        s3 = s2 * s
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2
        val = h00 * y[j] + h10 * h_j * m[j] + h01 * y[j + 1] + h11 * h_j * m[j + 1]
        res.append(val)

    return res


@dataclass
class ControlCurve:
    """Sparse control curve which can render MIDI events."""

    target: Literal["cc11", "cc64", "bend"]
    domain: Literal["time", "beats"] = "time"
    knots: list[tuple[float, float]] | None = None
    sample_rate_hz: float = 100.0
    resolution_hz: float | None = None
    clamp: bool = True
    units: Literal["semitones", "normalized"] = "semitones"
    eps_cc: float = 0.5
    eps_bend: float = 1.0
    max_events: int | None = None
    dedupe_tol: float = 1e-9

    def __post_init__(self) -> None:  # pragma: no cover - thin wrapper
        if self.knots is None:
            self.knots = []
        if self.resolution_hz is not None:
            warnings.warn(
                "resolution_hz is deprecated, use sample_rate_hz",
                DeprecationWarning,
                stacklevel=2,
            )
            self.sample_rate_hz = float(self.resolution_hz)

    # ---- validation -------------------------------------------------
    def validate(self) -> None:
        """Validate internal state raising ``ValueError`` on issues."""

        if self.target not in {"cc11", "cc64", "bend"}:
            raise ValueError("Invalid target")
        if self.domain not in {"time", "beats"}:
            raise ValueError("Invalid domain")
        if self.units not in {"semitones", "normalized"}:
            raise ValueError("Invalid units")
        if not self.knots:
            raise ValueError("ControlCurve requires at least one knot")
        prev_t = -math.inf
        for t, v in self.knots:
            if not math.isfinite(t) or t <= prev_t:
                raise ValueError("Knots must be strictly increasing and finite")
            if not math.isfinite(v):
                raise ValueError("Invalid knot value")
            prev_t = t

    # ---- sampling ---------------------------------------------------
    def sample(self, times: Sequence[float]) -> list[float]:
        """Return interpolated values at ``times``."""

        self.validate()
        k_t = [k[0] for k in self.knots]
        k_v = [k[1] for k in self.knots]
        vals = catmull_rom_monotone(k_t, k_v, times)
        if np is not None:
            vals = np.nan_to_num(vals).tolist()
        else:
            vals = [0.0 if not math.isfinite(v) else v for v in vals]
        if self.clamp and self.target != "bend":
            vals = [min(127, max(0, v)) for v in vals]
        return [float(v) for v in vals]

    # ---- MIDI rendering ---------------------------------------------
    def to_midi_cc(
        self,
        channel: int,
        cc_number: int,
        *,
        sample_rate_hz: float | None = None,
        max_events: int | None = None,
        tempo_map: Callable[[float], float] | None = None,
        start_offset_sec: float = 0.0,
    ) -> list[pretty_midi.ControlChange]:
        """Render the curve as MIDI CC events."""

        if self.target not in {"cc11", "cc64"}:
            raise ValueError("ControlCurve target must be CC")
        sr = sample_rate_hz or self.sample_rate_hz
        step_sec = 1.0 / sr
        start = self.knots[0][0]
        end = self.knots[-1][0]

        if self.domain == "time":
            n = int(math.ceil((end - start) / step_sec))
            t_query = [start + i * step_sec for i in range(n + 1)]
            out_times = t_query
        else:
            if tempo_map is None:
                raise ValueError("tempo_map required for beats domain")
            beats = [start]
            b = start
            while b < end:
                bpm = tempo_map(b)
                if not math.isfinite(bpm) or bpm <= 0:
                    raise ValueError("tempo_map must return finite positive BPM")
                b += step_sec * bpm / 60.0
                beats.append(min(b, end))
            t_query = beats
            out_times = [0.0]
            prev_b = beats[0]
            for bt in beats[1:]:
                bpm = tempo_map(prev_b)
                if not math.isfinite(bpm) or bpm <= 0:
                    raise ValueError("tempo_map must return finite positive BPM")
                out_times.append(out_times[-1] + (bt - prev_b) * 60.0 / bpm)
                prev_b = bt

        v = self.sample(t_query)
        t_raw, v_raw = _dedupe_by_tolerance(out_times, v, self.dedupe_tol)
        v_int = [int(round(min(127, max(0, v)))) for v in v_raw]
        t_list, v_list = dedupe_events(t_raw, v_int, eps=self.eps_cc)
        max_e = max_events if max_events is not None else self.max_events
        if max_e is not None and len(t_list) > max_e:
            simplified = ControlCurve.from_dense(
                t_raw, v_raw, tol=self.dedupe_tol, max_knots=max_e, target=self.target
            )
            t_raw = [t for t, _ in simplified.knots]
            v_raw = [v for _, v in simplified.knots]
            v_int = [int(round(min(127, max(0, v)))) for v in v_raw]
            t_list, v_list = dedupe_events(t_raw, v_int, eps=self.eps_cc)
        t_list = [t + start_offset_sec for t in t_list]
        pairs = sorted(zip(t_list, v_list), key=lambda p: p[0])
        events = [
            pretty_midi.ControlChange(number=cc_number, value=int(v), time=float(t))
            for t, v in pairs
        ]
        return events

    @staticmethod
    def convert_to_14bit(
        values: Sequence[float],
        range_semitones: float,
        units: str = "semitones",
    ) -> list[int]:
        """Convert ``values`` to 14-bit pitch-bend integers.

        Parameters
        ----------
        values:
            Sequence of bend values.
        range_semitones:
            Pitch-bend range in semitones.
        units:
            ``"semitones"`` interprets ``values`` directly in semitones,
            while ``"normalized"`` treats ``values`` as the range ``[-1, 1]``.
        """

        vals = ensure_scalar_floats(values)
        scale = 8192.0
        if units == "normalized":
            arr = [v * scale for v in vals]
        else:
            arr = [v / range_semitones * scale for v in vals]
        if np is not None:
            arr = np.round(arr).astype(int)
            arr = np.clip(arr, -8192, 8191)
            return arr.tolist()
        out = [int(round(v)) for v in arr]
        return [min(8191, max(-8192, v)) for v in out]

    def to_pitch_bend(
        self,
        channel: int,
        range_semitones: float = 2.0,
        *,
        sample_rate_hz: float | None = None,
        max_events: int | None = None,
        tempo_map: Callable[[float], float] | None = None,
        units: str | None = None,
        start_offset_sec: float = 0.0,
    ) -> list[pretty_midi.PitchBend]:
        """Render the curve as pitch-bend events."""

        if self.target != "bend":
            raise ValueError("ControlCurve target must be 'bend'")
        sr = sample_rate_hz or self.sample_rate_hz
        step_sec = 1.0 / sr
        start = self.knots[0][0]
        end = self.knots[-1][0]

        if self.domain == "time":
            n = int(math.ceil((end - start) / step_sec))
            t_query = [start + i * step_sec for i in range(n + 1)]
            out_times = t_query
        else:
            if tempo_map is None:
                raise ValueError("tempo_map required for beats domain")
            beats = [start]
            b = start
            while b < end:
                bpm = tempo_map(b)
                if not math.isfinite(bpm) or bpm <= 0:
                    raise ValueError("tempo_map must return finite positive BPM")
                b += step_sec * bpm / 60.0
                beats.append(min(b, end))
            t_query = beats
            out_times = [0.0]
            prev_b = beats[0]
            for bt in beats[1:]:
                bpm = tempo_map(prev_b)
                if not math.isfinite(bpm) or bpm <= 0:
                    raise ValueError("tempo_map must return finite positive BPM")
                out_times.append(out_times[-1] + (bt - prev_b) * 60.0 / bpm)
                prev_b = bt

        v = self.sample(t_query)
        v_raw = self.convert_to_14bit(v, range_semitones, units=units or self.units)
        t_raw, v_raw = _dedupe_by_tolerance(out_times, v_raw, self.dedupe_tol)
        t_list, v_list = dedupe_events(t_raw, v_raw, eps=self.eps_bend)
        max_e = max_events if max_events is not None else self.max_events
        if max_e is not None and len(t_list) > max_e:
            simplified = ControlCurve.from_dense(
                t_raw, v_raw, tol=self.dedupe_tol, max_knots=max_e, target=self.target
            )
            t_raw = [t for t, _ in simplified.knots]
            v_raw = [v for _, v in simplified.knots]
            t_list, v_list = dedupe_events(t_raw, v_raw, eps=self.eps_bend)
        t_list = [t + start_offset_sec for t in t_list]
        pairs = sorted(zip(t_list, v_list), key=lambda p: p[0])
        events = [pretty_midi.PitchBend(pitch=int(v), time=float(t)) for t, v in pairs]
        return events

    # ---- simplification ---------------------------------------------
    @staticmethod
    def from_dense(
        times: Sequence[float],
        values: Sequence[float],
        *,
        tol: float = 1.5,
        max_knots: int = 256,
        target: Literal["cc11", "cc64", "bend"] = "cc11",
    ) -> ControlCurve:
        """Create :class:`ControlCurve` from dense ``times``/``values``.

        Douglas–Peucker simplification is performed in ``(time, value)`` space
        so that the returned curve approximates the dense sequence within
        ``tol`` and contains at most ``max_knots`` knots.
        """

        pts = list(zip(ensure_scalar_floats(times), ensure_scalar_floats(values)))
        if not pts:
            raise ValueError("from_dense requires at least one point")

        def _dp(start: int, end: int, out: list[tuple[float, float]]) -> None:
            if len(out) >= max_knots:
                return
            t0, v0 = pts[start]
            t1, v1 = pts[end]
            if end - start <= 1:
                return
            # line distance
            max_dist = -1.0
            idx = None
            for i in range(start + 1, end):
                t, v = pts[i]
                if t1 == t0 and v1 == v0:
                    dist = math.hypot(t - t0, v - v0)
                else:
                    num = abs((v1 - v0) * t - (t1 - t0) * v + t1 * v0 - v1 * t0)
                    den = math.hypot(v1 - v0, t1 - t0)
                    dist = num / den
                if dist > max_dist:
                    max_dist = dist
                    idx = i
            if max_dist > tol and idx is not None:
                _dp(start, idx, out)
                out.append(pts[idx])
                _dp(idx, end, out)

        simplified: list[tuple[float, float]] = [pts[0]]
        _dp(0, len(pts) - 1, simplified)
        simplified.append(pts[-1])
        simplified = sorted(set(simplified), key=lambda p: p[0])
        if len(simplified) > max_knots:
            simplified = simplified[:max_knots]
        return ControlCurve(target=target, knots=simplified)
