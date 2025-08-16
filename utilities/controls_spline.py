"""Utilities for rendering sparse control curves to pretty_midi events.

PrettyMIDI's :class:`ControlChange` and :class:`PitchBend` objects do not
contain a channel attribute. Routing is therefore achieved by assigning events
to per-channel instruments, e.g. instruments named ``"channel0"``.
"""

from __future__ import annotations

import logging
import math
import warnings
from bisect import bisect_right
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

try:  # optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore
import pretty_midi

__all__ = [
    "ControlCurve",
    "catmull_rom_monotone",
    "dedupe_events",
    "dedupe_times_values",
    "ensure_scalar_floats",
]

_WARNED_RESOLUTION = False

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def ensure_scalar_floats(seq: Iterable[float]) -> list[float]:
    """Return ``seq`` as a list of Python ``float`` values."""
    return [float(x) for x in seq]


if np is not None:

    def as_array(xs: Iterable[float]):
        return np.asarray(list(xs), dtype=float)

    def clip(xs, lo, hi):
        return np.clip(xs, lo, hi)

    def round_int(xs):
        return np.rint(xs).astype(int)

else:  # pragma: no cover

    def as_array(xs: Iterable[float]):
        return [float(x) for x in xs]

    def clip(xs, lo, hi):
        lo_f = float(lo)
        hi_f = float(hi)
        return [max(lo_f, min(hi_f, float(x))) for x in xs]

    def round_int(xs):
        return [int(round(float(x))) for x in xs]


def dedupe_events(
    times: Sequence[float],
    values: Sequence[float],
    *,
    value_eps: float = 1e-6,
    time_eps: float = 1e-9,
):
    """Deduplicate nearly-identical consecutive ``times``/``values`` pairs.

    Keeps the first sample in any run of (nearly) equal events.
    Returns numpy arrays when numpy is available, otherwise Python lists.
    """

    if not times:
        if np is not None:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        return [], []

    out_t: list[float] = [float(times[0])]
    out_v: list[float] = [float(values[0])]
    for idx, (t, v) in enumerate(zip(times[1:], values[1:]), start=1):
        t = float(t)
        v = float(v)
        if abs(t - out_t[-1]) <= time_eps and abs(v - out_v[-1]) <= value_eps:
            continue
        out_t.append(t)
        out_v.append(v)
    # Always keep the final endpoint
    if out_t[-1] != float(times[-1]) or out_v[-1] != float(values[-1]):
        out_t.append(float(times[-1]))
        out_v.append(float(values[-1]))

    if np is not None:
        return np.asarray(out_t), np.asarray(out_v)
    return out_t, out_v


def dedupe_times_values(
    times: Sequence[float],
    values: Sequence[float],
    *,
    value_eps: float = 1e-6,
    time_eps: float = 1e-9,
):
    """Alias for :func:`dedupe_events` for backward compatibility."""

    return dedupe_events(times, values, value_eps=value_eps, time_eps=time_eps)


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
    if len(x) == 0:
        return []
    if len(x) == 1:
        return [y[0] for _ in q]

    # Slopes and secants
    h = [x[i + 1] - x[i] for i in range(len(x) - 1)]
    delta = [(y[i + 1] - y[i]) / h[i] if h[i] != 0 else 0.0 for i in range(len(x) - 1)]

    # Tangents m[i]
    m = [0.0] * len(x)
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, len(x) - 1):
        if delta[i - 1] * delta[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            denom = (w1 / delta[i - 1]) + (w2 / delta[i])
            m[i] = (w1 + w2) / denom if denom != 0 else 0.0

    res: list[float] = []
    for t in q:
        if t <= x[0]:
            res.append(y[0])
            continue
        if t >= x[-1]:
            res.append(y[-1])
            continue
        # Find segment j such that x[j] <= t <= x[j+1]
        j = bisect_right(x, t) - 1
        h_j = x[j + 1] - x[j]
        s = (t - x[j]) / h_j if h_j != 0 else 0.0
        s2 = s * s
        s3 = s2 * s
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2
        val = h00 * y[j] + h10 * h_j * m[j] + h01 * y[j + 1] + h11 * h_j * m[j + 1]
        res.append(val)

    return res


# -----------------------------------------------------------------------------
# Tempo mapping helpers
# -----------------------------------------------------------------------------


@dataclass
class TempoMap:
    """Piecewise-constant tempo map.

    Parameters
    ----------
    events:
        Sequence of ``(beat, bpm)`` pairs with strictly increasing beats and
        positive BPM values.
    """

    events: Sequence[tuple[float, float]]

    def __post_init__(self) -> None:
        beats: list[float] = []
        bpms: list[float] = []
        last = -float("inf")
        for beat, bpm in self.events:
            beat_f = float(beat)
            bpm_f = float(bpm)
            if beat_f <= last:
                raise ValueError("beats must be strictly increasing")
            if bpm_f <= 0 or not math.isfinite(bpm_f):
                raise ValueError("bpm must be positive and finite")
            beats.append(beat_f)
            bpms.append(bpm_f)
            last = beat_f
        self.beats = beats
        self.bpms = bpms
        self._sec: list[float] = [0.0]
        for i in range(1, len(beats)):
            dt = (beats[i] - beats[i - 1]) * 60.0 / bpms[i - 1]
            self._sec.append(self._sec[-1] + dt)

    def bpm_at(self, beat: float) -> float:
        idx = 0
        for i, b in enumerate(self.beats):
            if beat >= b:
                idx = i
            else:
                break
        return float(self.bpms[idx])

    def sec_at(self, beat: float) -> float:
        if beat <= self.beats[0]:
            return (beat - self.beats[0]) * 60.0 / self.bpms[0]
        for i in range(len(self.beats) - 1):
            b0, b1 = self.beats[i], self.beats[i + 1]
            if beat < b1:
                return self._sec[i] + (beat - b0) * 60.0 / self.bpms[i]
        return self._sec[-1] + (beat - self.beats[-1]) * 60.0 / self.bpms[-1]


def tempo_map_from_events(
    events: Sequence[tuple[float, float]],
) -> Callable[[float], float]:
    """Return a piecewise-constant tempo map callable.

    Parameters
    ----------
    events:
        Sequence of ``(beat, bpm)`` pairs with non-decreasing beats and
        strictly positive BPM values.
    """

    beats: list[float] = []
    bpms: list[float] = []
    last_beat = float("-inf")
    for beat, bpm in events:
        if beat < last_beat:
            raise ValueError("tempo events must have non-decreasing beats")
        if bpm <= 0:
            raise ValueError("bpm must be positive")
        beats.append(float(beat))
        bpms.append(float(bpm))
        last_beat = float(beat)

    def tempo(beat: float) -> float:
        idx = 0
        for i, b in enumerate(beats):
            if beat >= b:
                idx = i
            else:
                break
        return float(bpms[idx])

    return tempo


# -----------------------------------------------------------------------------
# Internal numeric helpers
# -----------------------------------------------------------------------------


def _resample(t_knots, v_knots, sample_rate_hz: float):
    """Resample ``t_knots``/``v_knots`` on an equally spaced time grid."""
    if sample_rate_hz <= 0 or len(t_knots) < 2:
        return t_knots, v_knots
    step = 1.0 / float(sample_rate_hz)
    t0 = float(t_knots[0])
    t1 = float(t_knots[-1])
    if np is not None:
        grid = np.arange(t0, t1, step, dtype=float)
        if grid.size == 0 or float(grid[-1]) < t1:
            grid = np.append(grid, t1)
        vals = catmull_rom_monotone(t_knots, v_knots, grid.tolist())
        return grid, np.asarray(vals, dtype=float)
    # Fallback without numpy
    grid: list[float] = []
    g = t0
    while g < t1 - 1e-12:
        grid.append(g)
        g += step
    grid.append(t1)
    vals = catmull_rom_monotone(t_knots, v_knots, grid)
    return grid, vals


def _dedupe_int(times, values, *, time_eps: float, keep_last: bool = False):
    out_t: list[float] = [float(times[0])]
    out_v: list[int] = [int(values[0])]
    for idx in range(1, len(times)):
        t = float(times[idx])
        v = int(values[idx])
        if v == out_v[-1] or abs(t - out_t[-1]) <= time_eps:
            if keep_last and idx == len(times) - 1:
                out_t.append(t)
                out_v.append(v)
            continue
        out_t.append(t)
        out_v.append(v)
    if keep_last and (out_t[-1] != float(times[-1]) or out_v[-1] != int(values[-1])):
        out_t.append(float(times[-1]))
        out_v.append(int(values[-1]))
    if np is not None:
        return np.asarray(out_t), np.asarray(out_v)
    return out_t, out_v


_ZERO_WARNING = "offset_sec must be non-negative; clamping to 0.0"


# -----------------------------------------------------------------------------
# ControlCurve
# -----------------------------------------------------------------------------


@dataclass
class ControlCurve:
    """Simple spline-based control curve.

    ``resolution_hz`` is a deprecated alias for ``sample_rate_hz``.

    Parameters
    ----------
    times, values:
        Sample locations and values.  ``domain`` controls whether ``times``
        are in seconds (``"time"``) or beats (``"beats"``).
    domain:
        ``"time"`` for absolute seconds or ``"beats"`` for beat positions.
    offset_sec:
        Shift applied to all rendered event times.  Negative values are
        clamped to ``0.0`` with a warning.
    """

    times: Sequence[float]
    values: Sequence[float]
    domain: str = "time"
    offset_sec: float = 0.0
    units: str | None = None
    sample_rate_hz: float = 0.0
    resolution_hz: float | None = None
    eps_cc: float = 0.5
    eps_bend: float = 1.0
    ensure_zero_at_edges: bool = True

    def __post_init__(self) -> None:
        # arrays where available for speed/consistency
        self.times = as_array(self.times)
        self.values = as_array(self.values)
        if self.offset_sec < 0.0:
            logging.warning(_ZERO_WARNING)
            self.offset_sec = 0.0
        self.units = self.units or "semitones"
        if self.resolution_hz is not None:
            global _WARNED_RESOLUTION
            if not _WARNED_RESOLUTION:
                warnings.warn(
                    "resolution_hz is deprecated; use sample_rate_hz",
                    DeprecationWarning,
                    stacklevel=2,
                )
                _WARNED_RESOLUTION = True
            if not self.sample_rate_hz:
                self.sample_rate_hz = float(self.resolution_hz)
            self.resolution_hz = None
        if self.sample_rate_hz < 0:
            raise ValueError("sample_rate_hz must be >= 0")

    # ---- validation -------------------------------------------------
    def validate(self) -> None:
        times = self.times if np is None else self.times.tolist()
        if len(times) != (len(self.values) if np is None else int(self.values.size)):
            raise ValueError("times and values must be same length")
        if len(times) == 0:
            return
        if self.sample_rate_hz < 0:
            raise ValueError("sample_rate_hz must be >= 0")
        diffs = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        if any(d < 0 for d in diffs):
            raise ValueError("times must be non-decreasing")

    # ---- domain conversion -----------------------------------------
    def _beats_to_times(
        self,
        beats,  # array-like
        tempo_map: (
            float | Sequence[tuple[float, float]] | Callable[[float], float] | TempoMap
        ),
        *,
        fold_halves: bool = False,
    ) -> list[float]:
        if isinstance(tempo_map, TempoMap):
            beats_list = (
                beats
                if isinstance(beats, list)
                else (beats.tolist() if np is not None else list(beats))
            )
            return [tempo_map.sec_at(b) for b in beats_list]
        if callable(tempo_map):
            tempo = tempo_map
        elif isinstance(tempo_map, int | float):
            const = float(tempo_map)
            if const <= 0 or not math.isfinite(const):
                raise ValueError("bpm must be positive and finite")
            tempo = lambda _b: const  # noqa: E731
        else:
            tempo = tempo_map_from_events(tempo_map)
        beats_list = (
            beats
            if isinstance(beats, list)
            else (beats.tolist() if np is not None else list(beats))
        )

        def _fold(bpm: float) -> float:
            if not fold_halves:
                return bpm
            while bpm < 60.0:
                bpm *= 2.0
            while bpm > 180.0:
                bpm /= 2.0
            return bpm

        times: list[float] = [0.0]
        for a, b in zip(beats_list[:-1], beats_list[1:]):
            mid = (a + b) / 2.0
            bpm = float(tempo(mid))
            if not math.isfinite(bpm) or bpm <= 0:
                raise ValueError("bpm must be positive and finite")
            bpm = _fold(bpm)
            dt = (b - a) * 60.0 / bpm
            times.append(times[-1] + dt)
        return times

    def _prep(
        self,
        tempo_map: (
            float | Sequence[tuple[float, float]] | Callable[[float], float] | TempoMap
        ),
        *,
        fold_halves: bool = False,
    ):
        self.validate()
        t, v = dedupe_events(self.times, self.values)
        if self.domain == "beats":
            t = self._beats_to_times(t, tempo_map, fold_halves=fold_halves)
        return t, v

    # ---- MIDI CC rendering -----------------------------------------
    def to_midi_cc(
        self,
        inst,
        cc_number: int,
        *,
        channel: int | None = None,
        time_offset: float = 0.0,
        tempo_map: (
            float | Sequence[tuple[float, float]] | Callable[[float], float]
        ) = 120.0,
        sample_rate_hz: float | None = None,
        resolution_hz: float | None = None,
        max_events: int | None = None,
        value_eps: float = 1e-6,
        time_eps: float = 1e-9,
        min_delta: float | None = None,
        fold_halves: bool = False,
    ) -> None:
        """Render the curve as MIDI CC events onto ``inst``.

        ``tempo_map`` may be a constant BPM number, a callable ``beat → bpm``
        function, or a list of ``(beat, bpm)`` pairs.  Endpoint events are
        preserved even when ``max_events`` trims intermediate points.
        ``resolution_hz`` is a deprecated alias for ``sample_rate_hz`` and is
        slated for removal in a future release.
        """

        if channel is not None and not 0 <= channel <= 15:
            raise ValueError("MIDI channel must be in 0..15")
        if not 0 <= cc_number <= 127:
            raise ValueError("CC number must be in 0..127")

        if resolution_hz is not None:
            warnings.warn(
                "resolution_hz is deprecated; use sample_rate_hz",
                DeprecationWarning,
                stacklevel=2,
            )
            if sample_rate_hz is None:
                sample_rate_hz = resolution_hz
        if sample_rate_hz is None or sample_rate_hz <= 0:
            sample_rate_hz = self.sample_rate_hz

        if max_events is not None and max_events < 2:
            max_events = 2

        t, v = self._prep(tempo_map, fold_halves=fold_halves)
        if sample_rate_hz and sample_rate_hz > 0:
            t, v = _resample(t, v, float(sample_rate_hz))
        t, v = dedupe_events(t, v, value_eps=value_eps, time_eps=time_eps)
        # clamp + round to MIDI domain
        vals = round_int(clip(v, 0, 127))
        t, vals = _dedupe_int(t, vals, time_eps=time_eps, keep_last=True)
        if min_delta is not None and len(vals) > 1:
            ft: list[float] = [t[0]]
            fv: list[int] = [int(vals[0])]
            for tt, vv in zip(t[1:-1], vals[1:-1]):
                if abs(vv - fv[-1]) >= min_delta:
                    ft.append(tt)
                    fv.append(int(vv))
            ft.append(t[-1])
            fv.append(int(vals[-1]))
            t, vals = ft, fv
        if max_events is not None and len(vals) > max_events:
            orig = len(vals)
            cc = ControlCurve.from_dense(t, vals, tol=1e-9, max_knots=max_events)
            t = cc.times if np is None else cc.times.tolist()
            vals = [
                int(round(v)) for v in (cc.values if np is None else cc.values.tolist())
            ]
            t, vals = _dedupe_int(t, vals, time_eps=time_eps, keep_last=True)
            logging.debug(
                "Simplified CC events %s→%s due to max_events=%s",
                orig,
                len(vals),
                max_events,
            )
        for tt, vv in zip(t, vals):
            inst.control_changes.append(
                pretty_midi.ControlChange(
                    number=int(cc_number),
                    value=int(vv),
                    time=float(tt + self.offset_sec + time_offset),
                )
            )

    # ---- Pitch bend rendering --------------------------------------
    def to_pitch_bend(
        self,
        inst,
        *,
        time_offset: float = 0.0,
        tempo_map: (
            float | Sequence[tuple[float, float]] | Callable[[float], float]
        ) = 120.0,
        bend_range_semitones: float = 2.0,
        sample_rate_hz: float | None = None,
        resolution_hz: float | None = None,
        max_events: int | None = None,
        value_eps: float = 1e-6,
        time_eps: float = 1e-9,
        units: str = "semitones",
        fold_halves: bool = False,
    ) -> None:
        """Render the curve as MIDI pitch-bend events onto ``inst``.

        Values are interpreted in ``units``: either ``"semitones"`` (scaled by
        ``bend_range_semitones``) or ``"normalized"`` where ``-1``..``1`` maps
        directly to the 14-bit bend range ``[-8192, 8191]``.  ``tempo_map`` has
        the same semantics as in :meth:`to_midi_cc`.  ``resolution_hz`` is a
        deprecated alias for ``sample_rate_hz``.
        Endpoint events are preserved when ``max_events`` is specified.
        """

        if resolution_hz is not None:
            warnings.warn(
                "resolution_hz is deprecated; use sample_rate_hz",
                DeprecationWarning,
                stacklevel=2,
            )
            if sample_rate_hz is None:
                sample_rate_hz = resolution_hz
        if sample_rate_hz is None or sample_rate_hz <= 0:
            sample_rate_hz = self.sample_rate_hz

        if max_events is not None and max_events < 2:
            max_events = 2

        t, v = self._prep(tempo_map, fold_halves=fold_halves)
        if sample_rate_hz and sample_rate_hz > 0:
            t, v = _resample(t, v, float(sample_rate_hz))
        t, v = dedupe_events(t, v, value_eps=value_eps, time_eps=time_eps)
        # convert to 14-bit domain
        if units == "normalized":
            vals = round_int(clip([x * 8192.0 for x in v], -8192, 8191))
        else:
            scale = 8192.0 / float(bend_range_semitones)
            vals = round_int(clip([x * scale for x in v], -8192, 8191))
        t, vals = _dedupe_int(t, vals, time_eps=time_eps, keep_last=True)
        if max_events is not None and len(vals) > max_events:
            orig = len(vals)
            cc = ControlCurve.from_dense(t, vals, tol=1e-9, max_knots=max_events)
            t = cc.times if np is None else cc.times.tolist()
            vals = [
                int(round(v)) for v in (cc.values if np is None else cc.values.tolist())
            ]
            t, vals = _dedupe_int(t, vals, time_eps=time_eps, keep_last=True)
            logging.debug(
                "Simplified bend events %s→%s due to max_events=%s",
                orig,
                len(vals),
                max_events,
            )
        if self.ensure_zero_at_edges and len(vals) > 0:
            if vals[0] != 0:
                t.insert(0, t[0])
                vals.insert(0, 0)
            if vals[-1] != 0:
                t.append(t[-1])
                vals.append(0)
        for tt, vv in zip(t, vals):
            inst.pitch_bends.append(
                pretty_midi.PitchBend(
                    pitch=int(vv), time=float(tt + self.offset_sec + time_offset)
                )
            )

    # ---- utils ------------------------------------------------------
    @staticmethod
    def convert_to_14bit(
        values: Sequence[float],
        range_semitones: float,
        units: str = "semitones",
    ) -> list[int]:
        """Convert ``values`` to 14-bit pitch-bend integers.

        ``units`` can be ``"semitones"`` (default) or ``"normalized"``.
        """

        vals = ensure_scalar_floats(values)
        if units == "normalized":
            arr = [v * 8192.0 for v in vals]
        else:
            scale = 8192.0 / float(range_semitones)
            arr = [v * scale for v in vals]
        if np is not None:
            return np.rint(np.clip(arr, -8192, 8191)).astype(int).tolist()
        return [int(round(max(-8192, min(8191, a)))) for a in arr]

    # ---- simplification --------------------------------------------
    @staticmethod
    def from_dense(
        times: Sequence[float],
        values: Sequence[float],
        *,
        tol: float = 1.5,
        max_knots: int = 256,
        target: str | None = None,  # kept for compatibility with older code
    ) -> ControlCurve:
        """Create :class:`ControlCurve` from dense ``times``/``values``.

        Douglas–Peucker simplification is performed in ``(time, value)`` space
        so that the returned curve approximates the dense sequence within
        ``tol`` and contains at most ``max_knots`` knots.
        """

        pts = list(zip(ensure_scalar_floats(times), ensure_scalar_floats(values)))
        if not pts:
            raise ValueError("from_dense requires at least one point")
        if len(pts) == 1:
            t0, v0 = pts[0]
            return ControlCurve([t0], [v0])

        def _dp(start: int, end: int, out: list[tuple[float, float]]) -> None:
            if len(out) >= max_knots:
                return
            t0, v0 = pts[start]
            t1, v1 = pts[end]
            if end - start <= 1:
                return
            # line distance
            max_dist = -1.0
            idx: int | None = None
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
        t_s, v_s = zip(*simplified)
        return ControlCurve(list(t_s), list(v_s))
