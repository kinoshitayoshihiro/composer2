"""Utilities for rendering sparse control curves to pretty_midi events.

PrettyMIDI's :class:`ControlChange` and :class:`PitchBend` objects do not
contain a channel attribute. Routing is therefore achieved by assigning events
to per-channel instruments, e.g. instruments named ``"channel0"``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from typing import Iterable, List, Tuple

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


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def ensure_scalar_floats(seq: Iterable[float]) -> List[float]:
    """Return ``seq`` as a list of Python ``float`` values."""

    return [float(x) for x in seq]


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

    out_t: List[float] = [float(times[0])]
    out_v: List[float] = [float(values[0])]
    for t, v in zip(times[1:], values[1:]):
        t = float(t)
        v = float(v)
        if abs(t - out_t[-1]) <= time_eps and abs(v - out_v[-1]) <= value_eps:
            continue
        out_t.append(t)
        out_v.append(v)

    if np is not None:
        return np.asarray(out_t), np.asarray(out_v)
    return out_t, out_v


def catmull_rom_monotone(
    times: Sequence[float], values: Sequence[float], query_times: Sequence[float]
) -> List[float]:
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

    res: List[float] = []
    for t in q:
        if t <= x[0]:
            res.append(y[0])
            continue
        if t >= x[-1]:
            res.append(y[-1])
            continue
        # Find segment j such that x[j] <= t <= x[j+1]
        j = 0
        for k in range(len(x) - 1):
            if x[k] <= t <= x[k + 1]:
                j = k
                break
        h_j = x[j + 1] - x[j]
        s = (t - x[j]) / h_j if h_j != 0 else 0.0
        s2 = s * s
        s3 = s2 * s
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2
        val = (
            h00 * y[j]
            + h10 * h_j * m[j]
            + h01 * y[j + 1]
            + h11 * h_j * m[j + 1]
        )
        res.append(val)

    return res


# -----------------------------------------------------------------------------
# Tempo mapping helpers
# -----------------------------------------------------------------------------

def tempo_map_from_events(
    events: "Sequence[tuple[float, float]]",
) -> "Callable[[float], float]":
    """Return a piecewise-constant tempo map callable.

    Parameters
    ----------
    events:
        Sequence of ``(beat, bpm)`` pairs with non-decreasing beats and
        strictly positive BPM values.
    """

    beats: List[float] = []
    bpms: List[float] = []
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

def _resample(
    t_knots, v_knots, sample_rate_hz: float
):
    """Resample ``t_knots``/``v_knots`` on an equally spaced time grid."""
    if sample_rate_hz <= 0 or len(t_knots) < 2:
        return t_knots, v_knots
    step = 1.0 / float(sample_rate_hz)
    t0 = float(t_knots[0])
    t1 = float(t_knots[-1])
    # Build grid; ensure the exact last time is included
    grid = []
    g = t0
    while g < t1 - 1e-12:
        grid.append(g)
        g += step
    grid.append(t1)
    vals = catmull_rom_monotone(t_knots, v_knots, grid)
    if np is not None:
        return np.asarray(grid, dtype=float), np.asarray(vals, dtype=float)
    return grid, vals


def _dedupe_int(
    times, values, *, time_eps: float, keep_last: bool = False
):
    out_t: List[float] = [float(times[0])]
    out_v: List[int] = [int(values[0])]
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


def _rdp_indices(t, v, max_points: int) -> list[int]:
    """Return indices subsampled via a simple RDP-like algorithm."""

    n = len(t)
    if n <= max_points:
        return list(range(n))

    keep: List[int] = [0, n - 1]

    def _seg_error(a: int, b: int) -> tuple[float, int | None]:
        ta, tb = float(t[a]), float(t[b])
        va, vb = float(v[a]), float(v[b])
        dt = tb - ta
        dv = vb - va
        denom = dt * dt + dv * dv
        max_dist = -1.0
        max_idx: int | None = None
        for i in range(a + 1, b):
            tt = float(t[i]) - ta
            vv = float(v[i]) - va
            if denom == 0:
                dist = abs(vv)
            else:
                proj = (tt * dt + vv * dv) / denom
                proj_t = ta + proj * dt
                proj_v = va + proj * dv
                dist = math.hypot(float(t[i]) - proj_t, float(v[i]) - proj_v)
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        return max_dist, max_idx

    while len(keep) < max_points:
        max_dist = -1.0
        max_idx: int | None = None
        keep_sorted = sorted(keep)
        for a, b in zip(keep_sorted[:-1], keep_sorted[1:]):
            dist, idx = _seg_error(a, b)
            if idx is not None and dist > max_dist:
                max_dist = dist
                max_idx = idx
        if max_idx is None:
            break
        keep.append(max_idx)

    return sorted(keep)


# -----------------------------------------------------------------------------
# ControlCurve
# -----------------------------------------------------------------------------

class ControlCurve:
    """Simple spline-based control curve.

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

    def __init__(
        self,
        times: "Sequence[float]",
        values: "Sequence[float]",
        *,
        domain: str = "time",
        offset_sec: float = 0.0,
        units: str | None = None,
        resolution_hz: float | None = None,
        eps_cc: float | None = None,
        eps_bend: float | None = None,
    ) -> None:
        # Numpy arrays where available for speed/consistency
        self.times = np.asarray(times, dtype=float) if np is not None else list(map(float, times))
        self.values = np.asarray(values, dtype=float) if np is not None else list(map(float, values))
        self.domain = domain
        if offset_sec < 0.0:
            warnings.warn("offset_sec must be non-negative; clamping to 0.0")
            offset_sec = 0.0
        self.offset_sec = float(offset_sec)
        # Optional defaults (kept for compatibility with older ctor usage in some code)
        self.units = units or "semitones"
        self.resolution_hz = resolution_hz if (resolution_hz is not None) else 0.0
        self.eps_cc = eps_cc if (eps_cc is not None) else 0.5
        self.eps_bend = eps_bend if (eps_bend is not None) else 1.0

    # ---- validation -------------------------------------------------
    def validate(self) -> None:
        times = self.times if np is None else self.times.tolist()
        if len(times) != (len(self.values) if np is None else int(self.values.size)):
            raise ValueError("times and values must be same length")
        if len(times) == 0:
            return
        diffs = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        if any(d < 0 for d in diffs):
            raise ValueError("times must be non-decreasing")

    # ---- domain conversion -----------------------------------------
    def _beats_to_times(
        self,
        beats,  # array-like
        tempo_map: float | Sequence[tuple[float, float]] | Callable[[float], float],
    ) -> List[float]:
        if callable(tempo_map):
            tempo = tempo_map
        elif isinstance(tempo_map, (int, float)):
            const = float(tempo_map)
            tempo = lambda _b: const  # noqa: E731
        else:
            tempo = tempo_map_from_events(tempo_map)
        beats_list = beats if isinstance(beats, list) else (beats.tolist() if np is not None else list(beats))
        times: List[float] = [0.0]
        for a, b in zip(beats_list[:-1], beats_list[1:]):
            mid = (a + b) / 2.0
            bpm = tempo(mid)
            dt = (b - a) * 60.0 / bpm
            times.append(times[-1] + dt)
        return times

    def _prep(
        self,
        tempo_map: float | Sequence[tuple[float, float]] | Callable[[float], float],
    ):
        self.validate()
        t, v = dedupe_events(self.times, self.values)
        if self.domain == "beats":
            t = self._beats_to_times(t, tempo_map)
        return t, v

    # ---- MIDI CC rendering -----------------------------------------
    def to_midi_cc(
        self,
        inst,
        cc_number: int,
        *,
        tempo_map: (
            float | Sequence[tuple[float, float]] | Callable[[float], float]
        ) = 120.0,
        sample_rate_hz: float | None = None,
        resolution_hz: float | None = None,
        max_events: int | None = None,
        value_eps: float = 1e-6,
        time_eps: float = 1e-9,
    ) -> None:
        """Render the curve as MIDI CC events onto ``inst``.

        ``tempo_map`` may be a constant BPM number, a callable ``beat → bpm``
        function, or a list of ``(beat, bpm)`` pairs.  Endpoint events are
        preserved even when ``max_events`` trims intermediate points.
        ``resolution_hz`` is a deprecated alias for ``sample_rate_hz`` and is
        slated for removal in a future release.
        """

        if resolution_hz is not None and sample_rate_hz is None:
            warnings.warn(
                "resolution_hz is deprecated; use sample_rate_hz",
                DeprecationWarning,
                stacklevel=2,
            )
            sample_rate_hz = resolution_hz

        if max_events is not None and max_events < 2:
            max_events = 2

        t, v = self._prep(tempo_map)
        if sample_rate_hz and sample_rate_hz > 0:
            t, v = _resample(t, v, float(sample_rate_hz))
        t, v = dedupe_events(t, v, value_eps=value_eps, time_eps=time_eps)
        # clamp + round to MIDI domain
        if np is not None:
            vals = np.clip(np.rint(v), 0, 127).astype(int)
        else:
            vals = [int(round(max(0, min(127, float(x))))) for x in v]
        t, vals = _dedupe_int(
            t, vals, time_eps=time_eps, keep_last=max_events is not None
        )
        if max_events is not None and len(vals) > max_events:
            idx = _rdp_indices(t, vals, max_events)
            t = [t[i] for i in idx]
            vals = [int(vals[i]) for i in idx]
            t, vals = _dedupe_int(t, vals, time_eps=time_eps, keep_last=True)
        for tt, vv in zip(t, vals):
            inst.control_changes.append(
                pretty_midi.ControlChange(
                    number=int(cc_number), value=int(vv), time=float(tt + self.offset_sec)
                )
            )

    # ---- Pitch bend rendering --------------------------------------
    def to_pitch_bend(
        self,
        inst,
        *,
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
    ) -> None:
        """Render the curve as MIDI pitch-bend events onto ``inst``.

        Values are interpreted in ``units``: either ``"semitones"`` (scaled by
        ``bend_range_semitones``) or ``"normalized"`` where ``-1``..``1`` maps
        directly to the 14-bit bend range ``[-8192, 8191]``.  ``tempo_map`` has
        the same semantics as in :meth:`to_midi_cc`.  ``resolution_hz`` is a
        deprecated alias for ``sample_rate_hz``.
        Endpoint events are preserved when ``max_events`` is specified.
        """

        if resolution_hz is not None and sample_rate_hz is None:
            warnings.warn(
                "resolution_hz is deprecated; use sample_rate_hz",
                DeprecationWarning,
                stacklevel=2,
            )
            sample_rate_hz = resolution_hz

        if max_events is not None and max_events < 2:
            max_events = 2

        t, v = self._prep(tempo_map)
        if sample_rate_hz and sample_rate_hz > 0:
            t, v = _resample(t, v, float(sample_rate_hz))
        t, v = dedupe_events(t, v, value_eps=value_eps, time_eps=time_eps)
        # convert to 14-bit domain
        if units == "normalized":
            if np is not None:
                vals = np.clip(np.rint(np.asarray(v) * 8192.0), -8192, 8191).astype(int)
            else:
                vals = [int(round(max(-8192, min(8191, float(x) * 8192.0)))) for x in v]
        else:
            scale = 8192.0 / float(bend_range_semitones)
            if np is not None:
                vals = np.clip(np.rint(np.asarray(v) * scale), -8192, 8191).astype(int)
            else:
                vals = [int(round(max(-8192, min(8191, float(x) * scale)))) for x in v]
        t, vals = _dedupe_int(
            t, vals, time_eps=time_eps, keep_last=max_events is not None
        )
        if max_events is not None and len(vals) > max_events:
            idx = _rdp_indices(t, vals, max_events)
            t = [t[i] for i in idx]
            vals = [int(vals[i]) for i in idx]
            t, vals = _dedupe_int(t, vals, time_eps=time_eps, keep_last=True)
        for tt, vv in zip(t, vals):
            inst.pitch_bends.append(
                pretty_midi.PitchBend(pitch=int(vv), time=float(tt + self.offset_sec))
            )

    # ---- utils ------------------------------------------------------
    @staticmethod
    def convert_to_14bit(
        values: Sequence[float],
        range_semitones: float,
        units: str = "semitones",
    ) -> List[int]:
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
    ) -> "ControlCurve":
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

        def _dp(start: int, end: int, out: List[Tuple[float, float]]) -> None:
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

        simplified: List[Tuple[float, float]] = [pts[0]]
        _dp(0, len(pts) - 1, simplified)
        simplified.append(pts[-1])
        simplified = sorted(set(simplified), key=lambda p: p[0])
        if len(simplified) > max_knots:
            simplified = simplified[: max_knots]
        t_s, v_s = zip(*simplified)
        return ControlCurve(list(t_s), list(v_s))
