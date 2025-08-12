from collections.abc import Callable, Sequence

import numpy as np

try:  # optional dependency
    from scipy.interpolate import CubicSpline
except Exception:  # pragma: no cover - optional
    CubicSpline = None  # type: ignore


def fit_spline(times, values, max_knots=16):
    """Fit a simple spline returning its knot positions and values."""
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    if t.size == 0:
        return t, v
    if t.size > max_knots:
        idx = np.linspace(0, t.size - 1, max_knots, dtype=int)
        t = t[idx]
        v = v[idx]
    return t, v


def eval_spline(knots_t, knots_v, query_t):
    """Evaluate a spline previously fit with :func:`fit_spline`."""
    t = np.asarray(knots_t, dtype=float)
    v = np.asarray(knots_v, dtype=float)
    q = np.asarray(query_t, dtype=float)
    if t.size < 2:
        return np.full_like(q, v[0] if v.size else 0.0, dtype=float)
    if CubicSpline is not None:
        cs = CubicSpline(t, v)
        return cs(q)
    # cubic Hermite fallback
    m = np.zeros_like(v)
    m[1:-1] = (v[2:] - v[:-2]) / (t[2:] - t[:-2])
    m[0] = (v[1] - v[0]) / (t[1] - t[0])
    m[-1] = (v[-1] - v[-2]) / (t[-1] - t[-2])
    result = np.zeros_like(q, dtype=float)
    for i in range(len(t) - 1):
        mask = (q >= t[i]) & (q <= t[i + 1])
        if not np.any(mask):
            continue
        h = t[i + 1] - t[i]
        s = (q[mask] - t[i]) / h
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        result[mask] = h00 * v[i] + h10 * h * m[i] + h01 * v[i + 1] + h11 * h * m[i + 1]
    return result


def tempo_map_from_events(
    events: "Sequence[tuple[float, float]]",
) -> "Callable[[float], float]":
    """Return a piecewise-constant tempo map callable.

    Parameters
    ----------
    events:
        Sequence of ``(beat, bpm)`` pairs with non-decreasing beats.
    """

    beats = []
    bpms = []
    last_beat = float("-inf")
    for beat, bpm in events:
        if beat < last_beat:
            raise ValueError("tempo events must have non-decreasing beats")
        if bpm <= 0:
            raise ValueError("bpm must be positive")
        beats.append(float(beat))
        bpms.append(float(bpm))
        last_beat = beat

    def tempo(beat: float) -> float:
        idx = 0
        for i, b in enumerate(beats):
            if beat >= b:
                idx = i
            else:
                break
        return float(bpms[idx])

    return tempo


def _resample(
    t_knots: np.ndarray, v_knots: np.ndarray, sample_rate_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Resample ``t_knots``/``v_knots`` on an equally spaced time grid."""
    if sample_rate_hz <= 0 or t_knots.size < 2:
        return t_knots, v_knots
    step = 1.0 / float(sample_rate_hz)
    t0 = float(t_knots[0])
    t1 = float(t_knots[-1])
    grid = np.arange(t0, t1, step)
    if grid.size == 0 or grid[-1] < t1:
        grid = np.append(grid, t1)
    vals = eval_spline(t_knots, v_knots, grid)
    return grid, vals


def dedupe_events(
    times: "Sequence[float]",
    values: "Sequence[float]",
    *,
    value_eps: float = 1e-6,
    time_eps: float = 1e-9,
):
    """Deduplicate nearly-identical consecutive ``times``/``values`` pairs."""

    if len(times) == 0:
        return np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    out_t = [float(times[0])]
    out_v = [float(values[0])]
    for t, v in zip(times[1:], values[1:]):
        t = float(t)
        v = float(v)
        if abs(t - out_t[-1]) <= time_eps and abs(v - out_v[-1]) <= value_eps:
            continue
        out_t.append(t)
        out_v.append(v)
    return np.asarray(out_t), np.asarray(out_v)


def _dedupe_int(
    times: "np.ndarray",
    values: "np.ndarray",
    *,
    time_eps: float,
    keep_last: bool = False,
):
    out_t = [float(times[0])]
    out_v = [int(values[0])]
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
    return np.asarray(out_t), np.asarray(out_v)


def _rdp_indices(t: np.ndarray, v: np.ndarray, max_points: int) -> list[int]:
    """Return indices subsampled via a simple RDP-like algorithm."""

    if len(t) <= max_points:
        return list(range(len(t)))
    keep = [0, len(t) - 1]
    while len(keep) < max_points:
        max_dist = -1.0
        max_idx = None
        keep_sorted = sorted(keep)
        for a, b in zip(keep_sorted[:-1], keep_sorted[1:]):
            ta, tb = t[a], t[b]
            va, vb = v[a], v[b]
            dt = tb - ta
            dv = vb - va
            denom = dt * dt + dv * dv
            for i in range(a + 1, b):
                tt = t[i] - ta
                vv = v[i] - va
                if denom == 0:
                    dist = abs(vv)
                else:
                    proj = (tt * dt + vv * dv) / denom
                    proj_t = ta + proj * dt
                    proj_v = va + proj * dv
                    dist = ((t[i] - proj_t) ** 2 + (v[i] - proj_v) ** 2) ** 0.5
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i
        if max_idx is None:
            break
        keep.append(max_idx)
    return sorted(keep)


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
    ) -> None:
        import warnings

        self.times = np.asarray(times, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.domain = domain
        if offset_sec < 0.0:
            warnings.warn("offset_sec must be non-negative; clamping to 0.0")
            offset_sec = 0.0
        self.offset_sec = float(offset_sec)

    def validate(self) -> None:
        if self.times.size != self.values.size:
            raise ValueError("times and values must be same length")
        if self.times.size == 0:
            return
        if np.any(np.diff(self.times) < 0):
            raise ValueError("times must be non-decreasing")

    def _beats_to_times(
        self,
        beats: np.ndarray,
        tempo_map: float | Sequence[tuple[float, float]] | Callable[[float], float],
    ) -> np.ndarray:
        if callable(tempo_map):
            tempo = tempo_map
        elif isinstance(tempo_map, int | float):
            const = float(tempo_map)
            tempo = lambda _b: const  # noqa: E731
        else:
            tempo = tempo_map_from_events(tempo_map)
        times = [0.0]
        for a, b in zip(beats[:-1], beats[1:]):
            mid = (a + b) / 2.0
            bpm = tempo(mid)
            dt = (b - a) * 60.0 / bpm
            times.append(times[-1] + dt)
        return np.asarray(times)

    def _prep(
        self,
        tempo_map: float | Sequence[tuple[float, float]] | Callable[[float], float],
    ) -> tuple[np.ndarray, np.ndarray]:
        self.validate()
        t, v = dedupe_events(self.times, self.values)
        if self.domain == "beats":
            t = self._beats_to_times(t, tempo_map)
        return t, v

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
        """Render the curve as MIDI CC events.

        ``tempo_map`` may be a constant BPM number, a callable ``beat → bpm``
        function, or a list of ``(beat, bpm)`` pairs.  Endpoint events are
        preserved even when ``max_events`` trims intermediate points.
        ``resolution_hz`` is a deprecated alias for ``sample_rate_hz`` and is
        slated for removal in 6 months.
        """

        import warnings

        import pretty_midi

        if resolution_hz is not None and sample_rate_hz is None:
            warnings.warn(
                "resolution_hz is deprecated; use sample_rate_hz",
                DeprecationWarning,
                stacklevel=2,
            )
            sample_rate_hz = resolution_hz  # TODO: remove in 6 months

        if max_events is not None and max_events < 2:
            max_events = 2

        t, v = self._prep(tempo_map)
        if sample_rate_hz and sample_rate_hz > 0:
            t, v = _resample(t, v, float(sample_rate_hz))
        t, v = dedupe_events(t, v, value_eps=value_eps, time_eps=time_eps)
        vals = np.clip(np.rint(v), 0, 127).astype(int)
        t, vals = _dedupe_int(
            t, vals, time_eps=time_eps, keep_last=max_events is not None
        )
        if max_events is not None and len(vals) > max_events:
            idx = _rdp_indices(t, vals, max_events)
            t = t[idx]
            vals = vals[idx]
            t, vals = _dedupe_int(t, vals, time_eps=time_eps, keep_last=True)
        for tt, vv in zip(t, vals):
            inst.control_changes.append(
                pretty_midi.ControlChange(
                    number=int(cc_number),
                    value=int(vv),
                    time=float(tt + self.offset_sec),
                )
            )

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
        """Render the curve as MIDI pitch-bend events.

        Values are interpreted in ``units``: either ``"semitones"`` (scaled by
        ``bend_range_semitones``) or ``"normalized"`` where ``-1``..``1`` maps
        directly to the 14-bit bend range ``[-8192, 8191]``.  ``tempo_map`` has
        the same semantics as in :meth:`to_midi_cc`.  ``resolution_hz`` is a
        deprecated alias for ``sample_rate_hz`` and will be removed in 6 months.
        Endpoint events are preserved when ``max_events`` is specified.
        """

        import warnings

        import pretty_midi

        if resolution_hz is not None and sample_rate_hz is None:
            warnings.warn(
                "resolution_hz is deprecated; use sample_rate_hz",
                DeprecationWarning,
                stacklevel=2,
            )
            sample_rate_hz = resolution_hz  # TODO: remove in 6 months

        if max_events is not None and max_events < 2:
            max_events = 2

        t, v = self._prep(tempo_map)
        if sample_rate_hz and sample_rate_hz > 0:
            t, v = _resample(t, v, float(sample_rate_hz))
        t, v = dedupe_events(t, v, value_eps=value_eps, time_eps=time_eps)
        if units == "normalized":
            vals = np.clip(np.rint(v * 8192.0), -8192, 8191).astype(int)
        else:
            scale = 8192.0 / float(bend_range_semitones)
            vals = np.clip(np.rint(v * scale), -8192, 8191).astype(int)
        t, vals = _dedupe_int(
            t, vals, time_eps=time_eps, keep_last=max_events is not None
        )
        if max_events is not None and len(vals) > max_events:
            idx = _rdp_indices(t, vals, max_events)
            t = t[idx]
            vals = vals[idx]
            t, vals = _dedupe_int(t, vals, time_eps=time_eps, keep_last=True)
        for tt, vv in zip(t, vals):
            inst.pitch_bends.append(
                pretty_midi.PitchBend(pitch=int(vv), time=float(tt + self.offset_sec))
            )
