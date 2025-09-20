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
from typing import Literal

try:  # optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore
import pretty_midi
from . import pb_math

__all__ = [
    "ControlCurve",
    "catmull_rom_monotone",
    "dedupe_events",
    "dedupe_times_values",
    "enforce_strictly_increasing",
    "ensure_scalar_floats",
    "tempo_map_from_prettymidi",
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
        return np.clip(np.asarray(xs, dtype=float), lo, hi)

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
    eps: float | None = None,
    value_eps: float = 1e-6,
    time_eps: float = 1e-9,
):
    """Deduplicate nearly-identical consecutive ``times``/``values`` pairs.

    Keeps the first sample in any run of (nearly) equal events.
    Returns numpy arrays when numpy is available, otherwise Python lists.
    """

    if len(times) == 0:
        if np is not None:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        return [], []

    if eps is not None:
        value_eps = time_eps = float(eps)

    out_t: list[float] = [float(times[0])]
    out_v: list[float] = [float(values[0])]
    for t, v in zip(times[1:], values[1:]):
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


def enforce_strictly_increasing(times: Sequence[float], eps: float) -> list[float]:
    """Ensure each successive time increases by at least ``eps``."""

    if len(times) == 0:
        return []
    out = [float(times[0])]
    for t in times[1:]:
        t = float(t)
        if t <= out[-1] + eps:
            t = out[-1] + eps
        out.append(t)
    return out


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


def tempo_map_from_prettymidi(pm: pretty_midi.PrettyMIDI) -> Callable[[float], float]:
    """Return a tempo map callable derived from a PrettyMIDI object."""

    times, bpms = pm.get_tempo_changes()
    beats = [pm.time_to_beat(t) for t in times]
    events = list(zip(beats, bpms))
    return tempo_map_from_events(events)


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


def _uniform_indices(length: int, target: int) -> list[int]:
    """Return evenly spaced indices keeping first/last and removing duplicates."""

    if length <= 0 or target <= 0:
        return []
    if target >= length:
        return list(range(length))
    if length == 1:
        return [0]
    if target == 1:
        return [length - 1]

    step = (length - 1) / float(target - 1)
    out: list[int] = [0]
    last = 0
    for i in range(1, target - 1):
        idx = int(round(i * step))
        if idx <= last:
            idx = last + 1
        if idx >= length - 1:
            break
        out.append(idx)
        last = idx
    if out[-1] != length - 1:
        out.append(length - 1)
    return out


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
    dedup_time_epsilon: float = 1e-4
    dedup_value_epsilon: float = 1.0
    target: Literal["cc11", "cc64", "bend"] = "cc11"

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
            if self.sample_rate_hz:
                warnings.warn("resolution_hz ignored", DeprecationWarning, stacklevel=2)
            else:
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
        tempo_map: float | Sequence[tuple[float, float]] | Callable[[float], float] | TempoMap,
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
        if not callable(tempo_map) and not isinstance(tempo_map, int | float):
            tm = TempoMap(tempo_map)
            beats_list = (
                beats
                if isinstance(beats, list)
                else (beats.tolist() if np is not None else list(beats))
            )
            return [tm.sec_at(b) for b in beats_list]
        if callable(tempo_map):
            tempo = tempo_map
        else:
            const = float(tempo_map)
            if const <= 0 or not math.isfinite(const):
                raise ValueError("bpm must be positive and finite")
            tempo = lambda _b: const  # noqa: E731
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
        tempo_map: float | Sequence[tuple[float, float]] | Callable[[float], float] | TempoMap,
        *,
        fold_halves: bool = False,
        value_eps: float | None = None,
        time_eps: float | None = None,
    ):
        self.validate()
        ve = self.dedup_value_epsilon if value_eps is None else float(value_eps)
        te = self.dedup_time_epsilon if time_eps is None else float(time_eps)
        t, v = dedupe_events(self.times, self.values, value_eps=ve, time_eps=te)
        if self.domain == "beats":
            t = self._beats_to_times(t, tempo_map, fold_halves=fold_halves)
        return t, v

    def sample(
        self,
        query_times: Sequence[float],
        *,
        tempo_map: float | Sequence[tuple[float, float]] | Callable[[float], float] = 120.0,
        fold_halves: bool = False,
    ) -> list[float]:
        """Return interpolated values at ``query_times``."""

        t, v = self._prep(tempo_map, fold_halves=fold_halves)
        return catmull_rom_monotone(t, v, query_times)

    # ---- MIDI CC rendering -----------------------------------------
    def to_midi_cc(
        self,
        inst,
        cc_number: int,
        *,
        time_offset: float = 0.0,
        tempo_map: float | Sequence[tuple[float, float]] | Callable[[float], float] = 120.0,
        sample_rate_hz: float | None = None,
        resolution_hz: float | None = None,
        max_events: int | None = None,
        value_eps: float = 1e-6,
        time_eps: float = 1e-9,
        dedup_value_epsilon: float | None = None,
        dedup_time_epsilon: float | None = None,
        min_delta: float | None = None,
        fold_halves: bool = False,
        simplify_mode: Literal["rdp", "uniform"] = "rdp",
    ) -> None:
        """Render the curve as MIDI CC events onto ``inst``.

        ``tempo_map`` may be a constant BPM number, a callable ``beat → bpm``
        function, or a list of ``(beat, bpm)`` pairs.  Endpoint events are
        preserved even when ``max_events`` trims intermediate points.
        ``resolution_hz`` is a deprecated alias for ``sample_rate_hz`` and is
        slated for removal in a future release.
        """

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

        if max_events is None:
            max_events = 2000
        elif max_events < 2:
            max_events = 2

        orig_len = len(self.times) if np is None else int(len(self.times))
        t, v = self._prep(
            tempo_map,
            fold_halves=fold_halves,
            value_eps=dedup_value_epsilon,
            time_eps=dedup_time_epsilon,
        )
        if sample_rate_hz and sample_rate_hz > 0:
            t, v = _resample(t, v, float(sample_rate_hz))
        t, v = dedupe_events(t, v, value_eps=value_eps, time_eps=time_eps)
        t = enforce_strictly_increasing(t.tolist() if hasattr(t, "tolist") else t, time_eps)
        if value_eps > 0 and len(v) >= 2:
            if max(v) - min(v) <= value_eps:
                if orig_len > 2:
                    t, v = [t[0], t[-1]], [v[0], v[-1]]
                else:
                    t, v = [t[0]], [v[0]]
            else:
                ft: list[float] = [t[0]]
                fv: list[float] = [v[0]]
                for i in range(1, len(v)):
                    if abs(v[i] - fv[-1]) > value_eps:
                        ft.append(t[i])
                        fv.append(v[i])
                if ft[-1] != t[-1] and (abs(v[-1] - fv[-1]) > value_eps or len(ft) == 1):
                    ft.append(t[-1])
                    fv.append(v[-1])
                t, v = ft, fv
        if max_events is not None and len(t) > max_events:
            if simplify_mode == "rdp":
                cc = ControlCurve.from_dense(
                    t,
                    v,
                    tol=1e-9,
                    max_knots=max_events,
                    units=self.units,
                    domain="time",
                )
                t = list(cc.times)
                v = list(cc.values)
            else:
                t_list = t.tolist() if hasattr(t, "tolist") else list(t)
                v_list = v.tolist() if hasattr(v, "tolist") else list(v)
                idxs = _uniform_indices(len(t_list), max_events)
                t = [t_list[i] for i in idxs]
                v = [v_list[i] for i in idxs]
        # clamp + round to MIDI domain
        vals = round_int(clip(v, 0, 127))
        t, vals = _dedupe_int(t, vals, time_eps=time_eps, keep_last=True)
        if value_eps > 0 and len(vals) > 1:
            ft: list[float] = [t[0]]
            fv: list[int] = [int(vals[0])]
            for tt, vv in zip(t[1:], vals[1:]):
                if vv == fv[-1]:
                    continue
                ft.append(tt)
                fv.append(int(vv))
            if ft[-1] != t[-1] or fv[-1] != int(vals[-1]):
                ft.append(t[-1])
                fv.append(int(vals[-1]))
            t, vals = ft, fv
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
            t_list = t.tolist() if hasattr(t, "tolist") else list(t)
            val_list = vals.tolist() if hasattr(vals, "tolist") else list(vals)
            idxs = _uniform_indices(len(val_list), max_events)
            t = [t_list[i] for i in idxs]
            vals = [int(val_list[i]) for i in idxs]
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
        tempo_map: float | Sequence[tuple[float, float]] | Callable[[float], float] = 120.0,
        bend_range_semitones: float = 2.0,
        sample_rate_hz: float | None = None,
        resolution_hz: float | None = None,
        max_events: int | None = None,
        value_eps: float = 1e-6,
        time_eps: float = 1e-9,
        dedup_value_epsilon: float | None = None,
        dedup_time_epsilon: float | None = None,
        units: str = "semitones",
        fold_halves: bool = False,
        simplify_mode: Literal["rdp", "uniform"] = "rdp",
    ) -> None:
        """Render the curve as MIDI pitch-bend events onto ``inst``.

        Values are interpreted in ``units``: either ``"semitones"`` (scaled by
        ``bend_range_semitones``) or ``"normalized"`` where ``-1``..``1`` maps
        directly to the 14-bit bend range ``[-8191, 8191]``.  ``tempo_map`` has
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

        if max_events is None:
            max_events = 2000
        elif max_events < 2:
            max_events = 2

        orig_len = len(self.times) if np is None else int(len(self.times))
        t, v = self._prep(
            tempo_map,
            fold_halves=fold_halves,
            value_eps=dedup_value_epsilon,
            time_eps=dedup_time_epsilon,
        )
        if sample_rate_hz and sample_rate_hz > 0:
            t, v = _resample(t, v, float(sample_rate_hz))
        t, v = dedupe_events(t, v, value_eps=value_eps, time_eps=time_eps)
        t = enforce_strictly_increasing(t.tolist() if hasattr(t, "tolist") else t, time_eps)
        if value_eps > 0 and len(v) >= 2:
            if max(v) - min(v) <= value_eps:
                if orig_len > 2:
                    t, v = [t[0], t[-1]], [v[0], v[-1]]
                else:
                    t, v = [t[0]], [v[0]]
            else:
                ft = [t[0]]
                fv = [v[0]]
                for i in range(1, len(v)):
                    if abs(v[i] - fv[-1]) > value_eps:
                        ft.append(t[i])
                        fv.append(v[i])
                if ft[-1] != t[-1] and (abs(v[-1] - fv[-1]) > value_eps or len(ft) == 1):
                    ft.append(t[-1])
                    fv.append(v[-1])
                t, v = ft, fv
        if max_events is not None and len(t) > max_events:
            if simplify_mode == "rdp":
                cc = ControlCurve.from_dense(
                    t,
                    v,
                    tol=1e-9,
                    max_knots=max_events,
                    units=self.units,
                    domain="time",
                )
                t = list(cc.times)
                v = list(cc.values)
            else:
                t_list = t.tolist() if hasattr(t, "tolist") else list(t)
                v_list = v.tolist() if hasattr(v, "tolist") else list(v)
                idxs = _uniform_indices(len(t_list), max_events)
                t = [t_list[i] for i in idxs]
                v = [v_list[i] for i in idxs]
        # convert to 14-bit domain
        if units == "normalized":
            vals = pb_math.norm_to_pb(v)
        else:
            vals = pb_math.semi_to_pb(v, bend_range_semitones)
        t, vals = _dedupe_int(t, vals, time_eps=time_eps, keep_last=True)
        if value_eps > 0 and len(vals) > 1:
            ft: list[float] = [t[0]]
            fv: list[int] = [int(vals[0])]
            for tt, vv in zip(t[1:], vals[1:]):
                if vv == fv[-1]:
                    continue
                ft.append(tt)
                fv.append(int(vv))
            if ft[-1] != t[-1] or fv[-1] != int(vals[-1]):
                ft.append(t[-1])
                fv.append(int(vals[-1]))
            t, vals = ft, fv
        if max_events is not None and len(vals) > max_events:
            t_list = t.tolist() if hasattr(t, "tolist") else list(t)
            val_list = vals.tolist() if hasattr(vals, "tolist") else list(vals)
            idxs = _uniform_indices(len(val_list), max_events)
            t = [t_list[i] for i in idxs]
            vals = [int(val_list[i]) for i in idxs]
        if not isinstance(t, list):
            t = t.tolist() if hasattr(t, "tolist") else list(t)
        if not isinstance(vals, list):
            vals = vals.tolist() if hasattr(vals, "tolist") else list(vals)
        if self.ensure_zero_at_edges and len(vals) > 0:
            orig_len = len(vals)
            if vals[0] != 0:
                t.insert(0, t[0])
                vals.insert(0, 0)
            if vals[-1] != 0:
                if orig_len == 1:
                    vals[-1] = 0
                elif orig_len == 2:
                    t.append(t[-1])
                    vals.append(0)
                else:
                    vals[-1] = 0
        if max_events is not None and len(vals) > max_events:
            idxs = _uniform_indices(len(vals), max_events)
            t = [t[i] for i in idxs]
            vals = [int(vals[i]) for i in idxs]
        for tt, vv in zip(t, vals):
            inst.pitch_bends.append(
                pretty_midi.PitchBend(pitch=int(vv), time=float(tt + self.offset_sec + time_offset))
            )

    # ---- utils ------------------------------------------------------
    @staticmethod
    def convert_to_14bit(
        values: Sequence[float],
        range_semitones: float,
        units: str = "semitones",
    ) -> Sequence[int]:
        """Convert ``values`` to 14-bit pitch-bend integers.

        ``units`` can be ``"semitones"`` (default) or ``"normalized"`` where
        ``-1``..``1`` maps to ``[-8191, 8191]``. Returns ``np.ndarray[int]`` when
        NumPy is available; otherwise ``list[int]``.
        """

        vals = ensure_scalar_floats(values)
        if units == "normalized":
            return pb_math.norm_to_pb(vals)
        else:
            return pb_math.semi_to_pb(vals, range_semitones)

    # ---- simplification --------------------------------------------
    @staticmethod
    def from_dense(
        times: Sequence[float],
        values: Sequence[float],
        *,
        tol: float = 1.5,
        max_knots: int = 256,
        target: Literal["cc11", "cc64", "bend"] = "cc11",
        units: str | None = None,
        domain: str = "time",
    ) -> ControlCurve:
        """Create :class:`ControlCurve` from a dense sequence.

        ``tol`` controls the maximum deviation allowed during
        Douglas–Peucker simplification.  ``max_knots`` caps the number of
        returned knots.  ``target`` is stored on the returned curve for
        downstream routing.
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
            simplified = simplified[: max_knots - 1] + [simplified[-1]]
        t_s, v_s = zip(*simplified)
        return ControlCurve(
            list(t_s),
            list(v_s),
            units=units,
            domain=domain,
            target=target,
        )
