import math
from typing import List, NamedTuple

class TimingBlend(NamedTuple):
    """Container for timing offset and velocity scaling."""

    offset_ql: float
    vel_scale: float


def interp_curve(curve: List[float], steps: int) -> List[float]:
    """Return ``curve`` resampled to ``steps`` points using linear interpolation."""
    if steps <= 0:
        return []
    if not curve:
        return [0.0] * steps
    if len(curve) == 1:
        return [float(curve[0])] * steps
    if len(curve) == steps:
        return [float(v) for v in curve]
    resampled: List[float] = []
    span = len(curve) - 1
    for i in range(steps):
        pos = i * span / float(steps - 1)
        idx = int(pos)
        frac = pos - idx
        v0 = float(curve[idx])
        v1 = float(curve[min(idx + 1, len(curve) - 1)])
        resampled.append(v0 * (1 - frac) + v1 * frac)
    return resampled


def interp_tempo(beat: float, curve: List[tuple[float, float]], default_bpm: float) -> float:
    """Return BPM at ``beat`` using ``curve`` with linear interpolation."""
    if not curve:
        return default_bpm
    curve = sorted(curve, key=lambda x: x[0])
    if beat <= curve[0][0]:
        return curve[0][1]
    if beat >= curve[-1][0]:
        return curve[-1][1]
    for i in range(1, len(curve)):
        b1, _ = curve[i]
        if beat < b1:
            b0, bpm0 = curve[i - 1]
            b1, bpm1 = curve[i]
            span = b1 - b0
            if span <= 0:
                return bpm1
            frac = (beat - b0) / span
            return bpm0 + (bpm1 - bpm0) * frac
    return curve[-1][1]


def _combine_timing(
    rel_offset: float,
    beat_len_ql: float,
    *,
    swing_ratio: float = 0.5,
    swing_type: str = "eighth",
    push_pull_curve: List[float] | None = None,
    tempo_bpm: float,
    max_push_ms: float = 80.0,
    vel_range: tuple[float, float] = (0.9, 1.1),
    return_vel: bool = False,
) -> float | TimingBlend:
    """Apply push-pull and swing timing with velocity scaling."""
    if beat_len_ql <= 0:
        result = TimingBlend(rel_offset, 1.0)
        return result if return_vel else result.offset_ql

    base_offset = rel_offset
    delta_push_ql = 0.0
    vel_scale = 1.0

    if push_pull_curve:
        try:
            ext_curve = list(push_pull_curve) + [push_pull_curve[0]]
            steps = len(push_pull_curve) * 4 + 1
            curve = interp_curve(ext_curve, steps)
            bar_len = beat_len_ql * len(push_pull_curve)
            pos = (rel_offset % bar_len) / bar_len * (steps - 1)
            idx = int(pos)
            frac = pos - idx
            val_ms = curve[idx] * (1 - frac) + curve[idx + 1] * frac
            delta_push_ql = (val_ms / 1000.0) * (tempo_bpm / 60.0)
            mag = min(abs(val_ms), max_push_ms) / float(max_push_ms)
            vmin, vmax = vel_range
            if val_ms >= 0:
                vel_scale = 1.0 + (vmax - 1.0) * mag
            else:
                vel_scale = 1.0 - (1.0 - vmin) * mag
        except Exception:
            delta_push_ql = 0.0
            vel_scale = 1.0

    delta_swing_ql = 0.0
    if abs(swing_ratio - 0.5) >= 1e-3:
        if swing_type == "eighth":
            subdivision_duration_ql = beat_len_ql / 2.0
        elif swing_type == "sixteenth":
            subdivision_duration_ql = beat_len_ql / 4.0
        else:
            subdivision_duration_ql = beat_len_ql
        if subdivision_duration_ql > 0:
            effective_beat_ql = subdivision_duration_ql * 2.0
            beat_num = math.floor(rel_offset / effective_beat_ql)
            within = rel_offset - beat_num * effective_beat_ql
            epsilon = subdivision_duration_ql * 0.1
            if abs(within - subdivision_duration_ql) < epsilon:
                swung = beat_num * effective_beat_ql + effective_beat_ql * swing_ratio
                delta_swing_ql = swung - rel_offset

    w = 0.5
    new_offset = base_offset + delta_swing_ql * (1.0 - w) + delta_push_ql * w
    new_offset = max(0.0, new_offset)

    result = TimingBlend(new_offset, vel_scale)
    return result if return_vel else result.offset_ql


def align_to_consonant(
    offset_ql: float,
    peaks_sec: list[float],
    bpm: float,
    lag_ms: float = 10.0,
    window_beats: float = 0.25,
) -> float:
    """Return ``offset_ql`` shifted earlier if a peak is nearby.

    Parameters
    ----------
    offset_ql : float
        Note position in beats.
    peaks_sec : list[float]
        Absolute consonant peak times in seconds.
    bpm : float
        Current tempo in beats per minute.
    lag_ms : float, optional
        Amount to shift earlier when aligned, by default ``10.0``.
    window_beats : float, optional
        Search window around ``offset_ql`` in beats, by default ``0.25``.

    Returns
    -------
    float
        Corrected offset. If no peak is found within ``window_beats`` of the
        position, the original value is returned.
    """

    if not peaks_sec:
        return offset_ql

    sec_per_beat = 60.0 / bpm
    offset_sec = offset_ql * sec_per_beat
    window_sec = abs(window_beats) * sec_per_beat
    lag_sec = lag_ms / 1000.0

    closest_peak: float | None = None
    min_abs_diff: float | None = None
    for peak in peaks_sec:
        diff = peak - offset_sec
        if abs(diff) <= window_sec:
            if min_abs_diff is None or abs(diff) < min_abs_diff:
                min_abs_diff = abs(diff)
                closest_peak = peak

    if closest_peak is not None:
        corrected_sec = max(0.0, closest_peak - lag_sec)
        return corrected_sec / sec_per_beat

    return offset_ql
