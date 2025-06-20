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
