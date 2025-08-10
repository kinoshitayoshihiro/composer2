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
