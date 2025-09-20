import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


PREDEFINED_CURVES = {
    "crescendo": [0.6, 0.8, 1.0],
    "decrescendo": [1.0, 0.8, 0.6],
}


def resolve_velocity_curve(curve_option):
    """Return a list of velocity scale factors for the given option."""
    if curve_option is None:
        return []
    if isinstance(curve_option, str):
        return PREDEFINED_CURVES.get(curve_option.lower(), [])
    if isinstance(curve_option, (list, tuple)):
        try:
            return [float(x) for x in curve_option]
        except Exception:
            return []
    return []


@lru_cache(maxsize=32)
def _interpolate_7pt_cached(curve_tuple, mode: str) -> list[float]:
    """Internal cached implementation operating on a tuple."""
    curve7 = list(curve_tuple)

    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - numpy may be missing
        np = None

    try:  # pragma: no cover - SciPy optional
        from scipy.interpolate import UnivariateSpline  # type: ignore
    except Exception:
        UnivariateSpline = None

    x_points = [i / 6 for i in range(7)]

    if mode == "spline" and np is not None and UnivariateSpline is not None:
        try:
            x = np.array(x_points, dtype=float)
            y = np.array(curve7, dtype=float)
            spline = UnivariateSpline(x, y, k=3, s=0.0)
            x_new = np.linspace(0.0, 1.0, 128)
            y_new = spline(x_new)
            result = [float(v) for v in y_new]
            result[0] = float(curve7[0])
            result[-1] = float(curve7[-1])
            return result
        except Exception:
            pass

    if np is not None:
        x = np.array(x_points)
        x_new = np.linspace(0.0, 1.0, 128)
        y_new = np.interp(x_new, x, curve7)
        result = [float(v) for v in y_new]
        result[-1] = float(curve7[-1])
        return result

    def lin_interp(x0, y0, x1, y1, x):
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    result = []
    for i in range(128):
        x_new = i / 127
        for j in range(6):
            if x_points[j] <= x_new <= x_points[j + 1]:
                result.append(
                    lin_interp(
                        x_points[j],
                        curve7[j],
                        x_points[j + 1],
                        curve7[j + 1],
                        x_new,
                    )
                )
                break
    if len(result) < 128:
        result.extend([float(curve7[-1])] * (128 - len(result)))
    if result:
        result[-1] = float(curve7[-1])
    return [float(v) for v in result]


def interpolate_7pt(curve7, *, mode: str = "spline") -> list[float]:
    """Interpolate a 7-point curve to 128 points.

    Parameters
    ----------
    mode:
        ``"spline"`` (default) uses a natural cubic spline when SciPy is
        available. ``"linear"`` forces simple linear interpolation.
        If SciPy is missing the function automatically falls back to the
        linear method.
    """
    if len(curve7) != 7:
        raise ValueError("curve7 must have length 7")

    mode = mode.lower()
    try:  # pragma: no cover - SciPy optional
        from scipy.interpolate import UnivariateSpline  # type: ignore
    except Exception:
        UnivariateSpline = None
    if mode == "spline" and UnivariateSpline is None:
        logger.warning("SciPy unavailable – falling back to linear interpolation")
        mode = "linear"
    mode = "linear" if mode == "linear" else "spline"
    return _interpolate_7pt_cached(tuple(float(v) for v in curve7), mode)
