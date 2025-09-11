from __future__ import annotations

"""Pitch‑bend math helpers.

Arrays use :func:`numpy.rint` (banker's rounding); scalars use :func:`round()`.
All conversions clip to ``[PB_MIN..PB_MAX]`` so ``±range`` maps to ``±8191`` exactly.
Functions return ``np.ndarray[int]`` when numpy is available and the input is
array‑like, otherwise ``list[int]`` or ``int``.
"""

from dataclasses import dataclass

try:  # numpy optional
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional
    np = None  # type: ignore

PB_MIN = -8191
PB_MAX = 8191
PB_CENTER = 8192  # backward compat
PB_FS = float(PB_MAX)
PITCHWHEEL_CENTER = PB_CENTER
PITCHWHEEL_RAW_MAX = PB_CENTER * 2 - 1
RAW_CENTER = PITCHWHEEL_CENTER  # legacy name
RAW_MAX = PITCHWHEEL_RAW_MAX  # legacy name
DELTA_MAX = PB_MAX  # legacy name
# Legacy aliases (do not remove; tests & old call sites may import these)
PBMIN = PB_MIN
PBMAX = PB_MAX
__all__ = [
    "PB_MIN",
    "PB_MAX",
    "PBMIN",
    "PBMAX",
    "PB_CENTER",
    "PB_FS",
    "PITCHWHEEL_CENTER",
    "PITCHWHEEL_RAW_MAX",
    "RAW_CENTER",
    "RAW_MAX",
    "DELTA_MAX",
    "clip_delta",
    "norm_to_raw",
    "raw_to_norm",
    "norm_to_pb",
    "pb_to_norm",
    "semi_to_pb",
    "pb_to_semi",
    "BendRange",
]
def clip_delta(delta: int) -> int:
    if delta > PB_MAX:
        return PB_MAX
    if delta < PB_MIN:
        return PB_MIN
    return delta


def norm_to_raw(norm: float) -> int:
    """Map normalized ``[-1,1]`` to raw 14‑bit ``0..16383`` (center ``8192``).

    Uses :func:`round` and clips to valid domain.
    """
    if norm > 1.0:
        norm = 1.0
    elif norm < -1.0:
        norm = -1.0
    delta = clip_delta(int(round(norm * PB_MAX)))
    raw = PITCHWHEEL_CENTER + delta
    if raw < 0:
        raw = 0
    elif raw > PITCHWHEEL_RAW_MAX:
        raw = PITCHWHEEL_RAW_MAX
    return raw


def raw_to_norm(raw: int) -> float:
    """Map raw 14‑bit value ``0..16383`` to normalized ``[-1,1]`` (center 0).

    Clips raw to valid range before conversion.
    """
    if raw < 0:
        raw = 0
    elif raw > PITCHWHEEL_RAW_MAX:
        raw = PITCHWHEEL_RAW_MAX
    delta = raw - PITCHWHEEL_CENTER
    return float(delta) / PB_FS


def norm_to_pb(vals):
    """Convert normalized ``[-1,1]`` values to pitch‑bend integers.

    Arrays use :func:`numpy.rint`; scalars use :func:`round`.
    Returns ``np.ndarray[int]`` when numpy is available and ``vals`` is
    array‑like, otherwise ``list[int]``/``int``.
    """
    if np is not None and hasattr(vals, "__len__"):
        arr = np.asarray(vals, dtype=float)
        arr = np.clip(arr, -1.0, 1.0)
        return np.clip(np.rint(arr * PB_MAX), PB_MIN, PB_MAX).astype(int)
    if isinstance(vals, (list, tuple)):
        return [norm_to_pb(v) for v in vals]
    v = float(vals)
    if v > 1.0:
        v = 1.0
    elif v < -1.0:
        v = -1.0
    val = int(round(v * PB_MAX))
    if val > PB_MAX:
        val = PB_MAX
    elif val < PB_MIN:
        val = PB_MIN
    return val


def pb_to_norm(vals):
    """Convert pitch‑bend integers to normalized ``[-1,1]`` floats.

    Returns ``np.ndarray`` when numpy is available and ``vals`` is array‑like,
    else ``list``/``float``.
    """
    if np is not None and hasattr(vals, "__len__"):
        arr = np.asarray(vals, dtype=float)
        arr = np.clip(arr, PB_MIN, PB_MAX)
        return arr / PB_MAX
    if isinstance(vals, (list, tuple)):
        return [pb_to_norm(v) for v in vals]
    v = float(vals)
    if v > PB_MAX:
        v = PB_MAX
    elif v < PB_MIN:
        v = PB_MIN
    return v / PB_MAX


def semi_to_pb(vals, bend_range_semitones):
    """Convert semitone offsets to pitch‑bend integers.

    ``bend_range_semitones`` is the ±range; endpoints map to ±8191 exactly.
    """
    scale = PB_MAX / float(bend_range_semitones) if bend_range_semitones else 0.0
    if np is not None and hasattr(vals, "__len__"):
        arr = np.asarray(vals, dtype=float)
        arr = np.clip(arr, -bend_range_semitones, bend_range_semitones)
        return np.clip(np.rint(arr * scale), PB_MIN, PB_MAX).astype(int)
    if isinstance(vals, (list, tuple)):
        return [semi_to_pb(v, bend_range_semitones) for v in vals]
    v = float(vals)
    if v > bend_range_semitones:
        v = bend_range_semitones
    elif v < -bend_range_semitones:
        v = -bend_range_semitones
    val = int(round(v * scale))
    if val > PB_MAX:
        val = PB_MAX
    elif val < PB_MIN:
        val = PB_MIN
    return val


def pb_to_semi(vals, bend_range_semitones):
    """Convert pitch‑bend integers to semitone offsets.

    ``bend_range_semitones`` is the ±range; inputs clip to ``PB_MIN``..``PB_MAX``.
    """
    scale = float(bend_range_semitones) / PB_MAX if PB_MAX else 0.0
    if np is not None and hasattr(vals, "__len__"):
        arr = np.asarray(vals, dtype=float)
        arr = np.clip(arr, PB_MIN, PB_MAX)
        return arr * scale
    if isinstance(vals, (list, tuple)):
        return [pb_to_semi(v, bend_range_semitones) for v in vals]
    v = float(vals)
    if v > PB_MAX:
        v = PB_MAX
    elif v < PB_MIN:
        v = PB_MIN
    return v * scale


@dataclass(frozen=True)
class BendRange:
    semitones: float = 2.0

    def cents_to_norm(self, cents: float) -> float:
        """Convert cents offset to normalized [-1,1] given bend range.
        Center maps to 0.
        """
        max_cents = self.semitones * 100.0
        if max_cents <= 0:
            return 0.0
        v = cents / max_cents
        if v > 1.0:
            v = 1.0
        elif v < -1.0:
            v = -1.0
        return v

    def norm_to_cents(self, norm: float) -> float:
        max_cents = self.semitones * 100.0
        if norm > 1.0:
            norm = 1.0
        elif norm < -1.0:
            norm = -1.0
        return norm * max_cents

