"""Pitch-bend conversion helpers.

Mappings use ``PB_MAX`` (=8191) rather than 8192 so that ±range inputs map
exactly to ±8191. Array code paths apply ``numpy.rint`` (banker's rounding)
while scalar paths use ``round``.
"""

from __future__ import annotations
from typing import Iterable, Union, Sequence, overload, List
from collections.abc import Iterable as ABCIterable

try:
    import numpy as np  # optional
    from numpy.typing import NDArray
except Exception:  # pragma: no cover
    np = None  # type: ignore
    NDArray = List  # type: ignore

PB_MIN = -8191
PB_MAX = 8191
PB_FS = 8191.0  # full-scale ("PB full scale")
PITCHWHEEL_CENTER = int(PB_MAX + 1)
PITCHWHEEL_RAW_MAX = int(PITCHWHEEL_CENTER * 2 - 1)

__all__ = [
    "PB_MIN",
    "PB_MAX",
    "PB_FS",
    "PITCHWHEEL_CENTER",
    "PITCHWHEEL_RAW_MAX",
    "norm_to_pb",
    "pb_to_norm",
    "semi_to_pb",
    "pb_to_semi",
]


def _as_array(vals: Sequence[float]):
    return np.asarray(vals, dtype=float)


def _clip_round_scalar(x: float) -> int:
    if x < PB_MIN:
        x = PB_MIN
    elif x > PB_MAX:
        x = PB_MAX
    return int(round(x))


def _clip_round_vec(x):
    x = np.clip(x, PB_MIN, PB_MAX)
    return np.rint(x).astype(int)


def _clip_scalar(x: float) -> float:
    if x < PB_MIN:
        return float(PB_MIN)
    if x > PB_MAX:
        return float(PB_MAX)
    return float(x)


# Overloads help type checkers distinguish between scalar, list, and ndarray paths
@overload
def norm_to_pb(x: NDArray[np.floating]) -> NDArray[np.int_]:
    ...

@overload
def norm_to_pb(x: Iterable[float]) -> List[int]:
    ...

@overload
def norm_to_pb(x: float) -> int:
    ...

def norm_to_pb(x: Union[float, Iterable[float]]):
    """Convert normalized values ``[-1..1]`` to pitch-bend integers ``[-8191..8191]``.

    Array inputs use ``numpy.rint`` while scalars use ``round``. Returns
    ``np.ndarray`` if NumPy is available; otherwise returns ``list[int]`` or
    ``int`` for scalar input.
    """
    if np is None:
        if isinstance(x, ABCIterable) and not isinstance(x, (str, bytes)):
            return [_clip_round_scalar(float(v) * PB_FS) for v in x]
        return _clip_round_scalar(float(x) * PB_FS)
    else:
        if np.isscalar(x):
            return _clip_round_scalar(float(x) * PB_FS)
        arr = _as_array(x) * PB_FS
        return _clip_round_vec(arr)


@overload
def semi_to_pb(x: NDArray[np.floating], semi_range: float) -> NDArray[np.int_]:
    ...

@overload
def semi_to_pb(x: Iterable[float], semi_range: float) -> List[int]:
    ...

@overload
def semi_to_pb(x: float, semi_range: float) -> int:
    ...

def semi_to_pb(x: Union[float, Iterable[float]], semi_range: float):
    """Map semitone values ``[-semi_range..semi_range]`` to pitch-bend integers.

    Array inputs use ``numpy.rint`` while scalars use ``round``.
    """
    if semi_range == 0:
        if isinstance(x, ABCIterable) and not isinstance(x, (str, bytes)):
            return [0 for _ in x]
        return 0
    scale = PB_FS / float(semi_range)
    if np is None:
        if isinstance(x, ABCIterable) and not isinstance(x, (str, bytes)):
            return [_clip_round_scalar(float(v) * scale) for v in x]
        return _clip_round_scalar(float(x) * scale)
    else:
        if np.isscalar(x):
            return _clip_round_scalar(float(x) * scale)
        arr = _as_array(x) * scale
        return _clip_round_vec(arr)


@overload
def pb_to_norm(pb: NDArray[np.int_]) -> NDArray[np.floating]:
    ...

@overload
def pb_to_norm(pb: Iterable[int]) -> List[float]:
    ...

@overload
def pb_to_norm(pb: int) -> float:
    ...

def pb_to_norm(pb: Union[int, Iterable[int]]):
    """Convert pitch-bend integers to normalized floats ``[-1..1]``.

    Values outside ``[PB_MIN, PB_MAX]`` are clipped. Returns ``np.ndarray`` if
    NumPy is available; otherwise returns ``list[float]`` or ``float`` for
    scalar input.
    """
    if np is None:
        if isinstance(pb, ABCIterable) and not isinstance(pb, (str, bytes)):
            return [_clip_scalar(v) / PB_FS for v in pb]
        return _clip_scalar(pb) / PB_FS
    else:
        if np.isscalar(pb):
            return _clip_scalar(pb) / PB_FS
        arr = _as_array(pb)
        arr = np.clip(arr, PB_MIN, PB_MAX)
        return arr / PB_FS


@overload
def pb_to_semi(pb: NDArray[np.int_], semi_range: float) -> NDArray[np.floating]:
    ...

@overload
def pb_to_semi(pb: Iterable[int], semi_range: float) -> List[float]:
    ...

@overload
def pb_to_semi(pb: int, semi_range: float) -> float:
    ...

def pb_to_semi(pb: Union[int, Iterable[int]], semi_range: float):
    """Convert pitch-bend integers to semitone values.

    Values outside ``[PB_MIN, PB_MAX]`` are clipped.
    """
    if semi_range == 0:
        if isinstance(pb, ABCIterable) and not isinstance(pb, (str, bytes)):
            return [0 for _ in pb]
        return 0
    if np is None:
        if isinstance(pb, ABCIterable) and not isinstance(pb, (str, bytes)):
            return [_clip_scalar(v) / PB_FS * semi_range for v in pb]
        return _clip_scalar(pb) / PB_FS * semi_range
    else:
        if np.isscalar(pb):
            return _clip_scalar(pb) / PB_FS * semi_range
        arr = _as_array(pb)
        arr = np.clip(arr, PB_MIN, PB_MAX)
        return arr / PB_FS * semi_range
