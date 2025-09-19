"""Pitch-bend math utilities for 14-bit MIDI values.

このモジュールは 14-bit のピッチベンド値と正規化値/半音値の相互変換を
集約し、丸め・クリップのポリシーを統一します。

Return-type policy:
    - NumPy が利用可能な場合: ``numpy.ndarray[int]`` を返す
    - NumPy が利用できない場合: ``list[int]`` もしくは ``int`` を返す

スケーリングは常に 8191 (= ``PB_MAX``) を用います。これにより
``-1`` → ``-8191``、``+1`` → ``+8191`` が厳密に一致します。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore

PB_MIN = -8191
PB_MAX = 8191
PB_FS = 8191.0
PITCHWHEEL_CENTER = PB_MAX + 1  # 8192
PITCHWHEEL_RAW_MAX = PITCHWHEEL_CENTER * 2 - 1  # 16383
PB_CENTER = PITCHWHEEL_CENTER
RAW_CENTER = PITCHWHEEL_CENTER
RAW_MAX = PITCHWHEEL_RAW_MAX
DELTA_MAX = PB_MAX

__all__ = [
    "PB_MIN",
    "PB_MAX",
    "PB_FS",
    "PITCHWHEEL_CENTER",
    "PITCHWHEEL_RAW_MAX",
    "PB_CENTER",
    "RAW_CENTER",
    "RAW_MAX",
    "DELTA_MAX",
    "norm_to_pb",
    "pb_to_norm",
    "semi_to_pb",
    "pb_to_semi",
    "semito_pb",
    "norm_to_raw",
    "raw_to_norm",
    "clip_delta",
    "BendRange",
]

Number = Union[int, float]


def _clip_int(x: Number) -> int:
    v = int(round(x))
    if v < PB_MIN:
        return PB_MIN
    if v > PB_MAX:
        return PB_MAX
    return v


def norm_to_pb(x: Union[Number, Sequence[Number]]):
    """正規化値 ``[-1..1]`` をピッチベンド値 ``[-8191..8191]`` へ変換する。"""

    if np is not None and hasattr(x, "__len__"):
        arr = np.asarray(x, dtype=float)
        out = np.rint(arr * PB_FS).astype(int)
        out = np.clip(out, PB_MIN, PB_MAX)
        return out
    if isinstance(x, (list, tuple)):
        return [_clip_int(v * PB_FS) for v in x]  # type: ignore[arg-type]
    return _clip_int(x * PB_FS)  # type: ignore[arg-type]


def pb_to_norm(x: Union[Number, Sequence[Number]]):
    """ピッチベンド値 ``[-8191..8191]`` を正規化値 ``[-1..1]`` に変換する。"""

    if np is not None and hasattr(x, "__len__"):
        arr = np.asarray(x, dtype=float)
        return np.clip(arr / PB_FS, -1.0, 1.0)
    if isinstance(x, (list, tuple)):
        return [max(-1.0, min(1.0, v / PB_FS)) for v in x]  # type: ignore[arg-type]
    val = x / PB_FS  # type: ignore[arg-type]
    return max(-1.0, min(1.0, val))


def semi_to_pb(semi: Union[Number, Sequence[Number]], bend_range_semi: Number):
    """半音値をピッチベンド値へ変換する。``±bend_range`` → ``±8191``。"""

    scale = PB_FS / float(bend_range_semi)
    if np is not None and hasattr(semi, "__len__"):
        arr = np.asarray(semi, dtype=float)
        out = np.rint(arr * scale).astype(int)
        out = np.clip(out, PB_MIN, PB_MAX)
        return out
    if isinstance(semi, (list, tuple)):
        return [_clip_int(v * scale) for v in semi]  # type: ignore[arg-type]
    return _clip_int(semi * scale)  # type: ignore[arg-type]


def pb_to_semi(pb: Union[Number, Sequence[Number]], bend_range_semi: Number):
    """ピッチベンド値を半音値へ変換する。``±8191`` → ``±bend_range``。"""

    inv = float(bend_range_semi) / PB_FS
    limit = float(bend_range_semi)
    if np is not None and hasattr(pb, "__len__"):
        arr = np.asarray(pb, dtype=float) * inv
        return np.clip(arr, -limit, limit)
    if isinstance(pb, (list, tuple)):
        return [max(-limit, min(limit, v * inv)) for v in pb]  # type: ignore[arg-type]
    val = float(pb) * inv
    return max(-limit, min(limit, val))


# Backwards compatibility alias (historical name)
semito_pb = semi_to_pb


def norm_to_raw(x: Union[Number, Sequence[Number]]):
    """正規化値 ``[-1..1]`` を 14-bit ピッチホイール生値 ``[0..16383]`` へ。"""

    if np is not None and hasattr(x, "__len__"):
        arr = np.asarray(x, dtype=float)
        out = np.rint(arr * PB_FS).astype(int)
        out = np.clip(out, PB_MIN, PB_MAX)
        return out + PITCHWHEEL_CENTER
    if isinstance(x, (list, tuple)):
        return [PITCHWHEEL_CENTER + _clip_int(v * PB_FS) for v in x]  # type: ignore[arg-type]
    return PITCHWHEEL_CENTER + _clip_int(x * PB_FS)  # type: ignore[arg-type]


def raw_to_norm(x: Union[Number, Sequence[Number]]):
    """14-bit ピッチホイール生値 ``[0..16383]`` を正規化値 ``[-1..1]`` へ。"""

    if np is not None and hasattr(x, "__len__"):
        arr = np.asarray(x, dtype=float)
        return np.clip((arr - PITCHWHEEL_CENTER) / PB_FS, -1.0, 1.0)
    if isinstance(x, (list, tuple)):
        return [max(-1.0, min(1.0, (v - PITCHWHEEL_CENTER) / PB_FS)) for v in x]  # type: ignore[arg-type]
    val = (float(x) - PITCHWHEEL_CENTER) / PB_FS
    return max(-1.0, min(1.0, val))


def clip_delta(delta: Number) -> int:
    """ピッチベンド中心からの差分値を ``±DELTA_MAX`` にクリップする。"""

    v = int(round(delta))
    if v < -DELTA_MAX:
        return -DELTA_MAX
    if v > DELTA_MAX:
        return DELTA_MAX
    return v


@dataclass(frozen=True)
class BendRange:
    """半音幅 ``semitones`` を基準としたセント ↔︎ 正規化変換。"""

    semitones: float

    def cents_to_norm(self, cents: Number) -> float:
        """セント値を ``[-1..1]`` の正規化値へ。"""

        if self.semitones <= 0:
            return 0.0
        span = float(self.semitones) * 100.0
        val = max(-span, min(span, float(cents)))
        return val / span

    def norm_to_cents(self, norm: Number) -> float:
        """正規化値を ``[-semitones*100 .. +semitones*100]`` セントへ。"""

        span = float(self.semitones) * 100.0
        val = max(-1.0, min(1.0, float(norm)))
        return val * span

