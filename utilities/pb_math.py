from __future__ import annotations

"""Pitch‑bend math utilities (14‑bit), with explicit rounding/clip policy.

Policy
- RAW range: 0..16383, center=8192
- Signed delta in LSB: [-8191, +8191]
- Normalized float domain: [-1.0, +1.0] maps to [-8191, +8191]
- Rounding: banker's not desired; use round‑half‑away via int(x+0.5*sign)
"""

from dataclasses import dataclass

RAW_MAX = 16383
RAW_CENTER = 8192
DELTA_MAX = 8191  # max magnitude from center


def _round_half_away(x: float) -> int:
    if x >= 0:
        return int(x + 0.5)
    return int(x - 0.5)


def clip_delta(delta: int) -> int:
    if delta > DELTA_MAX:
        return DELTA_MAX
    if delta < -DELTA_MAX:
        return -DELTA_MAX
    return delta


def norm_to_raw(norm: float) -> int:
    """Map normalized [-1,1] to raw 14‑bit 0..16383 with center 8192.

    Rounds half away from zero; clips to domain.
    """
    if norm > 1.0:
        norm = 1.0
    elif norm < -1.0:
        norm = -1.0
    delta = _round_half_away(norm * DELTA_MAX)
    delta = clip_delta(delta)
    raw = RAW_CENTER + delta
    if raw < 0:
        raw = 0
    elif raw > RAW_MAX:
        raw = RAW_MAX
    return raw


def raw_to_norm(raw: int) -> float:
    """Map raw 14‑bit value (0..16383) to normalized [-1,1] with center 0.
    Clips raw to valid range before conversion.
    """
    if raw < 0:
        raw = 0
    elif raw > RAW_MAX:
        raw = RAW_MAX
    delta = raw - RAW_CENTER
    return float(delta) / float(DELTA_MAX)


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

