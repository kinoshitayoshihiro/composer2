from __future__ import annotations
# deque is required for the EMA history buffer
from collections import deque
from statistics import median
import math


class VelocitySmoother:
    def __init__(
        self,
        window: int = 8,
        *,
        alpha_min: float = 0.15,
        alpha_max: float = 0.85,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if alpha_min <= 0 or alpha_max <= 0 or alpha_min > alpha_max:
            raise ValueError("invalid alpha range")
        self.window = window
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.history: deque[int] = deque(maxlen=window)
        self.value: float | None = None

    def reset(self) -> None:
        """Clear internal state before starting a new phrase."""
        self.history.clear()
        self.value = None

    def _alpha(self) -> float:
        if len(self.history) < 2:
            return 1.0
        med = median(self.history)
        max_dev = max(abs(v - med) for v in self.history)
        alpha = max_dev / 25.0
        if alpha < self.alpha_min:
            alpha = self.alpha_min
        elif alpha > self.alpha_max:
            alpha = self.alpha_max
        return alpha

    def smooth(self, raw: int) -> int:
        raw = int(raw)
        self.history.append(raw)
        if self.value is None:
            self.value = float(raw)
            return max(1, min(127, raw))
        alpha = self._alpha()
        self.value = self.value + alpha * (raw - self.value)
        result = int(round(self.value))
        return max(1, min(127, result))


class EMASmoother:
    """Exponential moving average velocity smoother with adaptive ``alpha``."""

    def __init__(self, initial_alpha: float = 0.5, window: int = 16) -> None:
        if not 0 < initial_alpha <= 1:
            raise ValueError("initial_alpha must be in (0, 1]")
        if window <= 0:
            raise ValueError("window must be positive")
        self.alpha = float(initial_alpha)
        self.window = window
        self.value: float | None = None
        self.history: deque[int] = deque(maxlen=window)

    def reset(self) -> None:
        self.value = None
        self.history.clear()

    @staticmethod
    def alpha_for_window(vals: list[int]) -> float:
        """Return smoothing coefficient based on median absolute deviation."""
        if not vals:
            return 0.5
        med = median(vals)
        dev = [abs(v - med) for v in vals]
        mad = median(dev)
        # sigmoid based scaling
        sig = 1.0 / (1.0 + math.exp(-mad / 20.0))
        alpha = 0.15 + 0.6 * sig
        if alpha < 0.15:
            alpha = 0.15
        elif alpha > 0.75:
            alpha = 0.75
        return alpha

    def smooth(self, raw: int) -> int:
        """Return smoothed velocity after updating internal state."""

        raw = int(raw)
        self.history.append(raw)
        if self.value is None:
            self.value = float(raw)
            return max(1, min(127, raw))
        self.alpha = self.alpha_for_window(list(self.history))
        self.value = self.value + self.alpha * (raw - self.value)
        result = int(round(self.value))
        return max(1, min(127, result))

    # compatibility alias for older callers
    def update(self, raw: int) -> int:  # noqa: D401 - simple wrapper
        return self.smooth(raw)
