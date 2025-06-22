from __future__ import annotations
from collections import deque
from statistics import median


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
