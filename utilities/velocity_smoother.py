from statistics import median
from typing import List


class EmaSmoother:
    def __init__(self, alpha_min: float = 0.15, alpha_max: float = 0.6, k: float = 8.0) -> None:
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.k = k
        self.history: List[int] = []
        self.value: float | None = None
        self.n_history = 8

    def reset(self) -> None:
        self.history.clear()
        self.value = None

    def _calc_alpha(self) -> float:
        if len(self.history) < 2:
            return self.alpha_min
        med = median(self.history)
        mad = median(abs(v - med) for v in self.history)
        if mad == 0:
            return self.alpha_min
        alpha = self.k * mad / 127.0
        return max(self.alpha_min, min(self.alpha_max, alpha))

    def smooth(self, raw_vel: int) -> int:
        self.history.append(int(raw_vel))
        if len(self.history) > self.n_history:
            self.history.pop(0)
        if self.value is None:
            self.value = float(raw_vel)
            return raw_vel
        alpha = self._calc_alpha()
        self.value = self.value + alpha * (raw_vel - self.value)
        return int(round(self.value))
