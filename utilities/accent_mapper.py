from __future__ import annotations

import random
from typing import Dict


class AccentMapper:
    """Map accent intensity and ghost-hat density using a vocal heatmap."""

    def __init__(self, heatmap: Dict[int, int] | None, global_settings: Dict[str, object] | None, *, rng: random.Random | None = None) -> None:
        self.heatmap = heatmap or {}
        self.max_heatmap_value = max(self.heatmap.values()) if self.heatmap else 0
        gs = global_settings or {}
        self.accent_threshold = float(gs.get("accent_threshold", 0.6))
        gd_range = gs.get("ghost_density_range", (0.3, 0.8))
        if isinstance(gd_range, (list, tuple)) and len(gd_range) >= 2:
            self.ghost_density_min = float(gd_range[0])
            self.ghost_density_max = float(gd_range[1])
        else:
            self.ghost_density_min = 0.3
            self.ghost_density_max = 0.8
        self.rng = rng or random.Random()
        self._rw_state = 0.0

    def begin_bar(self) -> None:
        """Advance the random walk for a new bar."""
        self._random_walk()

    def _random_walk(self) -> float:
        step = self.rng.uniform(-0.05, 0.05)
        self._rw_state = max(0.0, min(1.0, self._rw_state + step))
        return self._rw_state

    def accent(self, rel_beat: int, velocity: int) -> int:
        """Return velocity with heatmap accent applied."""
        heat = self.heatmap.get(rel_beat, 0)
        if heat < self.accent_threshold * self.max_heatmap_value:
            return velocity
        factor = 1.0 + 0.2 * self._random_walk()
        vel = int(round(velocity * factor))
        return max(1, min(127, vel))

    def maybe_ghost_hat(self, rel_beat: int) -> bool:
        """Return ``True`` if a ghost hat should be inserted at ``rel_beat``."""
        rnd = self.rng.random()
        target = self.rng.uniform(self.ghost_density_min, self.ghost_density_max)
        return rnd < target
