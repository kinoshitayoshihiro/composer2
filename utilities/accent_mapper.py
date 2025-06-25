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
        self._rw_state: int = 0
        self._step_range = int(gs.get("random_walk_step", 8))

    def begin_bar(self) -> None:
        """Advance the velocity random walk by one step."""
        step = self.rng.randint(-self._step_range, self._step_range)
        self._rw_state = int(
            max(-self._step_range, min(self._step_range, self._rw_state + step))
        )


    def accent(self, rel_beat: int, velocity: int, *, apply_walk: bool = False) -> int:
        """Return velocity with optional accent boost and random walk applied."""
        vel = int(velocity)
        heat = self.heatmap.get(rel_beat, 0)
        if heat >= self.accent_threshold * self.max_heatmap_value:
            vel = int(round(vel * 1.2))
        if apply_walk:
            vel += self._rw_state
        return max(1, min(127, vel))

    def maybe_ghost_hat(self, rel_beat: int) -> bool:
        """Return ``True`` if a ghost hat should be inserted at ``rel_beat``."""
        rnd = self.rng.random()
        target = self.rng.uniform(self.ghost_density_min, self.ghost_density_max)
        return rnd < target
