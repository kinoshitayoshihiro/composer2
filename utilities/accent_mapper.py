from __future__ import annotations

import random
from typing import Dict

from .velocity_smoother import EMASmoother


class AccentMapper:
    """Map accent intensity and ghost-hat density using a vocal heatmap."""

    def __init__(
        self,
        heatmap: Dict[int, int] | None,
        global_settings: Dict[str, object] | None,
        *,
        rng: random.Random | None = None,
        ema_smoother: EMASmoother | None = None,
        use_velocity_ema: bool = False,
        walk_after_ema: bool | None = None,
    ) -> None:
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
        if "walk_after_ema" in gs and walk_after_ema is not None and bool(gs["walk_after_ema"]) != bool(walk_after_ema):
            raise ValueError("walk_after_ema specified twice with different values.")
        self.walk_after_ema = bool(walk_after_ema if walk_after_ema is not None else gs.get("walk_after_ema", False))
        self.use_velocity_ema = use_velocity_ema
        self.ema_smoother = ema_smoother or EMASmoother(window=16)
        self.debug_rw_values: list[tuple[float, int]] = []

    def set_step_range(self, step: int) -> None:
        """Set the random walk step range."""
        self._step_range = int(step)

    def begin_bar(self, bar_start_offset_ql: float = 0.0, step_range: int | None = None) -> None:
        """Advance the velocity random walk by one step.

        Parameters
        ----------
        bar_start_offset_ql : float, optional
            Absolute start offset of the bar in quarter lengths.
        step_range : int | None, optional
            If provided, update the random walk step range before advancing.
        """
        if step_range is not None:
            self._step_range = int(step_range)
        step = self.rng.randint(-self._step_range, self._step_range)
        self._rw_state = int(
            max(-self._step_range, min(self._step_range, self._rw_state + step))
        )
        self.debug_rw_values.append((bar_start_offset_ql, self._rw_state))


    def accent(self, rel_beat: int, velocity: int, *, apply_walk: bool = True) -> int:
        """Return velocity with optional accent boost and random walk applied."""
        vel = int(velocity)
        heat = self.heatmap.get(rel_beat, 0)
        if heat >= self.accent_threshold * self.max_heatmap_value:
            vel = int(round(vel * 1.2))
        if self.use_velocity_ema:
            if self.walk_after_ema:
                vel = self.ema_smoother.update(vel)
                if apply_walk:
                    vel += self._rw_state
            else:
                if apply_walk:
                    vel += self._rw_state
                vel = self.ema_smoother.update(vel)
        elif apply_walk:
            vel += self._rw_state
        return max(1, min(127, vel))

    def maybe_ghost_hat(self, rel_beat: int) -> bool:
        """Return ``True`` if a ghost hat should be inserted at ``rel_beat``."""
        rnd = self.rng.random()
        target = self.rng.uniform(self.ghost_density_min, self.ghost_density_max)
        return rnd < target

    @staticmethod
    def map_layer(layer: str | int, *, rng: random.Random | None = None) -> int:
        """Return a MIDI velocity for ``layer``.

        Parameters
        ----------
        layer : str | int
            Either ``"low"``, ``"mid"``, ``"high"`` or a numeric velocity.
        rng : random.Random | None
            Optional RNG for the random range selection.
        """
        r = rng or random.Random()
        if isinstance(layer, int):
            return max(1, min(127, layer))
        s = str(layer).strip().lower()
        if s == "low":
            return r.randint(40, 55)
        if s in {"mid", "middle"}:
            return r.randint(70, 85)
        if s == "high":
            return r.randint(100, 110)
        try:
            return max(1, min(127, int(float(s))))
        except Exception:
            return r.randint(70, 85)
