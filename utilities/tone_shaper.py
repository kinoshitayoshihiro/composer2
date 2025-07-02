from __future__ import annotations


class ToneShaper:
    """Select amp/cabinet presets and emit CC events."""

    def __init__(self, presets: dict[str, int] | None = None) -> None:
        self.presets = {"clean": 0, "drive": 32, "svt": 64, "fuzz": 96}
        if presets:
            self.presets.update(presets)

    def choose_preset(self, avg_velocity: float, intensity: str) -> str:
        """Return preset name for given velocity and intensity."""
        if intensity in {"high", "very_high"} or avg_velocity >= 85:
            return "drive"
        if intensity in {"medium_high", "medium"}:
            return "svt"
        if intensity in {"medium_low", "low"} or avg_velocity < 60:
            return "clean"
        return "fuzz"

    def to_cc_events(self, preset_name: str, offset: float) -> list[dict]:
        value = self.presets.get(preset_name, self.presets["clean"])
        return [{"time": float(offset), "number": 31, "value": value}]

__all__ = ["ToneShaper"]
