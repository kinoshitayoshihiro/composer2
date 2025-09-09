from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List


@dataclass
class ControlSplineModel:
    """Placeholder spline-based control model."""

    control_points: List[Any] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"control_points": self.control_points}, fh)

    @classmethod
    def load(cls, path: str | Path) -> "ControlSplineModel":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(control_points=data.get("control_points", []))
