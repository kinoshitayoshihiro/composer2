from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from utilities import groove_sampler_v2


class PercGenerator:
    """Simple percussion generator using :mod:`groove_sampler_v2`."""

    def __init__(self, model: Path | str, cond: Dict[str, str] | None = None) -> None:
        self.model_path = Path(model)
        self.model = groove_sampler_v2.load(self.model_path)
        self.cond: Dict[str, str] = cond or {}

    def sample(self, bars: int = 4, **kwargs: Any) -> list[dict[str, float | str]]:
        return groove_sampler_v2.generate_events(
            self.model,
            bars=bars,
            cond=self.cond,
            **kwargs,
        )
