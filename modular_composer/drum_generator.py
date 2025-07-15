from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from utilities import groove_sampler_v2
from utilities import transformer_sampler


class DrumGenerator:
    """Simplified drum generator supporting n-gram or transformer backend."""

    def __init__(
        self,
        model: Path | str,
        cond: Dict[str, str] | None = None,
        *,
        backend: str = "ngram",
    ) -> None:
        self.model_path = Path(model)
        self.backend = backend
        if backend == "transformer":
            self.model = transformer_sampler.load(self.model_path)
        else:
            self.model = groove_sampler_v2.load(self.model_path)
        self.cond: Dict[str, str] = cond or {}

    def sample(self, bars: int = 4, **kwargs: Any) -> list[dict[str, float | str]]:
        """Generate drum events."""
        if self.backend == "transformer":
            length = bars * 16
            temperature = {"drums": float(kwargs.get("temperature", 1.0))}
            return transformer_sampler.sample_multi(self.model, {}, length, temperature)["drums"]
        return groove_sampler_v2.generate_events(
            self.model,
            bars=bars,
            cond=self.cond,
            **kwargs,
        )
