from __future__ import annotations
from pathlib import Path
from typing import Any

from utilities import perc_sampler_v1


class PercGenerator:
    """Simple percussion generator based on n-gram sampling."""

    def __init__(self, model_path: str | Path = "models/perc_ngram.pkl") -> None:
        self.model_path = Path(model_path)
        self.model: perc_sampler_v1.PercModel | None = None
        if self.model_path.exists():
            self.model = perc_sampler_v1.load(self.model_path)
        self.history: list[str] = []

    def reset(self) -> None:
        self.history.clear()

    def generate_bar(self, *, temperature: float = 1.0) -> list[dict[str, Any]]:
        if self.model is None:
            return []
        return perc_sampler_v1.generate_bar(self.history, model=self.model, temperature=temperature)
