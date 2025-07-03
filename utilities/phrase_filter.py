from __future__ import annotations

from typing import Iterable, Sequence


class NGramDiversityFilter:
    """Detect overly similar phrase sequences using n-gram overlap."""

    def __init__(self, n: int = 8, max_sim: float = 0.8) -> None:
        self.n = max(1, int(n))
        self.max_sim = float(max_sim)
        self._prev: set[tuple[str, ...]] | None = None

    def _ngrams(self, items: Sequence[str]) -> set[tuple[str, ...]]:
        return {
            tuple(items[i : i + self.n])
            for i in range(len(items) - self.n + 1)
        }

    def too_similar(self, events: Iterable[dict]) -> bool:
        labels = [str(ev.get("instrument", "")) for ev in events]
        grams = self._ngrams(labels)
        if self._prev is None:
            self._prev = grams
            return False
        if not grams:
            self._prev = grams
            return False
        sim = len(grams & self._prev) / max(len(grams), 1)
        self._prev = grams
        return sim >= self.max_sim


__all__ = ["NGramDiversityFilter"]
