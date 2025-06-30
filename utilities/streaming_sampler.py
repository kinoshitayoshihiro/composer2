"""Real-time streaming sampler utilities."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

try:
    import sounddevice as _sd
except Exception:  # pragma: no cover - optional
    _sd = None

sd: Any | None = _sd

def _null_sink(_: Event) -> None:
    """Dummy sink when no audio backend is available."""
    return

from .groove_sampler_ngram import RESOLUTION, Event, State


class BaseSampler(Protocol):
    def feed_history(self, events: list[State]) -> None:
        ...

    def next_step(self, *, cond: dict | None, rng: random.Random) -> Event:
        ...


def _default_sink(ev: Event) -> None:
    print(f"{ev['instrument']} at {ev['offset']:.3f}s")


@dataclass
class RealtimePlayer:
    sampler: BaseSampler
    bpm: float
    sink: callable = _default_sink
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    buffer_bars: int = 1

    def play(self, bars: int = 4) -> None:
        if sd is None and self.sink is _default_sink:
            self.sink = _null_sink
        sec_per_step = 60.0 / self.bpm / (RESOLUTION / 4)
        start = self.clock()
        rng = random.Random(0)
        history: list[State] = []
        for bar in range(bars):
            for _ in range(RESOLUTION):
                ev = self.sampler.next_step(cond=None, rng=rng)
                ts = start + bar * RESOLUTION * sec_per_step + ev["offset"] * 60.0 / self.bpm
                delay = ts - self.clock()
                if delay > 0:
                    self.sleep(delay)
                self.sink(ev)
                step = int(round(ev["offset"] * (RESOLUTION / 4)))
                history.append((step, ev["instrument"]))
            if (bar + 1) % self.buffer_bars == 0:
                self.sampler.feed_history(history)
                history.clear()
        if history:
            self.sampler.feed_history(history)
        return

