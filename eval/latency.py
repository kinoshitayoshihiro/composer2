from __future__ import annotations

import random
from collections.abc import Callable

from utilities.groove_sampler_ngram import Event
from utilities.streaming_sampler import BaseSampler, RealtimePlayer


class _ConstSampler(BaseSampler):
    def __init__(self) -> None:
        self.step = 0

    def feed_history(self, events: list[tuple[int, str]]) -> None:
        return

    def next_step(
        self, *, cond: dict[str, float | int] | None, rng: random.Random
    ) -> Event:
        ev: Event = {
            "instrument": "kick",
            "offset": self.step / 4,
            "velocity": 100,
            "duration": 0.25,
        }
        self.step = (self.step + 1) % 16
        return ev


def measure_latency(
    player: RealtimePlayer,
    *,
    bars: int = 1,
) -> float:
    """Return average lead-time in milliseconds."""

    deltas: list[float] = []
    orig_sink: Callable[[Event], None] = player.sink

    def _schedule_time(ev: Event) -> float:
        bar = int(ev["offset"] // 4)
        beat = ev["offset"] % 4
        return start + bar * 4 * beat_sec + beat * beat_sec

    def _sink(ev: Event) -> None:
        send = player.clock()
        delta = (send - _schedule_time(ev)) * 1000.0
        deltas.append(delta)
        orig_sink(ev)

    beat_sec = 60.0 / player.bpm
    start = player.clock()
    player.sink = _sink
    player.play(bars=bars)
    player.sink = orig_sink
    return float(sum(deltas) / len(deltas)) if deltas else 0.0


def evaluate_model(model_path: str, backend: str = "ngram") -> dict[str, float]:
    """Measure latency for ``model_path`` using a dummy sampler."""

    sampler = _ConstSampler()
    clock_val = 0.0

    def _clock() -> float:
        return clock_val

    def _sleep(d: float) -> None:
        nonlocal clock_val
        clock_val += d

    player = RealtimePlayer(sampler, bpm=120.0, clock=_clock, sleep=_sleep)
    avg = measure_latency(player, bars=1)
    return {"avg_ms": avg}
