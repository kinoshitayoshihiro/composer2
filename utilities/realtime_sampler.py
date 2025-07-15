from __future__ import annotations

import threading
from typing import Any, Iterable

from .ring_buffer import RingBuffer


class RealtimeSampler:
    """Pull small audio chunks and push note events into a ring buffer."""

    def __init__(
        self,
        drum_gen: Any,
        perc_gen: Any,
        buffer: RingBuffer,
        *,
        sample_rate: int = 44100,
        chunk_size: int = 128,
    ) -> None:
        self.drum_gen = drum_gen
        self.perc_gen = perc_gen
        self.buffer = buffer
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._tick = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            events = self._next_events()
            self.buffer.push([(self._tick, events)])
            self._tick += self.chunk_size

    def _next_events(self) -> list[dict[str, Any]]:
        drum = self.drum_gen.generate_bar() if hasattr(self.drum_gen, "generate_bar") else []
        perc = self.perc_gen.generate_bar() if hasattr(self.perc_gen, "generate_bar") else []
        return [*drum, *perc]


__all__ = ["RealtimeSampler"]
