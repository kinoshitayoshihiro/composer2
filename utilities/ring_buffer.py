from __future__ import annotations

from typing import Any, Tuple


class RingBuffer:
    """Lock-free ring buffer storing timestamped event lists."""

    def __init__(self, size: int = 256) -> None:
        self.size = max(1, int(size))
        self._times: list[int] = [0] * self.size
        self._events: list[list[Any] | None] = [None] * self.size
        self._start = 0
        self._end = 0
        self._full = False

    def push(self, items: list[Tuple[int, list[Any]]]) -> None:
        """Append ``items`` to the buffer.

        Each item is ``(timestamp, events)``.
        """
        for ts, ev in items:
            self._times[self._end] = int(ts)
            self._events[self._end] = list(ev)
            if self._full:
                self._start = (self._start + 1) % self.size
            self._end = (self._end + 1) % self.size
            self._full = self._end == self._start

    def pop_until(self, tick: int) -> list[Any]:
        """Pop all events with timestamp ``<= tick``."""
        out: list[Any] = []
        while (self._start != self._end) or self._full:
            ts = self._times[self._start]
            if ts > tick:
                break
            ev = self._events[self._start]
            if ev:
                out.extend(ev)
            self._start = (self._start + 1) % self.size
            self._full = False
        return out


__all__ = ["RingBuffer"]
