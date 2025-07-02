from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class LiveBuffer:
    """Simple ring buffer that pre-generates bars ahead of playback."""

    def __init__(
        self,
        generator: Callable[[int], Any],
        *,
        buffer_ahead: int = 4,
        parallel_bars: int = 1,
        logger: logging.Logger | None = None,
        warn_level: int = logging.WARNING,
    ) -> None:
        self.generator = generator
        self.buffer_ahead = max(1, int(buffer_ahead))
        self.logger = logger or logging.getLogger(__name__)
        self.warn_level = warn_level
        self.executor = ThreadPoolExecutor(max_workers=max(1, int(parallel_bars)))
        self.buffer: deque[Any] = deque()
        self.next_index = 0
        self._fill()

    def _fill(self) -> None:
        while len(self.buffer) < self.buffer_ahead:
            idx = self.next_index
            self.next_index += 1
            fut = self.executor.submit(self.generator, idx)
            self.buffer.append(fut)

    def get_next(self) -> Any:
        if not self.buffer:
            self.logger.log(self.warn_level, "LiveBuffer underrun; regenerating")
            self._fill()
        fut = self.buffer.popleft()
        try:
            result = fut.result()
        except Exception as exc:  # pragma: no cover - runtime safety
            self.logger.log(self.warn_level, "Generation failed: %s", exc)
            result = None
        self._fill()
        return result

    def shutdown(self) -> None:
        self.executor.shutdown()
