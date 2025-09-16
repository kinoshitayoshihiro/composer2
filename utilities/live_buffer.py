from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable
import random
import time
import os

# Set SPARKLE_DETERMINISTIC=1 to force deterministic RNG defaults for tests.
_SPARKLE_DETERMINISTIC = os.getenv("SPARKLE_DETERMINISTIC") == "1"


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
        start = time.monotonic()
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
        elapsed = time.monotonic() - start
        jitter = 0.03 + (0.005 if self.warn_level < logging.DEBUG else 0.0)
        if elapsed > jitter:
            self.logger.log(
                self.warn_level,
                "LiveBuffer jitter %.1f ms", elapsed * 1000.0
            )
        return result

    def shutdown(self) -> None:
        self.executor.shutdown()


def apply_late_humanization(
    notes: Iterable[Any], jitter_ms: tuple[int, int] = (5, 10), *, bpm: float = 120.0, rng: random.Random | None = None
) -> None:
    """Jitter note offsets right before playback."""

    if rng is None:
        rng = random.Random(0) if _SPARKLE_DETERMINISTIC else random.Random()
    lower, upper = jitter_ms
    scale = bpm / 60000.0
    for n in notes:
        jitter = rng.uniform(-abs(lower), abs(upper)) * scale
        if isinstance(n, dict):
            n["offset"] = float(n.get("offset", 0.0)) + jitter
        else:
            try:
                n.offset += jitter
            except Exception:  # pragma: no cover - protective
                continue
