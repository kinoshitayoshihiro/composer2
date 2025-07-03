from __future__ import annotations

from collections import deque
import threading
from typing import Any

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional
    sd = None  # type: ignore

try:
    import pyloudnorm as pyln  # type: ignore
except Exception:  # pragma: no cover - optional
    pyln = None  # type: ignore


class RealtimeLoudnessMeter:
    """Realtime loudness meter based on LUFS."""

    def __init__(self, sample_rate: int = 44100, window_sec: float = 1.0) -> None:
        self.sample_rate = sample_rate
        self.window_size = int(sample_rate * window_sec)
        self._meter = pyln.Meter(sample_rate) if pyln else None
        self._buf: deque[float] = deque(maxlen=self.window_size)
        self._lock = threading.Lock()
        self._lufs: float = 0.0
        self._stream: Any | None = None

    def _compute_lufs(self, samples: np.ndarray) -> float:
        if self._meter:
            return float(self._meter.integrated_loudness(samples))
        rms = np.sqrt(np.mean(samples ** 2)) or 1e-9
        return 20 * np.log10(rms)

    def _callback(self, indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        if status:
            return
        with self._lock:
            self._buf.extend(indata[:, 0])
            if len(self._buf) >= self.window_size:
                data = np.array(self._buf, dtype=np.float32)
                self._buf.clear()
                self._lufs = self._compute_lufs(data)

    def start(self, device: str | int | None = None) -> None:
        if sd is None:
            raise RuntimeError("sounddevice not available")
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            callback=self._callback,
            device=device,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_current_lufs(self) -> float:
        with self._lock:
            return float(self._lufs)

__all__ = ["RealtimeLoudnessMeter"]
