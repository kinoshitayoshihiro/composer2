from __future__ import annotations

import logging
import statistics
import time
import threading
from collections.abc import Callable
from typing import Any, List

try:
    import rtmidi
except Exception:  # pragma: no cover - optional
    rtmidi = None  # type: ignore

try:
    from generator.piano_ml_generator import PianoMLGenerator
except Exception:  # pragma: no cover - optional
    PianoMLGenerator = None  # type: ignore

from utilities.live_buffer import LiveBuffer
from utilities.tempo_utils import beat_to_seconds


class RtMidiStreamer:
    """Stream bars from :class:`PianoMLGenerator` via ``python-rtmidi``."""

    def __init__(self, port_name: str, generator: PianoMLGenerator) -> None:
        if rtmidi is None:
            raise RuntimeError("python-rtmidi required")
        self.port_name = port_name
        self.generator = generator
        self.logger = logging.getLogger(__name__)
        self.bpm = 120.0
        self._midi: Any | None = None
        self._buf: LiveBuffer | None = None
        self._context: List[dict[str, Any]] = []
        self._latencies: List[float] = []
        self._last_log = time.perf_counter()
        self._running = False
        self._thread: threading.Thread | None = None

    @staticmethod
    def list_ports() -> list[str]:
        if rtmidi is None:
            return []
        return rtmidi.MidiOut().get_ports()

    def start(
        self,
        bpm: float,
        buffer_bars: int,
        callback: Callable[[int], None] | None = None,
    ) -> None:
        """Open MIDI port and begin background streaming."""
        if self._running:
            return
        self.bpm = float(bpm)
        self._callback = callback
        self._buf = LiveBuffer(self._next_bar, buffer_ahead=max(1, int(buffer_bars)))
        self._bar = 0
        self._latencies.clear()
        self._last_log = time.perf_counter()
        self._midi = rtmidi.MidiOut()
        ports = self._midi.get_ports()
        if self.port_name not in ports:
            raise RuntimeError(f"Port '{self.port_name}' not found")
        self._midi.open_port(ports.index(self.port_name))
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop streaming and close the MIDI port."""
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._buf is not None:
            self._buf.shutdown()
            self._buf = None
        if self._midi is not None:
            try:
                self._midi.close_port()
            except Exception:
                pass
            self._midi = None
        if self._latencies:
            std_ms = statistics.pstdev(self._latencies) * 1000.0
            max_ms = max(self._latencies) * 1000.0
            self.logger.info("jitter stddev %.1f ms max %.1f ms", std_ms, max_ms)
            self._latencies.clear()

    def _next_bar(self, _idx: int) -> list[dict[str, Any]]:
        events = self.generator.step(self._context)
        self._context.extend(events)
        return events

    def on_tick(self) -> None:
        if self._buf is None:
            return
        bar = self._buf.get_next()
        if not bar:
            return
        beat_sec = 60.0 / self.bpm
        start = time.perf_counter()
        for ev in bar:
            offset = float(ev.get("offset", 0.0))
            dur = float(ev.get("duration", 0.25))
            pitch = int(ev.get("pitch", 60))
            vel = int(ev.get("velocity", 100))
            note_on = start + beat_to_seconds(offset, [{"beat": 0.0, "bpm": self.bpm}])
            time.sleep(max(0.0, note_on - time.perf_counter()))
            self._midi.send_message([0x90, pitch, vel])
            note_off = note_on + beat_sec * dur
            time.sleep(max(0.0, note_off - time.perf_counter()))
            self._midi.send_message([0x80, pitch, 0])
        elapsed = time.perf_counter() - start
        expected = beat_sec * 4.0
        self._latencies.append(elapsed - expected)
        self._bar += 1
        if self._callback:
            self._callback(self._bar)
        now = time.perf_counter()
        if now - self._last_log >= 30.0 and self._latencies:
            std_ms = statistics.pstdev(self._latencies) * 1000.0
            self.logger.info("jitter %.1f ms", std_ms)
            self._latencies.clear()
            self._last_log = now

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------
    def _run_loop(self) -> None:
        beat_sec = 60.0 / self.bpm
        bar_len = beat_sec * 4.0
        while self._running:
            start = time.perf_counter()
            self.on_tick()
            delay = bar_len - (time.perf_counter() - start)
            time.sleep(max(0.0, delay))
