from __future__ import annotations

import asyncio
import logging
import statistics
from collections.abc import Callable
import random

from music21 import stream

try:
    import rtmidi
except Exception:  # pragma: no cover - optional
    rtmidi = None

from .live_buffer import LiveBuffer, apply_late_humanization
from .tempo_utils import TempoMap, beat_to_seconds


class RtMidiStreamer:
    def __init__(
        self,
        port_name: str | None = None,
        *,
        bpm: float = 120.0,
        buffer_ms: float = 5.0,
        measure_latency: bool = False,
    ) -> None:
        if rtmidi is None:
            raise RuntimeError("python-rtmidi required")
        self._midi = rtmidi.MidiOut()
        ports = self._midi.get_ports()
        if port_name is None:
            if not ports:
                raise RuntimeError("No MIDI output ports")
            self.port_name = ports[0]
        else:
            if port_name not in ports:
                raise RuntimeError(f"Port '{port_name}' not found")
            self.port_name = port_name
        self._midi.open_port(ports.index(self.port_name))
        self.tempo = TempoMap([{"beat": 0.0, "bpm": bpm}])
        self.buffer = max(0.0, float(buffer_ms)) / 1000.0
        self.measure_latency = measure_latency
        self.latencies: list[float] = []
        self.logger = logging.getLogger(__name__)

    def _reconnect(self) -> None:
        try:
            self._midi.close_port()
        except Exception:
            pass
        self._midi = rtmidi.MidiOut()
        ports = self._midi.get_ports()
        if self.port_name not in ports:
            raise RuntimeError(f"Port '{self.port_name}' not found")
        self._midi.open_port(ports.index(self.port_name))

    def _send_with_retry(self, msg: list[int]) -> None:
        try:
            self._midi.send_message(msg)
        except Exception as exc:  # pragma: no cover - runtime safety
            self.logger.error("MIDI send failed: %s", exc)
            try:
                self._reconnect()
                self._midi.send_message(msg)
            except Exception as exc2:  # pragma: no cover - runtime safety
                self.logger.error("MIDI resend failed: %s", exc2)

    @staticmethod
    def list_ports() -> list[str]:
        if rtmidi is None:
            return []
        return rtmidi.MidiOut().get_ports()

    async def _play_note(self, start: float, end: float, pitch: int, velocity: int) -> None:
        loop = asyncio.get_running_loop()
        await asyncio.sleep(max(0.0, start - loop.time() - self.buffer))
        self._send_with_retry([0x90, pitch, velocity])
        sent = loop.time()
        if self.measure_latency:
            self.latencies.append(sent - start)
        await asyncio.sleep(max(0.0, end - loop.time()))
        self._send_with_retry([0x80, pitch, 0])

    async def play_stream(self, part: stream.Part, *, late_humanize: bool = False) -> None:
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        if late_humanize:
            apply_late_humanization(part.flatten().notes, rng=random.Random(), bpm=self.tempo.events[0]["bpm"] if self.tempo.events else 120.0)
        tasks = []
        for n in part.flatten().notes:
            start = start_time + beat_to_seconds(float(n.offset), self.tempo.events)
            end = start_time + beat_to_seconds(
                float(n.offset + n.quarterLength), self.tempo.events
            )
            pitch = int(n.pitch.midi)
            vel = int(max(0, min(127, n.volume.velocity or 64)))
            tasks.append(loop.create_task(self._play_note(start, end, pitch, vel)))
        if tasks:
            await asyncio.gather(*tasks)
        if self.measure_latency and self.latencies:
            mean = statistics.mean(self.latencies)
            std = statistics.pstdev(self.latencies)
            self.logger.info(
                "Latency mean %.3f ms, jitter %.3f ms",
                mean * 1000.0,
                std * 1000.0,
            )

    async def play_live(
        self,
        generator: Callable[[int], stream.Part],
        *,
        buffer_ahead: int = 4,
        parallel_bars: int = 1,
        late_humanize: bool = False,
    ) -> None:
        """Play parts sequentially using a background generation buffer."""
        buf = LiveBuffer(generator, buffer_ahead=buffer_ahead, parallel_bars=parallel_bars)
        idx = 0
        try:
            while True:
                part = buf.get_next()
                if part is None:
                    break
                await self.play_stream(part, late_humanize=late_humanize)
                idx += 1
        finally:
            buf.shutdown()

    def latency_stats(self) -> dict[str, float] | None:
        if not self.latencies:
            return None
        return {
            "mean_ms": statistics.mean(self.latencies) * 1000.0,
            "stdev_ms": statistics.pstdev(self.latencies) * 1000.0,
            "max_ms": max(self.latencies) * 1000.0,
        }

    # --------------------------------------------------------------
    # Low level CC helpers
    # --------------------------------------------------------------
    def send_cc(self, cc: int, value: int, time: float) -> None:
        """Send Control Change message at absolute ``time`` seconds."""
        loop = asyncio.get_event_loop()

        async def _task() -> None:
            await asyncio.sleep(max(0.0, time - loop.time() - self.buffer))
            self._send_with_retry([0xB0, int(cc), int(value)])

        loop.create_task(_task())


def stream_cc_events(part: stream.Part, streamer: RtMidiStreamer) -> None:
    """Send ``part.extra_cc`` events using ``streamer``."""
    now = asyncio.get_event_loop().time()
    for ev in getattr(part, "extra_cc", []):
        t = now + beat_to_seconds(float(ev.get("time", 0.0)), streamer.tempo.events)
        cc = int(ev.get("cc", 0))
        val = int(ev.get("val", 0))
        streamer.send_cc(cc, val, t)

