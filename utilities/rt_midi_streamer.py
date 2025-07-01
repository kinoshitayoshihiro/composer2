from __future__ import annotations

import asyncio
from typing import Optional

from music21 import stream

try:
    import rtmidi
except Exception:  # pragma: no cover - optional
    rtmidi = None

from .tempo_utils import TempoMap, beat_to_seconds


class RtMidiStreamer:
    def __init__(self, port_name: str | None = None, *, bpm: float = 120.0) -> None:
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

    @staticmethod
    def list_ports() -> list[str]:
        if rtmidi is None:
            return []
        return rtmidi.MidiOut().get_ports()

    async def _play_note(self, start: float, end: float, pitch: int, velocity: int) -> None:
        loop = asyncio.get_running_loop()
        await asyncio.sleep(max(0.0, start - loop.time()))
        self._midi.send_message([0x90, pitch, velocity])
        await asyncio.sleep(max(0.0, end - loop.time()))
        self._midi.send_message([0x80, pitch, 0])

    async def play_stream(self, part: stream.Part) -> None:
        loop = asyncio.get_running_loop()
        start_time = loop.time()
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
