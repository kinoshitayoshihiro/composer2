from __future__ import annotations

import time
from typing import Dict, Tuple

from music21 import note, stream

try:
    import rtmidi
except Exception:  # pragma: no cover - optional
    rtmidi = None  # type: ignore


class MIDIRecorder:
    """Simple real-time MIDI input recorder."""

    def __init__(self, port_name: str | None = None, *, bpm: float = 120.0) -> None:
        if rtmidi is None:
            raise RuntimeError("python-rtmidi required")
        self._midi = rtmidi.MidiIn()
        ports = self._midi.get_ports()
        if port_name is None:
            if not ports:
                raise RuntimeError("No MIDI input ports")
            self.port_name = ports[0]
        else:
            if port_name not in ports:
                raise RuntimeError(f"Port '{port_name}' not found")
            self.port_name = port_name
        self._midi.open_port(ports.index(self.port_name))
        self._midi.ignore_types(sysex=False, timing=False, sensing=False)
        self.bpm = bpm
        self._events: list[Tuple[float, list[int]]] = []
        self._recording = False
        self._last_time = 0.0
        self._start_time = 0.0

    def _callback(self, event: Tuple[list[int], float], _data: object | None = None) -> None:
        if not self._recording:
            return
        msg, delta = event
        self._last_time += float(delta)
        timestamp = self._last_time
        self._events.append((timestamp, list(msg)))

    def start_recording(self) -> None:
        """Begin capturing MIDI events."""
        self._events.clear()
        self._recording = True
        self._last_time = 0.0
        self._start_time = time.time()
        self._midi.set_callback(self._callback)

    def stop_recording(self) -> stream.Part:
        """Stop capture and return a :class:`music21.stream.Part`."""
        self._recording = False
        self._midi.cancel_callback()
        events = list(self._events)
        self._events.clear()
        part = stream.Part()
        beat = 60.0 / self.bpm
        active: Dict[int, Tuple[float, int]] = {}
        for ts, msg in events:
            status = msg[0] & 0xF0
            if status == 0x90 and msg[2] > 0:
                active[msg[1]] = (ts, msg[2])
            elif status in (0x80, 0x90) and msg[2] == 0 or status == 0x80:
                if msg[1] in active:
                    start, vel = active.pop(msg[1])
                    dur = ts - start
                    n = note.Note(midi=msg[1], quarterLength=dur / beat)
                    n.volume.velocity = vel
                    part.insert(start / beat, n)
        # close remaining notes
        for pitch, (start, vel) in active.items():
            dur = self._last_time - start
            n = note.Note(midi=pitch, quarterLength=dur / beat)
            n.volume.velocity = vel
            part.insert(start / beat, n)
        return part
