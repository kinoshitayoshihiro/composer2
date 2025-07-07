import time
from types import SimpleNamespace

import pytest

from realtime import rtmidi_streamer


class DummyMidiOut:
    def __init__(self) -> None:
        self.events: list[tuple[float, list[int]]] = []

    def get_ports(self):
        return ["dummy"]

    def open_port(self, idx: int) -> None:
        pass

    def send_message(self, msg):
        self.events.append((time.perf_counter(), msg))


class DummyGen:
    def __init__(self, *_a, **_k):
        self.calls = 0

    def step(self, _ctx):
        self.calls += 1
        return [
            {"pitch": 60, "velocity": 80, "offset": 0.0, "duration": 0.25},
            {"pitch": 62, "velocity": 70, "offset": 0.5, "duration": 0.25},
        ]


def test_rtmidi_stream(monkeypatch):
    midi = DummyMidiOut()
    monkeypatch.setattr(rtmidi_streamer, "rtmidi", SimpleNamespace(MidiOut=lambda: midi))
    monkeypatch.setattr(rtmidi_streamer, "PianoMLGenerator", lambda *_a, **_k: DummyGen())
    gen = DummyGen()
    streamer = rtmidi_streamer.RtMidiStreamer("dummy", gen)
    streamer.start(bpm=240.0, buffer_bars=1, callback=None)
    streamer.on_tick()
    streamer.stop()
    on_times = [t for t, msg in midi.events if msg[0] == 0x90]
    assert len(on_times) == 2
    diff = on_times[1] - on_times[0]
    assert abs(diff - 0.125) <= 0.02
