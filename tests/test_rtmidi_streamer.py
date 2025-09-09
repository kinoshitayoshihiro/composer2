import time
import types
from types import SimpleNamespace

from realtime import rtmidi_streamer

class DummyMidiOut:
    def __init__(self) -> None:
        self.events: list[tuple[float, list[int]]] = []

    def get_ports(self):
        return ["dummy"]

    def open_port(self, idx: int) -> None:
        pass

    def close_port(self) -> None:
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


def test_rtmidi_start_stop(monkeypatch):
    midi = DummyMidiOut()
    rtmidi_stub = types.ModuleType("rtmidi")
    rtmidi_stub.MidiOut = lambda: midi
    monkeypatch.setattr(rtmidi_streamer, "rtmidi", rtmidi_stub)
    monkeypatch.setattr(rtmidi_streamer.RtMidiStreamer, "_run_loop", lambda self: None)
    gen = DummyGen()
    streamer = rtmidi_streamer.RtMidiStreamer("dummy", gen)
    called: list[int] = []
    streamer.start(bpm=240.0, buffer_bars=1, callback=lambda b: called.append(b))
    streamer.on_tick()
    streamer.stop()
    assert called == [1]
    assert any(msg[0] == 0x90 for _, msg in midi.events)
