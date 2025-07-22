import asyncio
import time
from types import SimpleNamespace

from music21 import note, stream, volume

# Import the actual module to ensure monkeypatching affects the implementation
from utilities import rtmidi_streamer as rt_midi_streamer


class DummyMidiOut:
    def __init__(self) -> None:
        self.events: list[tuple[float, list[int]]] = []

    def get_ports(self):
        return ["dummy"]

    def open_port(self, idx: int) -> None:  # noqa: D401
        pass

    def send_message(self, msg):
        self.events.append((time.perf_counter(), msg))


def make_part():
    p = stream.Part()
    n1 = note.Note(60, quarterLength=0.25)
    n1.volume = volume.Volume(velocity=80)
    p.insert(0.0, n1)
    n2 = note.Note(62, quarterLength=0.25)
    n2.volume = volume.Volume(velocity=70)
    p.insert(0.5, n2)
    return p


def test_rt_streamer_dummy(monkeypatch):
    midi = DummyMidiOut()
    monkeypatch.setattr(rt_midi_streamer, "rtmidi", SimpleNamespace(MidiOut=lambda: midi))
    part = make_part()
    streamer = rt_midi_streamer.RtMidiStreamer(
        "dummy",
        bpm=120.0,
        buffer_ms=0.0,
        measure_latency=False,
    )
    asyncio.run(streamer.play_stream(part))
    on_times = [t for t, msg in midi.events if msg[0] == 0x90]
    assert len(on_times) == 2
    diff = on_times[1] - on_times[0]
    assert abs(diff - 0.25) <= 0.02
