import asyncio
import logging
import time
from types import SimpleNamespace

import pytest
from music21 import note, stream, volume

# Import the implementation module to patch the correct attributes
from utilities import rtmidi_streamer as rt_midi_streamer
from utilities.live_buffer import LiveBuffer


def slow_gen(idx: int) -> int:
    time.sleep(0.05)
    return idx


@pytest.mark.slow
def test_live_buffer_integration(caplog):
    buf = LiveBuffer(slow_gen, buffer_ahead=2, parallel_bars=1, warn_level=logging.CRITICAL)
    caplog.set_level(logging.CRITICAL)
    results = []
    for _ in range(5):
        results.append(buf.get_next())
        time.sleep(0.1)
    buf.shutdown()
    assert results == list(range(5))
    assert not any("underrun" in r.message for r in caplog.records)


class DummyMidiOut:
    def __init__(self) -> None:
        self.events: list[tuple[float, list[int]]] = []

    def get_ports(self):
        return ["dummy"]

    def open_port(self, idx: int) -> None:
        pass

    def send_message(self, msg):
        self.events.append((time.perf_counter(), msg))


def _make_part() -> stream.Part:
    p = stream.Part()
    n1 = note.Note(60, quarterLength=0.5)
    n1.volume = volume.Volume(velocity=80)
    p.insert(0.0, n1)
    n2 = note.Note(62, quarterLength=0.5)
    n2.volume = volume.Volume(velocity=70)
    p.insert(0.5, n2)
    return p


@pytest.mark.slow
def test_rt_play_live(monkeypatch):
    midi = DummyMidiOut()
    rtmidi_stub = types.ModuleType("rtmidi")
    rtmidi_stub.MidiOut = lambda: midi
    monkeypatch.setattr(rt_midi_streamer, "rtmidi", rtmidi_stub)

    def gen(idx: int):
        return _make_part() if idx == 0 else None

    streamer = rt_midi_streamer.RtMidiStreamer("dummy", bpm=120.0, buffer_ms=0.0)
    asyncio.run(
        streamer.play_live(gen, buffer_ahead=2, parallel_bars=2, late_humanize=True)
    )
    on_times = [t for t, msg in midi.events if msg[0] == 0x90]
    assert len(on_times) == 2
    diff = on_times[1] - on_times[0]
    bpm = 120
    assert abs(diff - (60 / bpm) / 2) < (60 / bpm) * 0.25
