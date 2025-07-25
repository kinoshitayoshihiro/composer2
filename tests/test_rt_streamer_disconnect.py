import asyncio
import logging
from types import SimpleNamespace

from music21 import note, stream, volume

# Import the implementation module so monkeypatching works correctly
from utilities import rtmidi_streamer as rt_midi_streamer


class FlakyMidiOut:
    def __init__(self) -> None:
        self.count = 0
        self.events: list[list[int]] = []
        self.opened = 0

    def get_ports(self):
        return ["dummy"]

    def open_port(self, idx: int) -> None:
        self.opened += 1

    def send_message(self, msg):
        self.count += 1
        if self.count == 1:
            raise RuntimeError("disconnected")
        self.events.append(msg)


def make_part():
    p = stream.Part()
    n1 = note.Note(60, quarterLength=0.25)
    n1.volume = volume.Volume(velocity=80)
    p.insert(0.0, n1)
    n2 = note.Note(62, quarterLength=0.25)
    n2.volume = volume.Volume(velocity=70)
    p.insert(0.5, n2)
    return p


def test_rt_streamer_disconnect(monkeypatch, caplog):
    midi = FlakyMidiOut()
    rtmidi_stub = types.ModuleType("rtmidi")
    rtmidi_stub.MidiOut = lambda: midi
    monkeypatch.setattr(rt_midi_streamer, "rtmidi", rtmidi_stub)
    part = make_part()
    streamer = rt_midi_streamer.RtMidiStreamer("dummy", buffer_ms=0)
    with caplog.at_level(logging.ERROR):
        asyncio.run(streamer.play_stream(part))
    assert any("MIDI send failed" in r.message for r in caplog.records)
    assert midi.opened >= 2
    assert len(midi.events) >= 2
