import types
import pytest
from music21 import note

from utilities import midi_capture


class DummyMidiIn:
    def __init__(self):
        self.callback = None
        self.events = []

    def get_ports(self):
        return ["dummy"]

    def open_port(self, idx):
        assert idx == 0

    def ignore_types(self, sysex=False, timing=False, sensing=False):
        pass

    def set_callback(self, cb):
        self.callback = cb

    def cancel_callback(self):
        self.callback = None

    def emit(self, msg, delta=0.0):
        if self.callback:
            self.callback((msg, delta), None)


@pytest.fixture
def recorder(monkeypatch):
    midi = DummyMidiIn()
    monkeypatch.setattr(midi_capture, "rtmidi", types.SimpleNamespace(MidiIn=lambda: midi))
    rec = midi_capture.MIDIRecorder("dummy", bpm=120.0)
    rec.start_recording()
    midi.emit([0x90, 60, 100], 0.0)
    midi.emit([0x80, 60, 0], 0.5)
    return rec, midi


def test_midi_recorder_basic(recorder):
    rec, midi = recorder
    part = rec.stop_recording()
    notes = list(part.flatten().notes)
    assert len(notes) == 1
    n: note.Note = notes[0]
    assert n.pitch.midi == 60
    assert abs(n.quarterLength - 1.0) < 0.001
