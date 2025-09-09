import pytest
pytest.importorskip("mido")
from music21 import stream, note
from utilities.midi_export import music21_to_mido


def test_extra_cc_written():
    part = stream.Part(id="p")
    n = note.Note("C4", quarterLength=1.0)
    n.volume.velocity = 90
    part.insert(0.0, n)
    part.extra_cc = [{"time": 0.0, "cc": 20, "val": 64}]
    mid = music21_to_mido(part)
    msgs = [
        msg
        for track in mid.tracks
        for msg in track
        if msg.type == "control_change" and msg.control == 20
    ]
    assert len(msgs) == 1
