from music21 import note, stream

from utilities.articulation_mapper import add_key_switches


def test_add_key_switches() -> None:
    part = stream.Part()
    part.append(note.Note("C2", quarterLength=1.0))
    part.append(note.Note("D2", quarterLength=1.0))

    add_key_switches(part, {"finger": 28, "slap": 25, "mute": 29})
    notes = list(part.flatten().notes)
    assert notes[0].pitch.midi == 28
    assert notes[0].offset == -2.0
    assert len(notes) == 3
