from music21 import stream, note
from utilities import humanizer


def test_offset_profile_shift():
    part = stream.Part()
    part.append(note.Note("C4"))
    humanizer.load_profiles({"test": {"shift_ql": 0.25}})
    humanizer.apply_offset_profile(part, "test")
    assert part.recurse().notes[0].offset == 0.25
