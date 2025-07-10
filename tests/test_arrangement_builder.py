from music21 import stream, note, instrument

from utilities.arrangement_builder import build_arrangement, score_to_pretty_midi


def _make_part(pitch: str) -> stream.Part:
    p = stream.Part()
    p.insert(0.0, instrument.Piano())
    p.append(note.Note(pitch, quarterLength=1.0))
    return p


def test_track_count_matches_generators() -> None:
    sections = [
        ("Verse", 0.0, {"piano": _make_part("C4"), "drums": _make_part("C2")}),
        ("Chorus", 4.0, {"piano": _make_part("D4"), "drums": _make_part("D2")}),
    ]
    score, generator_names = build_arrangement(sections)
    pm = score_to_pretty_midi(score)
    assert len(pm.instruments) == len(generator_names)
