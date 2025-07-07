import pytest
from music21 import note
from generator.vocal_generator import VocalGenerator


def test_assign_lyrics_to_notes():
    gen = VocalGenerator()
    midivocal_data = [
        {"offset": 0.0, "pitch": "C4", "length": 1.0, "velocity": 80},
        {"offset": 1.0, "pitch": "D4", "length": 1.0, "velocity": 80},
        {"offset": 2.0, "pitch": "E4", "length": 1.0, "velocity": 80},
        {"offset": 3.0, "pitch": "F4", "length": 1.0, "velocity": 80},
    ]
    part = gen.compose(midivocal_data, processed_chord_stream=[], humanize_opt=False, lyrics_words=["あい", "あい"])
    lyrics = [n.lyric for n in part.flatten().notes]
    assert lyrics == ["あ", "い", "あ", "い"]

