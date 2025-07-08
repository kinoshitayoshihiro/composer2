import logging
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


def test_single_word_two_notes():
    gen = VocalGenerator()
    midivocal_data = [
        {"offset": 0.0, "pitch": "C4", "length": 2.0, "velocity": 80},
        {"offset": 2.0, "pitch": "D4", "length": 2.0, "velocity": 80},
    ]
    part = gen.compose(midivocal_data, processed_chord_stream=[], humanize_opt=False, lyrics_words=["あい"])
    lyrics = [n.lyric for n in part.flatten().notes]
    assert lyrics == ["あ", "い"]


def test_vocal_lyrics_empty(caplog):
    gen = VocalGenerator()
    midivocal_data = [
        {"offset": 0.0, "pitch": "C4", "length": 1.0, "velocity": 80},
        {"offset": 1.0, "pitch": "D4", "length": 1.0, "velocity": 80},
    ]
    with caplog.at_level(logging.WARNING):
        part = gen.compose(
            midivocal_data, processed_chord_stream=[], humanize_opt=False, lyrics_words=[]
        )
    lyrics = [n.lyric for n in part.flatten().notes]
    assert lyrics == [None, None]
    assert any("lyrics_words" in r.getMessage() for r in caplog.records)


def test_vocal_lyrics_mismatch(caplog):
    gen = VocalGenerator()
    midivocal_data = [
        {"offset": 0.0, "pitch": "C4", "length": 1.0, "velocity": 80},
        {"offset": 1.0, "pitch": "D4", "length": 1.0, "velocity": 80},
    ]
    with caplog.at_level(logging.WARNING):
        part = gen.compose(
            midivocal_data,
            processed_chord_stream=[],
            humanize_opt=False,
            lyrics_words=["あ", "い", "う"],
        )
    lyrics = [n.lyric for n in part.flatten().notes]
    assert lyrics == ["あ", "い"]
    assert any("syllables" in r.getMessage() for r in caplog.records)

