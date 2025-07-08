import pytest
from music21 import note
from generator.vocal_generator import VocalGenerator, PhonemeArticulation, text_to_phonemes


def test_text_to_phonemes_roundtrip():
    text = "かきくけこ"
    expected = ["ka", "ki", "ku", "ke", "ko"]
    assert text_to_phonemes(text) == expected

    gen = VocalGenerator()
    midivocal_data = [
        {"offset": float(i), "pitch": "C4", "length": 1.0, "velocity": 80}
        for i in range(len(expected))
    ]
    part = gen.compose(midivocal_data, processed_chord_stream=[], humanize_opt=False, lyrics_words=[text])
    phonemes = []
    for n in part.flatten().notes:
        for a in n.articulations:
            if isinstance(a, PhonemeArticulation):
                phonemes.append(a.phoneme)
    assert phonemes == expected


def test_text_to_phonemes_multichar():
    assert text_to_phonemes("きゃきゅ") == ["kya", "kyu"]


def test_text_to_phonemes_multichar_full():
    # verify all multi-char entries map correctly
    assert text_to_phonemes("きゃきゅきょ") == ["kya", "kyu", "kyo"]


def test_text_to_phonemes_empty_and_unknown():
    # empty input returns empty list
    assert text_to_phonemes("") == []
    # unknown characters are returned as-is
    assert text_to_phonemes("♪ABC") == ["♪", "A", "B", "C"]
