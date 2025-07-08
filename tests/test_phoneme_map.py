import pytest
from music21 import note
from generator.vocal_generator import (
    VocalGenerator,
    PhonemeArticulation,
    text_to_phonemes,
    PHONEME_DICT,
)


@pytest.mark.parametrize("text,expected", PHONEME_DICT.items())
def test_text_to_phonemes_roundtrip(text, expected):
    res = text_to_phonemes(text)
    assert [p for p, _, _ in res] == [expected]

    gen = VocalGenerator()
    midivocal_data = [{"offset": 0.0, "pitch": "C4", "length": 1.0, "velocity": 80}]
    part = gen.compose(
        midivocal_data,
        processed_chord_stream=[],
        humanize_opt=False,
        lyrics_words=[text],
    )
    phonemes = [
        a.phoneme
        for n in part.flatten().notes
        for a in n.articulations
        if isinstance(a, PhonemeArticulation)
    ]
    assert phonemes == [expected]


def test_text_to_phonemes_multichar():
    res = text_to_phonemes("きゃきゅ")
    assert [p for p, _, _ in res] == ["kya", "kyu"]


def test_text_to_phonemes_multichar_full():
    # verify all multi-char entries map correctly
    res = text_to_phonemes("きゃきゅきょ")
    assert [p for p, _, _ in res] == ["kya", "kyu", "kyo"]


def test_phoneme_map_full():
    res = text_to_phonemes("がぎゃ")
    assert [p for p, _, _ in res] == ["ga", "gya"]


def test_text_to_phonemes_empty_and_unknown():
    # empty input returns empty list
    assert text_to_phonemes("") == []
    # unknown characters are returned as-is
    res = text_to_phonemes("♪ABC")
    assert [p for p, _, _ in res] == ["♪", "A", "B", "C"]
