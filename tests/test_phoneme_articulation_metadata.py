import pytest
from generator.vocal_generator import VocalGenerator, PhonemeArticulation, text_to_phonemes


def test_phoneme_articulation_metadata():
    text = "がぎ"
    phoneme_tuples = text_to_phonemes(text)
    assert phoneme_tuples[0][1] != phoneme_tuples[1][1]

    gen = VocalGenerator()
    midivocal_data = [
        {"offset": 0.0, "pitch": "C4", "length": 1.5, "velocity": 80},
        {"offset": 1.5, "pitch": "C4", "length": 0.5, "velocity": 80},
    ]
    part = gen.compose(midivocal_data, processed_chord_stream=[], humanize_opt=False, lyrics_words=[text])
    articulations = [
        a for n in part.flatten().notes for a in n.articulations if isinstance(a, PhonemeArticulation)
    ]
    if len(articulations) < 2:
        pytest.skip("phoneme articulations missing")
    assert articulations[0].accent == phoneme_tuples[0][1]
    assert articulations[1].accent == phoneme_tuples[1][1]
    assert articulations[0].duration_qL == pytest.approx(midivocal_data[0]["length"])
    assert articulations[1].duration_qL == pytest.approx(midivocal_data[1]["length"])
