import pytest
from music21 import harmony, instrument
from generator.piano_generator import PianoGenerator

class SimplePiano(PianoGenerator):
    def _get_pattern_keys(self, musical_intent, overrides):
        return "rh_test", "lh_test"

def make_gen():
    patterns = {
        "rh_test": {"pattern": [{"offset": 0, "duration": 1, "type": "chord"}], "length_beats": 1.0},
        "lh_test": {"pattern": [{"offset": 0, "duration": 1, "type": "root"}], "length_beats": 1.0},
    }
    return SimplePiano(
        part_name="piano",
        part_parameters=patterns,
        default_instrument=instrument.Piano(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        main_cfg={},
    )


def _distance(a, b):
    a_sorted = sorted(a, key=lambda p: p.ps)
    b_sorted = sorted(b, key=lambda p: p.ps)
    m = min(len(a_sorted), len(b_sorted))
    return sum(abs(a_sorted[i].ps - b_sorted[i].ps) for i in range(m))


def _candidates(base):
    n = len(base)
    out = []
    for inv in range(n):
        inv_pitches = [p.transpose(12) if i < inv else p for i, p in enumerate(base)]
        for shift in (-12, 0, 12):
            out.append([pp.transpose(shift) for pp in inv_pitches])
    return out


@pytest.mark.parametrize("progression", [["C", "Am"], ["F", "G"]])
def test_voice_leading_progressions(progression):
    gen = make_gen()
    prev = None
    for label in progression:
        cs = harmony.ChordSymbol(label)
        base = gen._get_voiced_pitches(cs, 4, 4, "closed")
        voiced = gen._voice_minimal_leap("RH", cs, 4, 4, "closed")
        if prev is not None:
            cand = _candidates(base)
            expected = min(_distance(c, prev) for c in cand)
            actual = _distance(voiced, prev)
            assert abs(actual - expected) < 1e-5
        prev = voiced


@pytest.mark.parametrize(
    "style, expected_low, expected_high",
    [
        ("spread", 60, 84),
        ("closed", 60, 72),
        ("inverted", 60, 84),
    ],
)
def test_voicing_style_pitch_range(style, expected_low, expected_high):
    gen = make_gen()
    cs = harmony.ChordSymbol("C")
    pitches = gen._voice_minimal_leap("RH", cs, 4, 4, style)
    mins = min(p.midi for p in pitches)
    maxs = max(p.midi for p in pitches)
    assert expected_low <= mins <= expected_high
    assert expected_low <= maxs <= expected_high


def test_add9_and_sus2_chords():
    gen = make_gen()
    cs_add9 = harmony.ChordSymbol("Cadd9")
    add9_pitches = gen._get_voiced_pitches(cs_add9, 4, 4, "closed")
    names_add9 = {p.name for p in add9_pitches}
    assert {"C", "E", "G", "D"}.issubset(names_add9)

    cs_sus2 = harmony.ChordSymbol("Gsus2")
    sus2_pitches = gen._get_voiced_pitches(cs_sus2, 3, 4, "closed")
    names_sus2 = {p.name for p in sus2_pitches}
    assert names_sus2 == {"G", "A", "D"}

