from music21 import pitch

from generator.guitar_generator import GuitarGenerator
from utilities.harmonic_utils import choose_harmonic


def test_natural_5f_harmonic():
    p = pitch.Pitch("C4")
    new, meta = choose_harmonic(p, [0, 0, 0, 0, 0, 0], [p])
    assert meta["type"] == "natural"
    assert meta["touch_fret"] == 15
    assert int(round(new.midi)) == 91


def test_artificial_harmonic():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=None,
        part_name="g",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        enable_harmonics=True,
        prob_harmonic=1.0,
        harmonic_types=["artificial"],
        max_harmonic_fret=40,
        rng_seed=2,
    )
    p = pitch.Pitch("C4")
    new, arts, _, _ = gen._maybe_harmonic(p, [p])
    assert int(round(new.midi)) == int(round(p.midi)) + 12


def test_base_midis_override():
    p = pitch.Pitch("C4")
    result = choose_harmonic(
        p,
        tuning_offsets=None,
        chord_pitches=[p],
        base_midis=[60, 65, 70],
    )
    assert result is not None
    _, meta = result
    assert meta["string_idx"] == 0


def test_choose_harmonic_none_with_small_fret():
    p = pitch.Pitch("C4")
    res = choose_harmonic(p, [0, 0, 0], [p], max_fret=2)
    assert res is None
