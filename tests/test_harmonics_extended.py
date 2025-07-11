from music21 import pitch

from generator.guitar_generator import GuitarGenerator
from utilities.harmonic_utils import apply_harmonic_to_pitch, choose_harmonic


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
    new, arts, _, _ = apply_harmonic_to_pitch(
        p,
        chord_pitches=[p],
        tuning_offsets=gen.tuning,
        base_midis=None,
        max_fret=gen.max_harmonic_fret,
        allowed_types=gen.harmonic_types,
        rng=gen.rng,
        prob=gen.prob_harmonic,
        volume_factor=gen.harmonic_volume_factor,
        gain_db=gen.harmonic_gain_db,
    )
    assert int(round(new.midi)) == int(round(p.midi)) + 12


def test_base_midis_override():
    p = pitch.Pitch("C4")
    result = choose_harmonic(
        p,
        tuning_offsets=None,
        chord_pitches=[p],
        open_string_midis=[60, 65, 70],
    )
    assert result is not None
    _, meta = result
    assert meta["string_idx"] == 0


def test_choose_harmonic_none_with_small_fret():
    p = pitch.Pitch("C4")
    res = choose_harmonic(p, [0, 0, 0], [p], max_fret=2)
    assert res is None
