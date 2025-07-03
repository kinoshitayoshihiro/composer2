from music21 import instrument
from generator.bass_generator import BassGenerator


def make_gen() -> BassGenerator:
    return BassGenerator(
        part_name="bass",
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        main_cfg={"global_settings": {"key_tonic": "C", "key_mode": "major"}},
    )


def test_cc31_emitted_once() -> None:
    gen = make_gen()
    section = {
        "section_name": "A",
        "absolute_offset": 0.0,
        "q_length": 2.0,
        "chord_symbol_for_voicing": "C",
        "musical_intent": {"intensity": "medium"},
        "part_params": {},
    }
    part = gen.compose(section_data=section)
    cc31 = [c for c in getattr(part, "extra_cc", []) if c.get("cc") == 31]
    assert len(cc31) == 1
