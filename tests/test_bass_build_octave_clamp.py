from music21 import instrument
from generator.bass_generator import BassGenerator


def make_gen() -> BassGenerator:
    gs = {"bass_range_lo": 40, "bass_range_hi": 72}
    cfg = {"global_settings": {"key_tonic": "Bb", "key_mode": "major"}}
    return BassGenerator(
        part_name="bass",
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="Bb",
        global_key_signature_mode="major",
        main_cfg=cfg,
        global_settings=gs,
    )


def test_build_up_range_clamp() -> None:
    gen = make_gen()
    section = {
        "key_signature": "Bb",
        "chord": "F7",
        "melody": [],
        "groove_kicks": [0.0],
    }
    part = gen.render_part(section, next_section_data={"chord": "Bbmaj7"})
    for n in part.flatten().notes:
        assert n.pitch.midi >= gen.bass_range_lo
