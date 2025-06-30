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


def _get_note_at(notes, target_off):
    for n in notes:
        if abs(float(n.offset) - target_off) <= 1 / 480:
            return n
    raise AssertionError(f"note at {target_off} not found")


def test_build_up_pattern() -> None:
    gen = make_gen()

    section_v = {
        "key_signature": "C",
        "chord": "G7",
        "melody": [],
        "groove_kicks": [0.0],
    }
    part = gen.render_part(section_v, next_section_data={"chord": "Cmaj7"})
    notes = list(part.flatten().notes)
    assert abs(notes[0].offset - 0.0) <= 1 / 480
    assert _get_note_at(notes, 2.0).pitch.name == "D"
    assert _get_note_at(notes, 3.0).pitch.name in {"E-", "D#"}
