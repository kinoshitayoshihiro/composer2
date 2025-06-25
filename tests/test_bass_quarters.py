import random
from music21 import instrument
from generator.bass_generator import BassGenerator


def make_gen():
    gs = {"swing_ratio": 0.0}
    cfg = {"global_settings": {"key_tonic": "C", "key_mode": "major"}, "rng_seed": 0}
    return BassGenerator(
        part_name="bass",
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        main_cfg=cfg,
        global_settings=gs,
        part_parameters={
            "root_quarters": {
                "pattern_type": "fixed_pattern",
                "pattern": [
                    {"offset": i, "duration": 1.0, "type": "root"} for i in range(4)
                ],
                "reference_duration_ql": 4.0,
            }
        },
    )


def test_quarter_notes_simple():
    gen = make_gen()
    section = {
        "section_name": "Verse",
        "absolute_offset": 0.0,
        "q_length": 4.0,
        "chord_symbol_for_voicing": "C",
        "part_params": {"bass": {"rhythm_key": "root_quarters", "velocity": 60}},
        "musical_intent": {},
    }
    part = gen.compose(section_data=section)
    notes = list(part.flatten().notes)
    assert len(notes) == 4
    assert notes[0].pitch.name == "C"
    step_range = gen.global_settings.get("random_walk_step", 8)
    vels = [n.volume.velocity for n in notes]
    assert max(vels) - min(vels) <= 2 * step_range
