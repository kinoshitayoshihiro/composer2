from music21 import instrument
from generator.piano_template_generator import PianoTemplateGenerator, PPQ


def make_gen():
    return PianoTemplateGenerator(
        part_name="piano",
        default_instrument=instrument.Piano(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )


def test_kick_lock_velocity():
    gen = make_gen()
    kicks = [0.0, 1.0, 2.0, 3.0]
    section = {
        "q_length": 4.0,
        "chord_symbol_for_voicing": "C",
        "groove_kicks": kicks,
        "musical_intent": {},
    }
    parts = gen.compose(section_data=section)
    all_notes = list(parts["piano_rh"].recurse().notes) + list(parts["piano_lh"].recurse().notes)
    base_vel = 70
    eps = 3 / PPQ
    for k in kicks:
        near = [n for n in all_notes if abs(float(n.offset) - k) <= eps]
        assert near
        assert all((n.volume.velocity or 0) > base_vel for n in near)
