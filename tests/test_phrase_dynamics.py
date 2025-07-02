from unittest.mock import patch
from pathlib import Path
from music21 import instrument, note, stream, volume as m21volume

from generator.bass_generator import BassGenerator


def test_phrase_intensity_applied(tmp_path):
    gen = BassGenerator(
        part_name="bass",
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        part_parameters={},
    )

    gen.set_phrase_map({"intro": {"bars": 1, "intensity": 0.5}})

    # Patch _render_part to produce a single note
    p = stream.Part(id="bass")
    n = note.Note("C3", quarterLength=4.0)
    n.volume = m21volume.Volume(velocity=80)
    p.insert(0, n)

    with patch.object(BassGenerator, "_render_part", return_value=p):
        part = gen.compose(
            section_data={
                "section_name": "intro",
                "absolute_offset": 0.0,
                "q_length": 4.0,
                "chord_symbol_for_voicing": "C",
                "part_params": {},
                "musical_intent": {},
            }
        )

    vel = part.flatten().notes[0].volume.velocity
    assert vel > 80

