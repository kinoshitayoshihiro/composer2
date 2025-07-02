from pathlib import Path
from unittest.mock import patch

from music21 import instrument, stream

from generator.bass_generator import BassGenerator


def test_custom_phrase_insertion(tmp_path: Path):
    tpl_path = tmp_path / "phrases.yaml"
    tpl_path.write_text(
        """
phrases:
  fill1:
    pattern:
      - [0.0, 60, 0.5]
      - [0.5, 62, 0.5]
    velocities: [80, 90]
"""
    )

    gen = BassGenerator(
        part_name="bass",
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        part_parameters={},
    )

    gen.load_phrase_templates(tpl_path)
    gen.set_phrase_insertions({"bridge": "fill1"})

    with patch.object(BassGenerator, "_render_part", return_value=stream.Part(id="bass")):
        part = gen.compose(
            section_data={
                "section_name": "bridge",
                "absolute_offset": 0.0,
                "q_length": 4.0,
                "chord_symbol_for_voicing": "C",
                "part_params": {},
                "musical_intent": {},
            }
        )

    notes = list(part.flatten().notes)
    assert len(notes) == 2
    assert notes[0].pitch.midi == 60
    assert notes[1].pitch.midi == 62

