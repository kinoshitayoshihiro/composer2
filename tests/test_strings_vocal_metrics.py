import pretty_midi
from music21 import instrument

from generator.strings_generator import StringsGenerator


def test_strings_vocal_metrics_smoke(tmp_path):
    gen = StringsGenerator(
        global_settings={"tempo_bpm": 120},
        default_instrument=instrument.StringInstrument(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        part_name="strings",
    )
    section = {"chord_symbol_for_voicing": "C", "q_length": 4.0}
    vm = {"onsets": [0.0], "rests": [(1.0, 1.0)], "consonant_peaks": []}
    parts = gen.compose(section_data=section, vocal_metrics=vm)
    assert isinstance(parts, dict) and parts
