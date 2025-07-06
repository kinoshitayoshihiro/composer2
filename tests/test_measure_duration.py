from music21 import instrument, stream
from generator.base_part_generator import BasePartGenerator


class DummyGenerator(BasePartGenerator):
    def _render_part(self, section_data, next_section_data=None):
        return stream.Part()


def test_measure_duration_property():
    gen = DummyGenerator(
        global_settings={},
        default_instrument=instrument.Piano(),
        part_name="dummy",
        global_tempo=120,
        global_time_signature="6/8",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )
    assert abs(gen.measure_duration - gen.bar_length) < 1e-6
