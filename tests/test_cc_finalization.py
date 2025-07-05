from music21 import stream, note
from generator.base_part_generator import BasePartGenerator

class DummyGenerator(BasePartGenerator):
    def _render_part(self, section_data, next_section_data=None):
        p = stream.Part(id="d")
        n = note.Note("C4", quarterLength=1.0)
        n.volume.velocity = 90
        p.append(n)
        p._extra_cc = {(0.5, 91, 70), (0.0, 31, 40)}
        return p

def test_finalize_cc_sorted():
    gen = DummyGenerator(
        global_settings={},
        default_instrument=None,
        part_name="dummy",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )
    sec = {"section_name": "A", "q_length": 1.0, "musical_intent": {"intensity": "medium"}}
    part = gen.compose(section_data=sec)
    times = [e["time"] for e in part.extra_cc]
    assert times == sorted(times)
    assert all("cc" in e for e in part.extra_cc)
