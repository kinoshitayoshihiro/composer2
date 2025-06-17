from music21 import stream, note

from modular_composer import compose
from generator.base_part_generator import BasePartGenerator
from utilities.generator_factory import GenFactory


class DummyGenerator(BasePartGenerator):
    def _render_part(self, section_data, next_section_data=None):
        p = stream.Part(id="dummy")
        p.insert(0.0, note.Note("C4"))
        return p


def test_element_insertion(monkeypatch, rhythm_library):
    def fake_build_from_config(cfg, rl):
        gen = DummyGenerator(
            global_settings={},
            default_instrument=None,
            part_name="dummy",
            global_tempo=120,
            global_time_signature="4/4",
            global_key_signature_tonic="C",
            global_key_signature_mode="major",
            main_cfg=cfg,
        )
        return {"dummy": gen}

    monkeypatch.setattr(GenFactory, "build_from_config", staticmethod(fake_build_from_config))

    main_cfg = {
        "global_settings": {"time_signature": "4/4", "tempo_bpm": 120},
        "sections_to_generate": ["A"],
        "part_defaults": {"dummy": {}},
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
    }
    chordmap = {
        "sections": {
            "A": {
                "processed_chord_events": [
                    {
                        "absolute_offset_beats": 0.0,
                        "humanized_duration_beats": 1.0,
                        "chord_symbol_for_voicing": "C",
                    }
                ],
                "musical_intent": {"emotion": "neutral"},
                "expression_details": {},
            }
        }
    }

    score, _ = compose(main_cfg, chordmap, rhythm_library)
    notes = list(score.parts[0].recurse().getElementsByClass(note.Note))
    assert len(notes) == 1
    assert notes[0].offset == 0.0
