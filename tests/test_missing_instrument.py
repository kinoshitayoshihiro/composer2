import inspect
from music21 import stream, note, instrument

import modular_composer as mc
from modular_composer import compose
from generator.base_part_generator import BasePartGenerator
from utilities.generator_factory import GenFactory


class DummyGenerator(BasePartGenerator):
    def _render_part(self, section_data, next_section_data=None):
        p = stream.Part()
        p.id = None  # keep id=None so compose assigns one
        p.insert(0.0, instrument.SnareDrum())
        p.append(note.Note('C4', quarterLength=section_data.get('q_length', 1.0)))
        return p


def test_missing_instrument(monkeypatch, rhythm_library):
    def fake_build_from_config(cfg, rl):
        gen = DummyGenerator(
            global_settings={},
            default_instrument=None,
            part_name='dummy',
            global_tempo=120,
            global_time_signature='4/4',
            global_key_signature_tonic='C',
            global_key_signature_mode='major',
            main_cfg=cfg,
        )
        return {'dummy': gen}

    monkeypatch.setattr(GenFactory, 'build_from_config', staticmethod(fake_build_from_config))

    captured = {}

    class DummyScore:
        def __init__(self, parts):
            frame = inspect.currentframe().f_back
            captured['part_streams'] = frame.f_locals['part_streams']
            self.parts = parts

    monkeypatch.setattr(mc.stream, 'Score', DummyScore)

    main_cfg = {
        'global_settings': {'time_signature': '4/4', 'tempo_bpm': 120},
        'sections_to_generate': ['A'],
        'part_defaults': {'dummy': {}},
        'paths': {'rhythm_library_path': 'data/rhythm_library.yml'},
    }
    chordmap = {
        'sections': {
            'A': {
                'processed_chord_events': [
                    {
                        'absolute_offset_beats': 0.0,
                        'humanized_duration_beats': 4.0,
                        'chord_symbol_for_voicing': 'C',
                    }
                ],
                'musical_intent': {'emotion': 'neutral'},
                'expression_details': {},
            }
        }
    }

    compose(main_cfg, chordmap, rhythm_library)

    for part in captured['part_streams'].values():
        insts = part.recurse().getElementsByClass(instrument.Instrument)
        assert len(insts) == 1
