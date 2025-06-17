from music21 import instrument
from utilities.generator_factory import GenFactory


def test_sax_generator_inserts_instrument(rhythm_library):
    main_cfg = {
        "global_settings": {
            "time_signature": "4/4",
            "tempo_bpm": 120,
            "key_tonic": "C",
            "key_mode": "major",
        },
        "sections_to_generate": ["A"],
        "part_defaults": {"sax": {"role": "sax"}},
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
    }

    gens = GenFactory.build_from_config(main_cfg, rhythm_library)
    sax_gen = gens["sax"]

    section = {
        "section_name": "Verse",
        "absolute_offset": 0.0,
        "q_length": 4.0,
        "chord_symbol_for_voicing": "C",
        "part_params": {},
        "musical_intent": {},
        "tonic_of_section": "C",
        "mode": "major",
    }

    part = sax_gen.compose(section_data=section)
    insts = part.getElementsByClass(instrument.Instrument)
    assert isinstance(insts.first(), instrument.AltoSaxophone)
