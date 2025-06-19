import pytest
from utilities.generator_factory import GenFactory

def test_build_from_config_default_drum_map(rhythm_library):
    main_cfg = {
        "global_settings": {
            "time_signature": "4/4",
            "tempo_bpm": 120,
            "key_tonic": "C",
            "key_mode": "major",
        },
        "sections_to_generate": ["A"],
        "part_defaults": {"drums": {"role": "drums"}},
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
    }

    gens = GenFactory.build_from_config(main_cfg, rhythm_library)
    drum_gen = gens["drums"]

    assert drum_gen.drum_map_name == "gm"
    assert drum_gen.drum_map.get("kick")[1] == 36

