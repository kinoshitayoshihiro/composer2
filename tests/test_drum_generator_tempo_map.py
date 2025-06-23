import pytest
from generator.drum_generator import DrumGenerator
from utilities.tempo_utils import TempoMap


def test_drum_generator_tempo_map_interpolation(tmp_path):
    tmap = TempoMap([
        {"beat": 0, "bpm": 120},
        {"beat": 4, "bpm": 60},
        {"beat": 8, "bpm": 140},
    ])
    cfg = {
        "global_settings": {"tempo_bpm": 120},
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
    }
    drum = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters={}, tempo_map=tmap)
    assert drum._current_bpm(0) == 120
    assert drum._current_bpm(2) == 90
    assert drum._current_bpm(4) == 60
    assert drum._current_bpm(6) == 100
    assert drum._current_bpm(8) == 140
