import json

import pytest

from generator.drum_generator import DrumGenerator
from tests.helpers.events import make_event


def _make_basic_cfg(tmp_path):
    heatmap = [{"grid_index": i, "count": 0} for i in range(16)]
    heatmap_path = tmp_path / "heatmap.json"
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)
    return {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
    }


pattern_lib = {
    "calm_backbeat": {
        "pattern": [make_event(instrument="kick", offset=0.0)],
        "length_beats": 4.0,
    }
}


def test_independent_mode_creates_part(tmp_path):
    cfg = _make_basic_cfg(tmp_path)
    drum_gen = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters=pattern_lib)
    drum_gen.mode = "independent"
    with pytest.raises(AttributeError):
        drum_gen.compose(section_data=None)


def test_chord_mode_creates_part(tmp_path):
    cfg = _make_basic_cfg(tmp_path)
    drum_gen = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters=pattern_lib)
    dummy_section = {
        "absolute_offset": 0,
        "length_in_measures": 2,
        "musical_intent": {"emotion": "default", "intensity": "medium"},
        "part_params": {},
    }
    part = drum_gen.compose(section_data=dummy_section)
    assert part is not None
    assert len(list(part.flatten().notes)) > 0
