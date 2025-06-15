import json
from pathlib import Path
from music21 import note

from generator.drum_generator import DrumGenerator, GM_DRUM_MAP, RESOLUTION

class AliasDrum(DrumGenerator):
    def _resolve_style_key(self, musical_intent, overrides, section_data=None):
        return "alias_pattern"

def test_drum_alias_mapping(tmp_path, rhythm_library):
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    heatmap_path = tmp_path / "heatmap.json"
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)
    pattern_lib = {
        "alias_pattern": {
            "pattern": [
                {"instrument": "hh", "offset": 0.0, "duration": 0.25},
                {"instrument": "shaker_soft", "offset": 1.0, "duration": 0.25},
            ],
            "length_beats": 2.0,
        }
    }
    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
    }
    drum = AliasDrum(
        main_cfg=cfg,
        part_name="drums",
        part_parameters=pattern_lib,
        global_time_signature="4/4",
        global_tempo=120,
    )
    section = {"absolute_offset": 0.0, "q_length": 2.0, "musical_intent": {}, "part_params": {}}
    part = drum.compose(section_data=section)
    mids = [n.pitch.midi for n in part.flatten().notes]
    assert GM_DRUM_MAP["closed_hi_hat"] in mids
    assert GM_DRUM_MAP["shaker"] in mids
