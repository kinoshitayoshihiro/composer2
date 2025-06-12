import json
from pathlib import Path

from music21 import note, stream

from generator.drum_generator import DrumGenerator, RESOLUTION

class GrooveTestDrum(DrumGenerator):
    def _resolve_style_key(self, musical_intent, overrides, section_data=None):
        return "simple"

def test_groove_offsets(tmp_path: Path, rhythm_library):
    # groove profile with simple offsets
    gp = {"0": 0.1, "4": -0.05}
    gp_path = tmp_path / "gp.json"
    with open(gp_path, "w") as f:
        json.dump(gp, f)

    # minimal heatmap data
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    heatmap_path = tmp_path / "heatmap.json"
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)

    cfg = {
        "global_settings": {"groove_profile_path": str(gp_path), "groove_strength": 1.0},
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
    }
    pattern_lib = rhythm_library.drum_patterns or {}
    pattern_lib["simple"] = {
        "pattern": [
            {"offset": 0.0, "duration": 0.25, "instrument": "snare"},
            {"offset": 0.25, "duration": 0.25, "instrument": "snare"},
        ],
        "length_beats": 4.0,
    }
    drum = GrooveTestDrum(
        main_cfg=cfg,
        part_name="drums",
        part_parameters=pattern_lib,
    )

    section = {"absolute_offset": 0.0, "q_length": 4.0, "musical_intent": {}, "part_params": {}}
    part = drum.compose(section_data=section)

    offsets = [float(n.offset) for n in part.flatten().notes]
    offsets = [round(o, 2) for o in offsets]
    assert offsets == [0.1, 0.2]
