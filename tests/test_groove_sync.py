import os
import sys
import json
import importlib.util
from pathlib import Path
from music21 import note, stream

# Set up module import similar to other tests
ROOT = Path(__file__).resolve().parents[1]
pkg = sys.modules.setdefault("generator", type(sys)("generator"))
pkg.__path__ = [str(ROOT / "generator")]

spec = importlib.util.spec_from_file_location(
    "generator.drum_generator",
    ROOT / "generator" / "drum_generator.py",
    submodule_search_locations=[str(ROOT / "generator")],
)
_mod = importlib.util.module_from_spec(spec)
sys.modules["generator.drum_generator"] = _mod
spec.loader.exec_module(_mod)
DrumGenerator = _mod.DrumGenerator
RESOLUTION = _mod.RESOLUTION

class GrooveTestDrum(DrumGenerator):
    def _resolve_style_key(self, musical_intent, overrides, section_data=None):
        return "simple"

def test_groove_offsets(tmp_path: Path):
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
    pattern_lib = {
        "simple": {
            "pattern": [
                {"offset": 0.0, "duration": 0.25, "instrument": "snare"},
                {"offset": 0.25, "duration": 0.25, "instrument": "snare"},
            ],
            "length_beats": 4.0,
        }
    }
    drum = GrooveTestDrum(main_cfg=cfg, part_name="drums", part_parameters=pattern_lib)

    section = {"absolute_offset": 0.0, "q_length": 4.0, "musical_intent": {}, "part_params": {}}
    part = drum.compose(section_data=section)

    offsets = [float(n.offset) for n in part.flatten().notes]
    offsets = [round(o, 2) for o in offsets]
    assert offsets == [0.1, 0.2]
