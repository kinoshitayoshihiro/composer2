import json
from pathlib import Path
from music21 import stream

from generator.drum_generator import DrumGenerator, RESOLUTION
from tests.helpers.events import make_event

class FillDrum(DrumGenerator):
    def _render_part(self, section_data, next_section_data=None):
        blocks = []
        for i in range(8):
            blocks.append({
                "absolute_offset": i * 4.0,
                "humanized_offset_beats": i * 4.0,
                "humanized_duration_beats": 4.0,
                "q_length": 4.0,
                "is_first_in_section": i == 0,
                "part_params": {"drums": {"final_style_key_for_render": "main"}},
            })
        part = stream.Part(id=self.part_name)
        self._render(blocks, part, section_data)
        return part

def test_fill_inserted_at_bar8(tmp_path):
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    heatmap_path = tmp_path / "heatmap.json"
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)

    pattern_lib = {
        "main": {
            "pattern": [
                make_event(instrument="kick", offset=0.0),
                make_event(instrument="kick", offset=2.0),
            ],
            "length_beats": 4.0,
            "fill_patterns": ["fill"],
            "preferred_fill_positions": [8],
        },
        "fill": {
            "pattern": [
                make_event(instrument="tom1", offset=0.0, velocity_factor=0.6),
                make_event(instrument="tom2", offset=1.0, velocity_factor=0.8),
                make_event(instrument="tom3", offset=2.0, velocity_factor=1.0),
                make_event(instrument="snare", offset=3.0, velocity_factor=1.2),
            ],
            "length_beats": 4.0,
        },
    }

    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
        "rng_seed": 0,
    }

    drum = FillDrum(main_cfg=cfg, part_name="drums", part_parameters=pattern_lib)
    section = {"absolute_offset": 0.0, "q_length": 32.0, "length_in_measures": 8, "part_params": {}}
    part = drum.compose(section_data=section)

    assert drum.get_fill_offsets() == [28.0]
    fill_notes = [n for n in part.flatten().notes if 28.0 <= float(n.offset) < 32.0]
    assert len(fill_notes) == 4
    velocities = {n.volume.velocity for n in fill_notes}
    assert len(velocities) >= 2


def test_fill_offsets_reset_between_calls(tmp_path):
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    heatmap_path = tmp_path / "heatmap.json"
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)

    pattern_lib = {
        "main": {
            "pattern": [
                make_event(instrument="kick", offset=0.0),
                make_event(instrument="kick", offset=2.0),
            ],
            "length_beats": 4.0,
            "fill_patterns": ["fill"],
            "preferred_fill_positions": [8],
        },
        "fill": {
            "pattern": [
                make_event(instrument="tom1", offset=0.0, velocity_factor=0.6),
                make_event(instrument="tom2", offset=1.0, velocity_factor=0.8),
                make_event(instrument="tom3", offset=2.0, velocity_factor=1.0),
                make_event(instrument="snare", offset=3.0, velocity_factor=1.2),
            ],
            "length_beats": 4.0,
        },
    }

    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
        "rng_seed": 0,
    }

    drum = FillDrum(main_cfg=cfg, part_name="drums", part_parameters=pattern_lib)
    section = {"absolute_offset": 0.0, "q_length": 32.0, "length_in_measures": 8, "part_params": {}}

    drum.compose(section_data=section)
    assert drum.get_fill_offsets() == [28.0]

    drum.compose(section_data=section)
    assert drum.get_fill_offsets() == [28.0]
