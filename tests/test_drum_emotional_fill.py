import json
from music21 import stream
from generator.drum_generator import DrumGenerator, RESOLUTION

class EmotionalFillDrum(DrumGenerator):
    def _render_part(self, section_data, next_section_data=None):
        num_bars = 4
        blocks = []
        for i in range(num_bars):
            blocks.append({
                "absolute_offset": i * 4.0,
                "humanized_offset_beats": i * 4.0,
                "humanized_duration_beats": 4.0,
                "q_length": 4.0,
                "musical_intent": {"emotion_intensity": 0.9 if i == 1 else 0.1},
                "part_params": {"drums": {"final_style_key_for_render": "main"}},
            })
        section_data["length_in_measures"] = num_bars
        part = stream.Part(id=self.part_name)
        self._render(blocks, part, section_data)
        return part


def test_emotional_peak_fill(tmp_path):
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    heatmap_path = tmp_path / "heatmap.json"
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)

    pattern_lib = {
        "main": {
            "pattern": [{"instrument": "kick", "offset": 0.0}],
            "length_beats": 4.0,
            "fill_patterns": ["f1", "f2"],
        },
        "f1": {"pattern": [{"instrument": "snare", "offset": 0.0}], "length_beats": 4.0},
        "f2": {"pattern": [{"instrument": "tom1", "offset": 0.0}], "length_beats": 4.0},
    }

    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
        "rng_seed": 0,
    }

    drum = EmotionalFillDrum(main_cfg=cfg, part_name="drums", part_parameters=pattern_lib)
    section = {"absolute_offset": 0.0, "q_length": 16.0, "length_in_measures": 4, "part_params": {}}
    part = drum.compose(section_data=section)

    offsets = drum.get_fill_offsets()
    assert offsets
    assert abs(offsets[0] - 4.0) < 0.1
    assert offsets[0] < 12.0
