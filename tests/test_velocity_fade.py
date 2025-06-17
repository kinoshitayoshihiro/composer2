import json
from music21 import stream
from generator.drum_generator import DrumGenerator, RESOLUTION

class FadeDrum(DrumGenerator):
    def _render_part(self, section_data, next_section_data=None):
        blocks = []
        for i in range(3):
            blocks.append({
                "absolute_offset": i * 4.0,
                "humanized_offset_beats": i * 4.0,
                "humanized_duration_beats": 4.0,
                "q_length": 4.0,
                "musical_intent": {"emotion_intensity": 0.9 if i == 1 else 0.1},
                "part_params": {"drums": {"final_style_key_for_render": "main"}},
            })
        section_data["length_in_measures"] = 3
        part = stream.Part(id=self.part_name)
        self._render(blocks, part, section_data)
        return part

def test_velocity_fade_into_fill(tmp_path):
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    heatmap_path = tmp_path / "heatmap.json"
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)

    pattern_lib = {
        "main": {
            "pattern": [{"instrument": "kick", "offset": i} for i in range(4)],
            "length_beats": 4.0,
            "fill_patterns": ["f"],
        },
        "f": {"pattern": [{"instrument": "snare", "offset": 0.0}], "length_beats": 4.0},
    }

    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
        "rng_seed": 1,
    }

    drum = FadeDrum(main_cfg=cfg, part_name="drums", part_parameters=pattern_lib)
    section = {"absolute_offset": 0.0, "q_length": 12.0, "length_in_measures": 3, "part_params": {}}
    part = drum.compose(section_data=section)

    offsets = drum.get_fill_offsets()
    assert offsets
    fill_offset = offsets[0]
    notes_before = sorted(
        [n for n in part.flatten().notes if fill_offset - 2.0 <= n.offset < fill_offset],
        key=lambda n: n.offset,
    )
    velocities = [n.volume.velocity for n in notes_before]
    assert velocities == sorted(velocities)
