import json
from music21 import stream, note, volume
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


<<<<<<< codex/add-fade_beats-parameter-and-update-tests
def test_velocity_fade_custom_width(tmp_path):
=======
def test_velocity_fade_respects_existing_velocities(tmp_path):
>>>>>>> infra/zero-green
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    heatmap_path = tmp_path / "heatmap.json"
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)

<<<<<<< codex/add-fade_beats-parameter-and-update-tests
    pattern_lib = {
        "main": {
            "pattern": [{"instrument": "kick", "offset": i} for i in range(4)],
            "length_beats": 4.0,
            "fill_patterns": ["f"],
            "options": {"fade_beats": 3.0},
        },
        "f": {"pattern": [{"instrument": "snare", "offset": 0.0}], "length_beats": 4.0},
    }

=======
    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
    }

    drum = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters={})
    part = stream.Part(id="drums")

    base_velocities = [50, 75, 100]
    for i, vel in enumerate(base_velocities):
        n = note.Note("C4")
        n.volume = volume.Volume(velocity=vel)
        n.offset = i * 0.5
        part.insert(n.offset, n)

    drum._velocity_fade_into_fill(part, 1.5, fade_beats=1.5)
    count = len(base_velocities)
    expected = []
    for idx, base in enumerate(base_velocities):
        scale = 0.8 + (0.2 * (idx + 1) / count)
        expected.append(int(max(1, min(127, base * scale))))

    final = [n.volume.velocity for n in sorted(part.flatten().notes, key=lambda n: n.offset)]
    assert final == expected


def test_velocity_fade_duplicate_offsets(tmp_path):
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    heatmap_path = tmp_path / "heatmap.json"
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)

>>>>>>> infra/zero-green
    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
<<<<<<< codex/add-fade_beats-parameter-and-update-tests
        "rng_seed": 1,
    }

    drum = FadeDrum(main_cfg=cfg, part_name="drums", part_parameters=pattern_lib)
    section = {"absolute_offset": 0.0, "q_length": 12.0, "length_in_measures": 3, "part_params": {}}
    part = drum.compose(section_data=section)

    offsets = drum.get_fill_offsets()
    assert offsets
    fill_offset = offsets[0]
    notes_before = sorted(
        [n for n in part.flatten().notes if fill_offset - 3.0 <= n.offset < fill_offset],
        key=lambda n: n.offset,
    )
    velocities = [n.volume.velocity for n in notes_before]
    assert velocities == sorted(velocities)
=======
    }

    drum = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters={})
    part = stream.Part(id="drums")

    for i in range(3):
        n = note.Note("C4")
        n.volume = volume.Volume(velocity=60)
        n.offset = i * 0.5
        part.insert(n.offset, n)

    drum.fill_offsets = [1.5, 1.5]
    for off in drum.fill_offsets:
        drum._velocity_fade_into_fill(part, off, fade_beats=1.5)

    final = [n.volume.velocity for n in sorted(part.flatten().notes, key=lambda n: n.offset)]
    assert final == sorted(final)
>>>>>>> infra/zero-green
