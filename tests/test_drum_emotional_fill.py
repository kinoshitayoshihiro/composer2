import json
from music21 import stream
from generator.drum_generator import DrumGenerator, RESOLUTION
from tests.helpers.events import make_event


class EmotionalFillDrum(DrumGenerator):
    def _render_part(self, section_data, next_section_data=None):
        lengths = [4.0, 2.0, 6.0, 4.0]
        blocks = []
        offset = 0.0
        for i, ln in enumerate(lengths):
            blocks.append(
                {
                    "absolute_offset": offset,
                    "humanized_offset_beats": offset,
                    "humanized_duration_beats": ln,
                    "q_length": ln,
                    "musical_intent": {"emotion_intensity": 0.9 if i == 1 else 0.1},
                    "part_params": {"drums": {"final_style_key_for_render": "main"}},
                }
            )
            offset += ln
        section_data["length_in_measures"] = int(sum(lengths) / 4)
        part = stream.Part(id=self.part_name)
        self._render(blocks, part, section_data)
        return part


class EdgePeakDrum(DrumGenerator):
    def __init__(self, lengths, intensities, **kwargs):
        super().__init__(**kwargs)
        self.lengths = lengths
        self.intensities = intensities

    def _render_part(self, section_data, next_section_data=None):
        blocks = []
        offset = 0.0
        for i, ln in enumerate(self.lengths):
            blocks.append(
                {
                    "absolute_offset": offset,
                    "humanized_offset_beats": offset,
                    "humanized_duration_beats": ln,
                    "q_length": ln,
                    "musical_intent": {"emotion_intensity": self.intensities[i]},
                    "part_params": {"drums": {"final_style_key_for_render": "main"}},
                }
            )
            offset += ln
        section_data["length_in_measures"] = int(sum(self.lengths) / 4)
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
            "pattern": [make_event(instrument="kick", offset=0.0)],
            "length_beats": 4.0,
            "fill_patterns": ["f1", "f2"],
        },
        "f1": {
            "pattern": [make_event(instrument="snare", offset=0.0)],
            "length_beats": 4.0,
        },
        "f2": {"pattern": [make_event(instrument="tom1", offset=0.0)], "length_beats": 4.0},
    }

    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
        "rng_seed": 0,
    }

    drum = EmotionalFillDrum(
        main_cfg=cfg, part_name="drums", part_parameters=pattern_lib
    )
    section = {
        "absolute_offset": 0.0,
        "q_length": 16.0,
        "length_in_measures": 4,
        "part_params": {},
    }
    part = drum.compose(section_data=section)

    offsets = drum.get_fill_offsets()
    assert offsets
    assert abs(offsets[0] - 4.0) < 0.1
    assert offsets[0] < 12.0


def test_preferred_fill_position_clamped_low(tmp_path):
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    heatmap_path = tmp_path / "heatmap.json"
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)

    pattern_lib = {
        "main": {
            "pattern": [make_event(instrument="kick", offset=0.0)],
            "length_beats": 4.0,
        },
    }

    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
        "rng_seed": 0,
    }

    drum = EdgePeakDrum(
        [4.0, 4.0],
        [0.9, 0.1],
        main_cfg=cfg,
        part_name="drums",
        part_parameters=pattern_lib,
    )
    section = {
        "absolute_offset": 0.0,
        "q_length": 8.0,
        "length_in_measures": 2,
        "part_params": {},
    }
    drum.compose(section_data=section)
    assert section.get("preferred_fill_positions") == [1]


def test_preferred_fill_position_clamped_high(tmp_path):
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    heatmap_path = tmp_path / "heatmap.json"
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)

    pattern_lib = {
        "main": {
            "pattern": [make_event(instrument="kick", offset=0.0)],
            "length_beats": 4.0,
        },
    }

    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
        "rng_seed": 0,
    }

    drum = EdgePeakDrum(
        [7.9, 0.1],
        [0.1, 0.9],
        main_cfg=cfg,
        part_name="drums",
        part_parameters=pattern_lib,
    )
    section = {
        "absolute_offset": 0.0,
        "q_length": 8.0,
        "length_in_measures": 2,
        "part_params": {},
    }
    drum.compose(section_data=section)
    assert section.get("preferred_fill_positions") == [2]
