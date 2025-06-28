import json
from pathlib import Path

from music21 import stream, pitch, note, duration as m21duration, volume as m21volume
from generator.drum_generator import DrumGenerator, RESOLUTION
from utilities.groove_sampler_ngram import Event
from typing import cast

class CurveDrum(DrumGenerator):
    def _resolve_style_key(self, musical_intent, overrides, section_data=None):
        return "curve"

    def _render_part(self, section_data, next_section_data=None):
        part = stream.Part(id=self.part_name)
        part.insert(0, self.default_instrument)
        events = [
            cast(
                Event,
                {
                    "offset": 0.0,
                    "duration": 0.25,
                    "instrument": "snare",
                    "velocity_factor": 1.0,
                    "velocity_layer": 0,
                },
            ),
            cast(
                Event,
                {
                    "offset": 0.5,
                    "duration": 0.25,
                    "instrument": "snare",
                    "velocity_factor": 1.0,
                    "velocity_layer": 1,
                },
            ),
        ]
        self._apply_pattern(
            part,
            cast(list[Event], events),
            section_data.get("absolute_offset", 0.0),
            4.0,
            80,
            "eighth",
            0.5,
            None,
            {},

            velocity_scale=1.0,
            velocity_curve=[0.5, 1.0],
        )
        return part

    def _make_hit(self, name: str, vel: int, ql: float, ev_def=None):
        n = note.Note()
        n.pitch = pitch.Pitch(midi=38)
        n.duration = m21duration.Duration(quarterLength=ql)
        n.volume = m21volume.Volume(velocity=vel)
        n.offset = 0.0
        return n

def test_velocity_curve_applied(tmp_path: Path, rhythm_library):
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    heatmap_path = tmp_path / "heat.json"
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)

    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "paths": {"drum_pattern_files": []},
    }
    pattern_lib = {
        "curve": {
            "pattern": [
                {"offset": 0.0, "duration": 0.25, "instrument": "snare", "velocity_factor": 1.0},
                {"offset": 0.5, "duration": 0.25, "instrument": "snare", "velocity_factor": 1.0},
            ],
            "length_beats": 4.0,
            "velocity_base": 80,
            "options": {"velocity_curve": [0.5, 1.0]},
        }
    }
    drum = CurveDrum(main_cfg=cfg, part_name="drums", part_parameters=pattern_lib)

    section = {"absolute_offset": 0.0, "q_length": 4.0, "musical_intent": {}, "part_params": {}}
    part = drum.compose(section_data=section)

    velocities = [n.volume.velocity for n in part.flatten().notes]
    assert velocities == [40, 80]


def test_velocity_curve_default(tmp_path: Path, rhythm_library):
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    heatmap_path = tmp_path / "heat.json"
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)

    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "paths": {"drum_pattern_files": []},
    }
    pattern_lib = {
        "base": {
            "pattern": [],
            "length_beats": 4.0,
            "velocity_base": 80,
        }
    }
    drum = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters=pattern_lib)

    resolved = drum._get_effective_pattern_def("base")
    assert resolved["velocity_curve"] == [1.0]
