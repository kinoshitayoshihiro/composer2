import json
from pathlib import Path
from music21 import stream
from generator.drum_generator import DrumGenerator, RESOLUTION
from utilities.groove_sampler_ngram import Event
from typing import cast


def make_gen(tmp_path: Path):
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    heatmap_path = tmp_path / "heat.json"
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)
    pattern = {
        "main": {
            "pattern": [
                {"instrument": "kick", "offset": 0.0, "velocity_layer": 0},
                {"instrument": "snare", "offset": 1.0, "velocity_layer": 1},
                {"instrument": "kick", "offset": 2.0, "velocity_layer": 2},
            ],
            "length_beats": 4.0,
            "velocity_base": 100,
            "options": {"velocity_curve": [0.5, 0.75, 1.0]},
        }
    }
    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
    }
    return DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters=pattern)


def test_velocity_curve_scaling(tmp_path: Path):
    gen = make_gen(tmp_path)
    part = stream.Part(id="drums")
    gen._apply_pattern(
        part,
        cast(list[Event], gen.part_parameters["main"]["pattern"]),
        0.0,
        4.0,
        100,
        "eighth",
        0.5,
        gen.global_ts,
        {},
        None,
        1.0,
        [0.5, 0.75, 1.0],
    )
    vels = [n.volume.velocity for n in part.flatten().notes]
    assert vels == [50, 75, 100]
