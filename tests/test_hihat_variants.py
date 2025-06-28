import json
from pathlib import Path
from music21 import stream
from generator.drum_generator import DrumGenerator, GM_DRUM_MAP, RESOLUTION
from utilities.groove_sampler_ngram import Event
from typing import cast


def make_gen(tmp_path: Path) -> DrumGenerator:
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    heatmap_path = tmp_path / "heat.json"
    with heatmap_path.open("w") as f:
        json.dump(heatmap, f)
    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
    }
    return DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters={})


def test_edge_and_pedal_hits(tmp_path: Path):
    gen = make_gen(tmp_path)
    part = stream.Part(id="drums")
    events = [
        cast(Event, {"instrument": "chh", "offset": 0.0, "velocity": 40}),
        cast(Event, {"instrument": "hh", "offset": 1.0, "velocity": 60, "pedal": True}),
    ]
    gen._apply_pattern(
        part,
        cast(list[Event], events),
        0.0,
        4.0,
        100,
        "eighth",
        0.5,
        gen.global_ts,
        {},
    )
    mids = [n.pitch.midi for n in part.flatten().notes]
    assert mids[0] == GM_DRUM_MAP["hh_edge"][1]
    assert mids[1] == GM_DRUM_MAP["hh_pedal"][1]

