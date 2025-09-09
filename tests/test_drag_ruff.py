import json
from pathlib import Path
from music21 import stream
from generator.drum_generator import DrumGenerator, GM_DRUM_MAP
from utilities.groove_sampler_ngram import Event
from typing import cast
from tests.helpers.events import make_event


def _minimal_cfg(tmp_path: Path):
    heatmap = [{"grid_index": i, "count": 0} for i in range(16)]
    hp = tmp_path / "heatmap.json"
    with hp.open("w") as f:
        json.dump(heatmap, f)
    return {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(hp),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
        "rng_seed": 0,
    }


def _collect_notes(drum: DrumGenerator, event: Event) -> list:
    part = stream.Part(id="drums")
    drum._apply_pattern(
        part,
        [event],
        0.0,
        2.0,
        event.get("velocity", 90),
        "eighth",
        0.5,
        drum.global_ts,
        {},
    )
    return sorted(part.flatten().notes, key=lambda n: n.offset)


def test_drag_and_ruff(tmp_path: Path):
    cfg = _minimal_cfg(tmp_path)
    drum = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters={})

    drag_evt = make_event(instrument="snare", offset=0.5, type="drag", velocity=100)
    ruff_evt = make_event(instrument="snare", offset=1.5, type="ruff", velocity=100)

    drag_notes = _collect_notes(drum, drag_evt)
    ruff_notes = _collect_notes(drum, ruff_evt)

    assert len(drag_notes) == 3
    assert len(ruff_notes) == 4

    for notes in (drag_notes, ruff_notes):
        main_offset = notes[-1].offset
        main_vel = notes[-1].volume.velocity
        for n in notes:
            assert int(n.pitch.midi) == GM_DRUM_MAP["snare"][1]
        for g in notes[:-1]:
            dt = (main_offset - g.offset) * 60 / drum.global_tempo
            assert 0 < dt <= 0.03
            ratio = g.volume.velocity / main_vel
            assert 0.3 <= ratio <= 0.6
