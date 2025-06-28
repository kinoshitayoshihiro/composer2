import pytest
from music21 import stream, meter

from generator.drum_generator import DrumGenerator, RESOLUTION
from utilities.groove_sampler_ngram import Event
from typing import cast
from utilities.timing_utils import align_to_consonant


def _cfg(tmp_path, radius_ms: float) -> dict:
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    hp = tmp_path / "heatmap.json"
    import json
    with open(hp, "w") as f:
        json.dump(heatmap, f)
    return {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(hp),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
        "global_settings": {"use_consonant_sync": True, "consonant_sync_mode": "note", "tempo_bpm": 120},
        "consonant_sync": {
            "lag_ms": 10.0,
            "note_radius_ms": radius_ms,
            "velocity_boost": 6,
        },
    }


def _apply(drum: DrumGenerator, off_ql: float) -> tuple[float, int]:
    part = stream.Part(id="drums")
    events = [cast(Event, {"instrument": "kick", "offset": off_ql})]
    drum._apply_pattern(
        part,
        cast(list[Event], events),
        0.0,
        1.0,
        100,
        "eighth",
        0.5,
        meter.TimeSignature("1/4"),
        {},
    )
    n = part.flatten().notes[0]
    return float(n.offset), int(n.volume.velocity)


def test_aligns_within_radius(tmp_path):
    cfg = _cfg(tmp_path, 30.0)
    drum = DrumGenerator(main_cfg=cfg, global_settings=cfg["global_settings"], part_name="drums", part_parameters={})
    drum.consonant_peaks = [0.25, 1.0, 2.5]
    off, vel = _apply(drum, 0.52)
    expected = (0.25 - 0.01) / (60 / 120)
    assert off == pytest.approx(expected, abs=1/480)
    assert vel >= 106


def test_no_align_when_radius_small(tmp_path):
    off, vel = align_to_consonant(
        0.52,
        [0.5],
        120,
        lag_ms=10.0,
        radius_ms=10.0,
        velocity_boost=6,
        return_vel=True,
    )
    assert off == pytest.approx(0.52, abs=1e-6)
    assert vel == 0
