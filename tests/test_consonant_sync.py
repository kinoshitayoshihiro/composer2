import json
from pathlib import Path

import pytest
from music21 import meter, stream

from generator.drum_generator import RESOLUTION, DrumGenerator
from utilities.groove_sampler_ngram import Event
from typing import cast
from tests.helpers.events import make_event
from utilities.timing_utils import align_to_consonant


def test_aligns_when_peak_nearby() -> None:
    off, vel = align_to_consonant(
        1.0,
        [0.48],
        120,
        lag_ms=10.0,
        radius_ms=30.0,
        velocity_boost=6,
        return_vel=True,
    )
    assert off == pytest.approx(0.94, abs=1e-6)
    assert vel == 6


def test_no_alignment_outside_window() -> None:
    off, vel = align_to_consonant(
        0.0,
        [0.3],
        120,
        lag_ms=10.0,
        radius_ms=30.0,
        velocity_boost=6,
        return_vel=True,
    )
    assert off == pytest.approx(0.0, abs=1e-6)
    assert vel == 0


def test_far_peak_no_shift() -> None:
    off, vel = align_to_consonant(
        0.0,
        [1.6],
        120,
        lag_ms=10.0,
        radius_ms=30.0,
        velocity_boost=6,
        return_vel=True,
    )
    assert off == pytest.approx(0.0, abs=1e-6)
    assert vel == 0


def _cfg(tmp_path: Path, mode: str) -> dict:
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    hp = tmp_path / "heatmap.json"
    with open(hp, "w") as f:
        json.dump(heatmap, f)
    return {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(hp),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
        "global_settings": {
            "use_consonant_sync": True,
            "consonant_sync_mode": mode,
            "tempo_bpm": 60,
        },
        "consonant_sync": {
            "lag_ms": 10.0,
            "note_radius_ms": 30.0,
            "velocity_boost": 6,
        },
    }


def _apply_hits(drum: DrumGenerator) -> list[tuple[float, int]]:
    part = stream.Part(id="drums")
    events = [
        make_event(instrument="kick", offset=0.5),
        make_event(instrument="snare", offset=1.25),
    ]
    drum._apply_pattern(
        part,
        events,
        0.0,
        2.0,
        100,
        "eighth",
        0.5,
        meter.TimeSignature("2/4"),
        {},
    )
    notes = sorted(part.flatten().notes, key=lambda n: n.offset)
    return [(float(n.offset), int(n.volume.velocity)) for n in notes]


@pytest.mark.parametrize("mode", ["note", "bar"])
def test_sync_modes(tmp_path: Path, mode: str) -> None:
    cfg = _cfg(tmp_path, mode)
    drum = DrumGenerator(
        main_cfg=cfg,
        global_settings=cfg["global_settings"],
        part_name="drums",
        part_parameters={},
    )
    drum.consonant_peaks = [0.50, 1.25]
    notes = _apply_hits(drum)
    if mode == "note":
        assert notes[0][0] == pytest.approx(0.49, abs=1 / 480)
        assert notes[1][0] == pytest.approx(1.24, abs=1 / 480)
        assert all(v >= 106 for _, v in notes)
    else:
        # bar mode should insert additional hits from PeakSynchroniser
        assert len(notes) >= 4


def test_invalid_mode_raises(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "foo")
    with pytest.raises(ValueError):
        DrumGenerator(
            main_cfg=cfg,
            global_settings=cfg["global_settings"],
            part_name="drums",
            part_parameters={},
        )
