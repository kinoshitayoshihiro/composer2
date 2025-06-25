import pytest

import json
from pathlib import Path
from music21 import stream, meter
from generator.drum_generator import DrumGenerator, RESOLUTION
from utilities.timing_utils import align_to_consonant


def test_aligns_when_peak_nearby() -> None:
    off = align_to_consonant(1.0, [0.48], bpm=120, lag_ms=10.0)
    assert off == pytest.approx(0.94, abs=1e-6)


def test_no_alignment_outside_window() -> None:
    off = align_to_consonant(0.0, [0.3], bpm=120, lag_ms=10.0)
    assert off == pytest.approx(0.0, abs=1e-6)


def test_far_peak_no_shift() -> None:
    off = align_to_consonant(0.0, [1.6], bpm=120, lag_ms=10.0)
    assert off == pytest.approx(0.0, abs=1e-6)


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
            "tempo_bpm": 120,
        },
        "consonant_sync": {"lag_ms": 10.0},
    }


def _apply_single_kick(drum: DrumGenerator) -> list[float]:
    part = stream.Part(id="drums")
    events = [{"instrument": "kick", "offset": 1.0}]
    drum._apply_pattern(
        part,
        events,
        0.0,
        4.0,
        100,
        "eighth",
        0.5,
        meter.TimeSignature("4/4"),
        {},
    )
    return [float(n.offset) for n in part.flatten().notes]


@pytest.mark.parametrize("mode", ["note", "bar"])
def test_sync_modes(tmp_path: Path, mode: str) -> None:
    cfg = _cfg(tmp_path, mode)
    drum = DrumGenerator(
        main_cfg=cfg,
        global_settings=cfg["global_settings"],
        part_name="drums",
        part_parameters={},
    )
    drum.consonant_peaks = [0.48]
    offsets = _apply_single_kick(drum)
    if mode == "note":
        assert offsets == pytest.approx([0.94], abs=1e-6)
    else:
        assert len(offsets) == 3
        assert offsets[0] == pytest.approx(1.0, abs=1e-6)
        assert offsets[1] == pytest.approx(1.0208, abs=1e-3)
        assert offsets[2] == pytest.approx(1.2708, abs=1e-3)


def test_invalid_mode_raises(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "foo")
    with pytest.raises(ValueError):
        DrumGenerator(
            main_cfg=cfg,
            global_settings=cfg["global_settings"],
            part_name="drums",
            part_parameters={},
        )

