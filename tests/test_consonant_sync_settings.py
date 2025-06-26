import json
from pathlib import Path

import pytest

from generator.drum_generator import DrumGenerator, RESOLUTION


def _base_cfg(tmp_path: Path) -> dict:
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
            "consonant_sync_mode": "note",
            "tempo_bpm": 120,
        },
        "consonant_sync": {"lag_ms": 10.0},
    }


def test_invalid_radius(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg["consonant_sync"]["note_radius_ms"] = 500.0
    with pytest.raises(ValueError):
        DrumGenerator(
            main_cfg=cfg,
            global_settings=cfg["global_settings"],
            part_name="drums",
            part_parameters={},
        )


def test_invalid_boost(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg["consonant_sync"]["velocity_boost"] = -5
    with pytest.raises(ValueError):
        DrumGenerator(
            main_cfg=cfg,
            global_settings=cfg["global_settings"],
            part_name="drums",
            part_parameters={},
        )
