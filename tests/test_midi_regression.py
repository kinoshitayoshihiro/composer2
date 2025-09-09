import os
from pathlib import Path

import pytest

from generator import ModularComposer
from tools.golden import compare_midi, update_golden


@pytest.mark.parametrize(
    "style",
    [
        "rock_drive_loop",
        "brush_light_loop",
        "funk_groove",
        "ballad_backbeat",
    ],
)
def test_midi_regression(style, tmp_path):
    composer = ModularComposer(global_settings={"tempo_bpm": 120})
    path = tmp_path / f"{style}.mid"
    composer.render(style=style, output_path=str(path))
    golden_path = Path(f"data/golden/{style}.b64")
    if os.getenv("UPDATE_GOLDENS") == "1":
        update_golden(path, golden_path)
        pytest.skip("golden regenerated")
    diff = compare_midi(golden_path, path)
    assert not diff, f"MIDI regression detected for {style}: {diff}"
