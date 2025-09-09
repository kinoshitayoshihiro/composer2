import glob
import pathlib
import subprocess
import sys
import importlib

import pytest

try:
    import torch  # noqa: F401
except Exception as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)
import pretty_midi

from utilities.midi_export import write_demo_bar


@pytest.mark.parametrize("mid_path", glob.glob("tests/golden_midi/*.mid"))
def test_gm_cli(mid_path: str) -> None:
    if pathlib.Path(mid_path).stat().st_size == 0:
        pytest.skip("golden MIDI missing")
    proc = subprocess.run([
        sys.executable,
        "-m",
        "modular_composer.cli",
        "gm-test",
        mid_path,
    ], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "All golden MIDI match." in proc.stdout


def test_gm_cli_update(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "demo.mid"
    write_demo_bar(target)
    pm = pretty_midi.PrettyMIDI(str(target))
    pm.instruments[0].notes[0].velocity = 64
    pm.write(str(target))
    proc = subprocess.run([
        sys.executable,
        "-m",
        "modular_composer.cli",
        "gm-test",
        str(target),
        "--update",
    ])
    assert proc.returncode == 0
    proc = subprocess.run([
        sys.executable,
        "-m",
        "modular_composer.cli",
        "gm-test",
        str(target),
    ], capture_output=True)
    assert proc.returncode == 0
