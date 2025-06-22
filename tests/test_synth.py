import os
import pathlib
import shutil
import pytest

from utilities.synth import render_midi
from utilities.midi_export import write_demo_bar


def test_render_midi(tmp_path: pathlib.Path) -> None:
    if shutil.which("fluidsynth") is None:
        pytest.skip("fluidsynth not installed")
    sf2 = os.environ.get("SF2_PATH")
    if not sf2 or not pathlib.Path(sf2).exists():
        pytest.skip("SoundFont not provided")

    midi = tmp_path / "bar.mid"
    wav = tmp_path / "out.wav"
    write_demo_bar(midi)
    render_midi(midi, wav, sf2)
    assert wav.exists() and wav.stat().st_size > 0
