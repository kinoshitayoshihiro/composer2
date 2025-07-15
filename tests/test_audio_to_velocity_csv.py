import numpy as np
import soundfile as sf
from pathlib import Path

from scripts.audio_to_velocity_csv import main as build


def test_silence_produces_empty_csv(tmp_path: Path) -> None:
    loops = tmp_path / "loops"
    loops.mkdir()
    sr = 22050
    sf.write(loops / "silence.wav", np.zeros(sr), sr)

    midi_dir = tmp_path / "midi"
    out_csv = tmp_path / "velocity.csv"
    build(["--loops-dir", str(loops), "--midi-dir", str(midi_dir), "--out", str(out_csv)])

    assert out_csv.read_text().strip().splitlines() == ["note_on,time,velocity"]
