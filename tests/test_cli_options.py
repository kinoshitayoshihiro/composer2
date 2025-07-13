import subprocess
import sys
from pathlib import Path
import pretty_midi


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(2):
        start = i * 0.5
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=36, start=start, end=start + 0.1)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_cli_beats_per_bar_and_sampling(tmp_path: Path) -> None:
    loops = tmp_path / "loops"
    loops.mkdir()
    _make_loop(loops / "a.mid")
    model_path = tmp_path / "model.pkl"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "utilities.groove_sampler_v2",
            "train",
            str(loops),
            "--auto-res",
            "--beats-per-bar",
            "8",
            "-o",
            str(model_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "utilities.groove_sampler_v2",
            "sample",
            str(model_path),
            "--top-k",
            "5",
            "--top-p",
            "0.8",
        ],
        check=True,
        capture_output=True,
    )

