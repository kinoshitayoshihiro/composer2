import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def test_cli_augment_data(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    sr = 8000
    sf.write(src / "a.wav", np.zeros(sr), sr)

    out_dir = tmp_path / "out"
    drums = tmp_path / "drums"
    drums.mkdir()

    cmd = [
        sys.executable,
        "scripts/train_velocity.py",
        "augment-data",
        "--wav-dir",
        str(src),
        "--out-dir",
        str(out_dir),
        "--drums-dir",
        str(drums),
        "--shifts",
        "0,1",
        "--rates",
        "1.0",
        "--snrs",
        "20",
    ]
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == 0
    files = list(out_dir.rglob("*.wav"))
    assert len(files) == 2


def test_cli_augment_data_errors(tmp_path: Path) -> None:
    # missing wav-dir
    cmd = [
        sys.executable,
        "scripts/train_velocity.py",
        "augment-data",
        "--wav-dir",
        str(tmp_path / "missing"),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == 1
    assert "wav-dir does not exist" in res.stderr

    src = tmp_path / "src"
    src.mkdir()
    sr = 8000
    sf.write(src / "a.wav", np.zeros(sr), sr)

    # out-dir auto created
    out_dir = tmp_path / "missing_out"
    cmd = [
        sys.executable,
        "scripts/train_velocity.py",
        "augment-data",
        "--wav-dir",
        str(src),
        "--out-dir",
        str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == 1

