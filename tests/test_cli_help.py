import subprocess
import sys
from pathlib import Path


def test_main_help() -> None:
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    res = subprocess.run(
        [sys.executable, "scripts/train_velocity.py", "-h"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0
    assert "augment-data" in res.stdout
    assert "--augment" in res.stdout


def test_augment_help() -> None:
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    res = subprocess.run(
        [sys.executable, "scripts/train_velocity.py", "augment-data", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0
    for flag in ["--wav-dir", "--out-dir", "--drums-dir", "--shifts", "--rates", "--snrs", "--progress"]:
        assert flag in res.stdout
