import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("soundfile")
pytest.importorskip("colorama")
pytest.importorskip("hydra")
import soundfile as sf

from scripts import train_velocity


def _make_wav(path: Path) -> None:
    path.write_bytes(b"RIFF0000WAVEfmt ")


# ---------------------- tests using direct main() ----------------------- #


def test_missing_drums_dir_main(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    rc = train_velocity.main(
        [
            "augment-data",
            "--drums-dir",
            str(tmp_path / "missing"),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 1


def test_missing_wav_dir_main(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    rc = train_velocity.main(
        [
            "augment-data",
            "--wav-dir",
            str(tmp_path / "missing"),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 1


def test_auto_create_out_dir_main(tmp_path: Path) -> None:
    drums = tmp_path / "drums"
    drums.mkdir()
    _make_wav(drums / "a.wav")
    out_dir = tmp_path / "out"
    rc = train_velocity.main(
        [
            "augment-data",
            "--drums-dir",
            str(drums),
            "--out-dir",
            str(out_dir),
            "--snrs",
            "1",
            "--shifts",
            "2",
            "--rates",
            "1.0",
            "2.0",
        ]
    )
    assert rc == 0
    assert len(list(out_dir.rglob("*.wav"))) >= 1


def test_seed_reproducible_main(tmp_path: Path) -> None:
    drums = tmp_path / "drums"
    drums.mkdir()
    _make_wav(drums / "a.wav")
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    rc1 = train_velocity.main(
        [
            "--seed",
            "42",
            "augment-data",
            "--drums-dir",
            str(drums),
            "--out-dir",
            str(out1),
        ]
    )
    rc2 = train_velocity.main(
        [
            "--seed",
            "42",
            "augment-data",
            "--drums-dir",
            str(drums),
            "--out-dir",
            str(out2),
        ]
    )
    assert rc1 == 0 and rc2 == 0
    files1 = sorted(p.name for p in out1.glob("*.wav"))
    files2 = sorted(p.name for p in out2.glob("*.wav"))
    assert files1 == files2


def test_param_variations_main(tmp_path: Path) -> None:
    drums = tmp_path / "d"
    drums.mkdir()
    _make_wav(drums / "a.wav")
    out_dir = tmp_path / "out"
    rc = train_velocity.main(
        [
            "--seed",
            "1",
            "augment-data",
            "--drums-dir",
            str(drums),
            "--out-dir",
            str(out_dir),
            "--snrs",
            "0",
            "1",
            "--shifts",
            "0",
            "1",
            "--rates",
            "1.0",
            "1.5",
        ]
    )
    assert rc == 0
    files = sorted(out_dir.glob("*.wav"))
    assert len(files) == 8
    sizes = {f.stat().st_size for f in files}
    assert len(sizes) > 1


def test_out_dir_not_writable_main(tmp_path: Path) -> None:
    drums = tmp_path / "drums"
    drums.mkdir()
    _make_wav(drums / "a.wav")
    out_dir = tmp_path / "out"
    out_dir.write_text("not a dir")
    rc = train_velocity.main(
        [
            "augment-data",
            "--drums-dir",
            str(drums),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 1


# ---------------------- tests invoking CLI subprocess ---------------------- #


@pytest.fixture(autouse=True)
def set_pythonpath(monkeypatch):
    # Ensure the project root is on PYTHONPATH for subprocess calls
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    monkeypatch.setenv("PYTHONPATH", env["PYTHONPATH"])
    return env


def test_cli_augment_data(tmp_path: Path, set_pythonpath) -> None:
    src = tmp_path / "src"
    src.mkdir()
    sr = 8000
    sf.write(src / "a.wav", np.zeros(sr), sr)

    out_dir = tmp_path / "out"
    drums = tmp_path / "drums"
    drums.mkdir()

    res = subprocess.run(
        [
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
        ],
        capture_output=True,
        text=True,
        env=set_pythonpath,
    )
    assert res.returncode == 0
    files = list(out_dir.rglob("*.wav"))
    assert len(files) == 2


def test_cli_augment_data_errors(tmp_path: Path, set_pythonpath) -> None:
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
    res = subprocess.run(cmd, capture_output=True, text=True, env=set_pythonpath)
    assert res.returncode == 1
    assert "wav-dir does not exist" in res.stderr

    # missing drums-dir
    src = tmp_path / "src"
    src.mkdir()
    sr = 8000
    sf.write(src / "a.wav", np.zeros(sr), sr)
    cmd = [
        sys.executable,
        "scripts/train_velocity.py",
        "augment-data",
        "--wav-dir",
        str(src),
        "--out-dir",
        str(tmp_path / "out2"),
    ]
    res2 = subprocess.run(cmd, capture_output=True, text=True, env=set_pythonpath)
    assert res2.returncode == 1
    assert "drums-dir does not exist" in res2.stderr
