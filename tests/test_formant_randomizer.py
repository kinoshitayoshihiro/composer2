from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("librosa")
pytest.importorskip("soundfile")
from click.testing import CliRunner

from modular_composer.cli import cli


def test_randomize_stem_cli(tmp_path: Path) -> None:
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = np.sin(2 * np.pi * 440 * t)
    inp = tmp_path / "in.wav"
    sf.write(inp, y, sr)

    out = tmp_path / "out.wav"
    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "randomize-stem",
            "--input",
            str(inp),
            "--cents",
            "50",
            "--formant",
            "0",
            "-o",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    assert out.exists()
    data, rate = sf.read(out)
    assert rate == sr
    assert pytest.approx(len(data) / rate, 0.001) == pytest.approx(len(y) / sr, 0.001)


def test_randomize_stem_help() -> None:
    runner = CliRunner()
    res = runner.invoke(cli, ["randomize-stem", "--help"])
    assert res.exit_code == 0
    out = res.output.lower()
    assert "--input" in out
    assert "--cents" in out
    assert "--formant" in out
    assert "--out" in out


def test_randomize_stem_missing_args() -> None:
    runner = CliRunner()
    res = runner.invoke(cli, ["randomize-stem"])
    assert res.exit_code != 0
