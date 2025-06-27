from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
from click.testing import CliRunner

from utilities.loop_ingest import cli, load_cache, save_cache, scan_loops

try:
    import librosa  # noqa: F401
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False


def _make_midi(path: Path, pitch: int) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(8):
        start = i * 0.5
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + 0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def _make_wav(path: Path) -> None:
    sr = 16000
    y = np.zeros(sr, dtype=np.float32)
    y[sr // 2] = 1.0
    sf.write(path, y, sr)


def test_scan_loops_and_cache(tmp_path: Path) -> None:
    _make_midi(tmp_path / "a.mid", 36)
    _make_midi(tmp_path / "b.mid", 38)
    exts = ["mid"]
    if HAVE_LIBROSA:
        _make_wav(tmp_path / "c.wav")
        exts.append("wav")
    data = scan_loops(tmp_path, exts=exts)
    expected = 16 + (1 if HAVE_LIBROSA else 0)
    assert sum(len(d["tokens"]) for d in data) >= expected
    assert all("tempo_bpm" in d and "bar_beats" in d for d in data)
    cache = tmp_path / "cache.pkl"
    save_cache(data, cache, ppq=480, resolution=16)
    loaded = load_cache(cache)
    assert loaded == data

    runner = CliRunner()
    result = runner.invoke(cli, ["info", str(cache)])
    assert result.exit_code == 0
    assert "files:" in result.output


def test_cli_warns_missing_librosa(tmp_path: Path) -> None:
    if HAVE_LIBROSA:
        pytest.skip("requires missing librosa")
    _make_wav(tmp_path / "a.wav")
    runner = CliRunner()
    out = tmp_path / "cache.pkl"
    result = runner.invoke(
        cli,
        ["scan", str(tmp_path), "--ext", "wav", "--out", str(out), "--no-progress"],
    )
    assert result.exit_code == 0
    assert "Install it with pip install librosa" in result.output
