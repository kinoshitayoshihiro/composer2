from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("soundfile")
pytest.importorskip("pretty_midi")
pytest.importorskip("librosa")
pytest.importorskip("music21")

import importlib

import numpy as np

from utilities.bass_timbre_dataset import BassTimbreDataset, _normalize

assert importlib.import_module("yaml")


def _write_sine(path: Path, freq: float, sr: int = 24000, dur: float = 0.9) -> None:
    import soundfile as sf  # type: ignore[import]

    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(path, y, sr)


def _write_midi(path: Path) -> None:
    import pretty_midi  # type: ignore[import]

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for i, start in enumerate([0.0, 0.3, 0.6]):
        note = pretty_midi.Note(velocity=100, pitch=60 + i, start=start, end=start + 0.1)
        inst.notes.append(note)
    pm.instruments.append(inst)
    pm.write(str(path))



@pytest.mark.parametrize("cache", [True, False])
def test_bass_timbre_dataset(tmp_path: Path, cache: bool) -> None:
    import subprocess
    import sys

    import torch

    root = tmp_path / "paired"
    root.mkdir()
    _write_sine(root / "0001__wood.wav", 440.0)
    _write_sine(root / "0001__synth.wav", 880.0)
    _write_midi(root / "0001__wood.mid")

    ds = BassTimbreDataset(root, src_suffix="wood", tgt_suffixes=["synth"], cache=cache, max_len=50)
    assert len(ds) == 1
    assert np.allclose(_normalize(np.zeros((128, 10))), np.full((128, 10), 0.5, dtype=np.float32))
    item = ds[0]
    assert item["src"].shape[0] == 128
    assert abs(item["src"].shape[1] - 43) <= 5

    ds.write_cache()
    ds_cached = BassTimbreDataset(root, src_suffix="wood", tgt_suffixes=["synth"], cache=True, max_len=50)
    cached = ds_cached[0]
    assert torch.allclose(item["src"], cached["src"])
    assert torch.allclose(item["tgt"], cached["tgt"])

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(Path("scripts/extract_timbre_dataset.py")),
        "--in_dir",
        str(root),
        "--out_dir",
        str(out_dir),
        "--src",
        "wood",
        "--tgt",
        "synth",
        "--max_len",
        "50",
        "--num_workers",
        "1",
    ]
    if cache:
        cmd.append("--no_cache")
    pytest.importorskip("librosa")
    subprocess.check_call(cmd)

    manifest = out_dir / "dataset.csv"
    assert manifest.exists()
    with manifest.open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["id", "src", "tgt", "n_frames"]
    assert len(rows) == len(ds.pairs) + 1

