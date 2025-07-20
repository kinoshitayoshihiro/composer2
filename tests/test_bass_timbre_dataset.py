from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import pretty_midi
import torch

from utilities.bass_timbre_dataset import BassTimbreDataset


def _write_sine(path: Path, freq: float, sr: int = 24000, dur: float = 0.9) -> None:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(path, y, sr)


def _write_midi(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for i, start in enumerate([0.0, 0.3, 0.6]):
        note = pretty_midi.Note(velocity=100, pitch=60 + i, start=start, end=start + 0.1)
        inst.notes.append(note)
    pm.instruments.append(inst)
    pm.write(str(path))


def test_bass_timbre_dataset(tmp_path: Path) -> None:
    root = tmp_path / "paired"
    root.mkdir()
    _write_sine(root / "0001__wood.wav", 440.0)
    _write_sine(root / "0001__synth.wav", 880.0)
    _write_midi(root / "0001__wood.mid")

    ds = BassTimbreDataset(root, src_suffix="wood", tgt_suffixes=["synth"], cache=False, max_len=50)
    assert len(ds) == 1
    item = ds[0]
    assert item["src"].shape[0] == 128
    assert abs(item["src"].shape[1] - 43) <= 5

    ds.write_cache()
    ds_cached = BassTimbreDataset(root, src_suffix="wood", tgt_suffixes=["synth"], cache=True, max_len=50)
    cached = ds_cached[0]
    assert torch.allclose(item["src"], cached["src"])
    assert torch.allclose(item["tgt"], cached["tgt"])
