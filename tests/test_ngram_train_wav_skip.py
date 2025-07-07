from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from utilities import groove_sampler_ngram, loop_ingest

pytestmark = pytest.mark.requires_audio


def _make_midi(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def _make_wav(path: Path) -> None:
    sr = 16000
    y = np.zeros(sr, dtype=np.float32)
    sf.write(path, y, sr)


def test_train_skip_wav(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(loop_ingest, "HAVE_LIBROSA", False)
    _make_midi(tmp_path / "a.mid")
    _make_wav(tmp_path / "b.wav")
    model = groove_sampler_ngram.train(tmp_path, ext="mid,wav", order=1)
    assert model["order"] == 1

