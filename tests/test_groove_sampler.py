import pickle
from pathlib import Path

import pretty_midi
import pytest

from utilities import groove_sampler_ngram
from utilities.drum_map_registry import GM_DRUM_MAP


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(4):
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=36, start=i * 0.5, end=i * 0.5 + 0.1)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_sample_non_empty(tmp_path: Path) -> None:
    for i in range(5):
        _make_loop(tmp_path / f"{i}.mid")
    model = groove_sampler_ngram.train(tmp_path, ext="midi", order=3)
    events = groove_sampler_ngram.sample(model, bars=1, seed=0)
    assert events

    assert model["version"] == groove_sampler_ngram.VERSION
    assert model["resolution"] == groove_sampler_ngram.RESOLUTION


def test_perc_label_exists() -> None:
    assert "perc" in GM_DRUM_MAP


def test_train_wav(tmp_path: Path) -> None:
    pytest.importorskip("librosa")
    import numpy as np
    import soundfile as sf

    wav = tmp_path / "loop.wav"
    sf.write(wav, np.zeros(1000), 22050)
    with pytest.raises(ValueError):
        groove_sampler_ngram.train(tmp_path, ext="wav", order=2)


def test_load_mismatch(tmp_path: Path) -> None:
    _make_loop(tmp_path / "a.mid")
    model = groove_sampler_ngram.train(tmp_path, ext="midi", order=2)
    path = tmp_path / "m.pkl"
    groove_sampler_ngram.save(model, path)
    with path.open("rb") as fh:
        data = pickle.load(fh)
    data["resolution"] = groove_sampler_ngram.RESOLUTION + 1
    with path.open("wb") as fh:
        pickle.dump(data, fh)
    with pytest.raises(RuntimeError):
        groove_sampler_ngram.load(path)


def _make_two_bar_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(8):
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=36, start=i * 0.5, end=i * 0.5 + 0.1)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_multi_bar_training(tmp_path: Path) -> None:
    _make_two_bar_loop(tmp_path / "two.mid")
    model = groove_sampler_ngram.train(tmp_path, ext="midi", order=2)
    steps = {state[0] for state in model["freq"][0][()].keys()}
    assert steps and all(0 <= s < groove_sampler_ngram.RESOLUTION for s in steps)
    events = groove_sampler_ngram.sample(model, bars=1, seed=1)
    assert len(events) <= groove_sampler_ngram.RESOLUTION
    assert all(ev["offset"] < 4.0 for ev in events)

