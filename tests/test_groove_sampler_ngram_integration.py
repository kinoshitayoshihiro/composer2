from pathlib import Path
from unittest import mock

import pretty_midi

from utilities import groove_sampler_ngram, loop_ingest


def _make_midi(path: Path, pitches: list[int]) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i, p in enumerate(pitches):
        start = i * 0.25
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=p, start=start, end=start + 0.05)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def _fake_scan_wav(_path: Path, resolution: int, ppq: int) -> loop_ingest.LoopEntry:
    tokens = [(i, "perc", 100, 0) for i in range(4)]
    return {
        "file": _path.name,
        "tokens": tokens,
        "tempo_bpm": 120.0,
        "bar_beats": 4,
        "section": "unknown",
        "heat_bin": 0,
        "intensity": "mid",
    }


def test_train_sample_mixed(tmp_path: Path, monkeypatch: mock.MagicMock) -> None:
    _make_midi(tmp_path / "a.mid", [36, 38, 42, 46])
    _make_midi(tmp_path / "b.mid", [36, 38, 42, 46])
    wav_path = tmp_path / "c.wav"
    wav_path.write_bytes(b"\x00\x00")
    monkeypatch.setattr(loop_ingest, "_scan_wav", _fake_scan_wav)
    model = groove_sampler_ngram.train(tmp_path, ext="mid,wav", order="auto")
    events = groove_sampler_ngram.sample(model, bars=1, seed=0)
    assert events
