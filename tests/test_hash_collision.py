import warnings
from pathlib import Path
from unittest import mock

import pretty_midi

from utilities import groove_sampler_ngram


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_hash_collision(monkeypatch: mock.MagicMock, tmp_path: Path) -> None:
    _make_loop(tmp_path / "a.mid")
    _make_loop(tmp_path / "b.mid")

    def const_hash(_data: bytes, _sha1: bool, _bits: int = 64) -> int:
        if not _sha1:
            return 1
        if _bits == 64:
            return 1
        return 2

    monkeypatch.setattr(groove_sampler_ngram, "_hash_bytes", const_hash)
    with warnings.catch_warnings(record=True) as rec:
        model = groove_sampler_ngram.train(tmp_path, order=1)
    assert model
    assert any("collision" in str(w.message).lower() for w in rec)

