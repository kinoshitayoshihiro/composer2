import pytest
from pathlib import Path
from unittest import mock

from utilities import groove_sampler_ngram
from tests.test_hash_collision import _make_loop


def test_hash_collision_fail(monkeypatch: mock.MagicMock, tmp_path: Path) -> None:
    _make_loop(tmp_path / "a.mid")
    _make_loop(tmp_path / "b.mid")

    def const_hash(_data: bytes, _sha1: bool, _bits: int = 64) -> int:
        return 1

    monkeypatch.setattr(groove_sampler_ngram, "_hash_bytes", const_hash)
    with pytest.raises(RuntimeError):
        groove_sampler_ngram.train(tmp_path, order=1)

