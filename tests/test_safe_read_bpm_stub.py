import pytest

from utilities import groove_sampler_v2 as module


class _DummyPM:
    def __init__(self, bpm: float) -> None:
        self._bpm = bpm

    def get_tempo_changes(self):
        return [0.0], [self._bpm]


def test_safe_read_bpm_uses_first_bpm_from_stub():
    pm = _DummyPM(134.0)
    bpm = module._safe_read_bpm(pm, default_bpm=120.0, fold_halves=False)
    assert bpm == pytest.approx(134.0)
    assert module._safe_read_bpm.last_source == "pretty_midi"
