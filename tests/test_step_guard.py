import random
from pathlib import Path

import pretty_midi
import pytest

from utilities import groove_sampler_ngram as gs


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_step_out_of_range_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_loop(tmp_path / "a.mid")
    model = gs.train(tmp_path, order=1)
    called = False

    def fake_sample_next(*args, **kwargs):
        nonlocal called
        if not called:
            called = True
            return 20, "kick"
        return 0, "kick"

    monkeypatch.setattr(gs, "_sample_next", fake_sample_next)
    orig_res = gs.RESOLUTION
    monkeypatch.setattr(gs, "RESOLUTION", 1)
    with pytest.warns(RuntimeWarning):
        events, hist = gs.generate_bar(None, model, rng=random.Random(0))
    monkeypatch.setattr(gs, "RESOLUTION", orig_res)
    assert len(events) == 1
    assert round(events[0]["offset"] * 4) == 0
