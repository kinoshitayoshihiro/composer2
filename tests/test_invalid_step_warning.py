import random
from pathlib import Path
import warnings

import pretty_midi
import pytest

from utilities import groove_sampler_ngram as gs


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_invalid_step_warning(tmp_path: Path, monkeypatch) -> None:
    _make_loop(tmp_path / "a.mid")
    model = gs.train(tmp_path, order=1)

    orig = gs._sample_next

    def fake_sample_next(history, model_arg, rng, **kwargs):
        if not hasattr(fake_sample_next, "called"):
            fake_sample_next.called = True
            return gs.RESOLUTION + 1, "kick"
        return 0, "kick"

    monkeypatch.setattr(gs, "_sample_next", fake_sample_next)

    with warnings.catch_warnings(record=True) as rec:
        events, history = gs.generate_bar([], model, rng=random.Random(0))

    assert len(rec) == 1
    assert issubclass(rec[0].category, RuntimeWarning)
    assert all(0 <= e["offset"] < 4 for e in events)
    assert all(0 <= s < gs.RESOLUTION for s, _ in history)

    monkeypatch.setattr(gs, "_sample_next", orig)
