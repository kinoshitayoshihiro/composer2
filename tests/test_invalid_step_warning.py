import logging
import random
from pathlib import Path

import pretty_midi

from utilities import groove_sampler_ngram as gs


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_invalid_step_warning(tmp_path: Path, monkeypatch, caplog) -> None:
    _make_loop(tmp_path / "a.mid")
    model = gs.train(tmp_path, order=1)

    orig = gs._sample_next

    def fake_sample_next(history, model_arg, rng, **kwargs):
        if not hasattr(fake_sample_next, "called"):
            fake_sample_next.called = True
            return gs.RESOLUTION + 1, "kick"
        return orig(history, model_arg, rng, **kwargs)

    monkeypatch.setattr(gs, "_sample_next", fake_sample_next)

    prev = [(gs.RESOLUTION * 2, "snare")]
    with caplog.at_level(logging.DEBUG):
        events, history = gs.generate_bar(prev, model, rng=random.Random(0))

    assert "invalid step" in caplog.text
    assert len(history) == len(prev)

    monkeypatch.setattr(gs, "_sample_next", orig)
