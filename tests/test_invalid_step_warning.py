import random
from pathlib import Path
import pretty_midi
import pytest

from utilities import groove_sampler_ngram as gs


def _loop(p: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    pm.instruments.append(inst)
    pm.write(str(p))


def test_invalid_step_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _loop(tmp_path / "a.mid")
    model = gs.train(tmp_path, order=2)
    called = False

    def fake(*args, **kwargs):
        nonlocal called
        if not called:
            called = True
            return gs.RESOLUTION + 5, "kick"
        return 0, "kick"

    monkeypatch.setattr(gs, "_sample_next", fake)
    with pytest.warns(RuntimeWarning):
        ev, hist = gs.generate_bar(None, model, rng=random.Random(0))
    assert all(e["offset"] >= 0 for e in ev)
    assert hist and hist[0][0] < gs.RESOLUTION
