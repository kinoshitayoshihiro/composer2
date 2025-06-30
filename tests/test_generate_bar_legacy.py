from pathlib import Path

import pretty_midi
import pytest

from utilities import groove_sampler_ngram as gs


def _loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_generate_bar_legacy(tmp_path: Path) -> None:
    _loop(tmp_path / "l.mid")
    model = gs.train(tmp_path, order=1)
    hist: list[gs.State] = []
    with pytest.deprecated_call():
        events, history = gs.generate_bar_legacy(hist, model)
    assert events
    assert history is hist
