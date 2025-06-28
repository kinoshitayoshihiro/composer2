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


def test_error_message_contains_context(tmp_path: Path) -> None:
    _make_loop(tmp_path / "a.mid")
    model = gs.train(tmp_path, order=1)
    model["prob"][0] = {}
    with pytest.raises(RuntimeError) as exc:
        gs.generate_bar(None, model, rng=random.Random(0))
    msg = str(exc.value)
    assert "context" in msg and "aux" in msg

