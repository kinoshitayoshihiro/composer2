import time
from pathlib import Path

import pretty_midi
import pytest

from utilities import groove_sampler_ngram as gs


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(16):
        start = i * 0.25
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=start, end=start + 0.05))
    pm.instruments.append(inst)
    pm.write(str(path))


@pytest.mark.ci_perf
def test_perf_gate(tmp_path: Path) -> None:
    _make_loop(tmp_path / "loop.mid")
    start = time.perf_counter()
    model = gs.train(tmp_path, order=2)
    gs.sample(model, bars=64, temperature=0.0, top_k=1)
    elapsed = time.perf_counter() - start
    assert elapsed <= 0.80
