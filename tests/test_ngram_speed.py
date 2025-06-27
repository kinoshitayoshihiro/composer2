import time
from pathlib import Path

import pretty_midi

from utilities import groove_sampler_ngram


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(16):
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=36, start=i * 0.25, end=i * 0.25 + 0.1)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_sample_speed(tmp_path: Path) -> None:
    for i in range(10):
        _make_loop(tmp_path / f"{i}.mid")
    model = groove_sampler_ngram.train(tmp_path, order=3)
    t0 = time.perf_counter()
    groove_sampler_ngram.sample(model, bars=1000, seed=0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5
