import random
from pathlib import Path

import pretty_midi

from utilities import groove_sampler_ngram as gs


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(4):
        start = i * 0.25
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=36, start=start, end=start + 0.05)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_sampler_humanize_flag(tmp_path: Path) -> None:
    _make_loop(tmp_path / "a.mid")
    model = gs.train(tmp_path, order=1)
    ev = gs.sample(model, bars=1, seed=0, humanize_vel=True, humanize_micro=True)
    assert ev
    for e in ev:
        assert 1 <= e["velocity"] <= 127
        micro = round(e["offset"] * gs.PPQ) % (gs.PPQ // 4)
        assert -45 <= micro <= 45
