import statistics
from pathlib import Path

import pretty_midi

from utilities import groove_sampler_ngram as gs


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    pattern = [
        (80, -10),
        (90, -5),
        (100, 5),
        (110, 10),
    ]
    for i in range(16):
        vel, micro = pattern[i % 4]
        start = i * 0.25 + micro / gs.PPQ
        inst.notes.append(
            pretty_midi.Note(velocity=vel, pitch=36, start=start, end=start + 0.05)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_humanize_sampling(tmp_path: Path) -> None:
    _make_loop(tmp_path / "a.mid")
    model = gs.train(tmp_path, order=1)
    events = gs.sample(model, bars=16, seed=0, humanize_vel=True, humanize_micro=True)
    assert len(events) >= 64
    vels = [e["velocity"] for e in events]
    micros = []
    step_ticks = gs.PPQ // 4
    for e in events:
        off_ticks = round(e["offset"] * gs.PPQ)
        step = round(e["offset"] * 4)
        micros.append(off_ticks - step * step_ticks)
    assert statistics.pstdev(vels) > 0
    assert statistics.pstdev(micros) > 0
