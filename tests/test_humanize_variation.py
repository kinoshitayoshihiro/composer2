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


def _micro_values(events: list[gs.Event]) -> list[int]:
    step_ticks = gs.PPQ // 4
    values = []
    for e in events:
        off_ticks = round(float(e["offset"]) * gs.PPQ)
        step = round(float(e["offset"]) * 4)
        values.append(off_ticks - step * step_ticks)
    return values


def test_humanize_variation(tmp_path: Path) -> None:
    _make_loop(tmp_path / "a.mid")
    model = gs.train(tmp_path, order=1)

    ev_plain = gs.sample(model, bars=16, seed=0)
    assert statistics.pstdev([e["velocity"] for e in ev_plain]) == 0

    ev_vel = gs.sample(model, bars=16, seed=0, humanize_vel=True)
    assert statistics.pstdev([e["velocity"] for e in ev_vel]) > 0

    ev_micro = gs.sample(model, bars=16, seed=0, humanize_micro=True)
    assert statistics.pstdev(_micro_values(ev_micro)) > 0
