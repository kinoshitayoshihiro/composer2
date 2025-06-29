from pathlib import Path

import pretty_midi

from utilities import groove_sampler_ngram as gs


def _mk_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(4):
        start = i * 0.25
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=start, end=start + 0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def _micro_from_offset(off: float) -> int:
    step_size = 4 / gs.RESOLUTION
    idx = round(off / step_size)
    grid = idx * step_size
    return round((off - grid) * gs.PPQ)


def test_humanize_ranges(tmp_path: Path) -> None:
    _mk_loop(tmp_path / "a.mid")
    model = gs.train(tmp_path, order=1)
    ev = gs.sample(model, bars=16, humanize_vel=True, humanize_micro=True)
    assert len(ev) >= 64
    for e in ev:
        assert 1 <= int(e["velocity"]) <= 127
        micro = _micro_from_offset(float(e["offset"]))
        assert abs(micro) <= 30

